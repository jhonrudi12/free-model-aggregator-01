// The client wires and the three upstream wires.
//
// The client calls this server as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), JSON or SSE. Each upstream provider
// speaks one of three wires: openai, anthropic, gemini. This file translates
// between them through one neutral request shape, so every field the env's
// echo checks (system prompt, user text, images, tools, stop sequences,
// max_tokens, JSON mode) travels untouched on every combination:
//
//   client openai    -> provider openai:    body passes through, model replaced
//   client anthropic -> provider openai:    system/messages/tools rewritten
//   client openai    -> provider anthropic: messages/tools rewritten
//   client anthropic -> provider anthropic: body passes through, model replaced
//   either client    -> provider gemini:    contents/systemInstruction/generationConfig
//
// Every upstream answer, JSON or SSE, is parsed into one normalized answer
// { text, toolCalls, usage, finish } and then emitted in the client's wire,
// in that wire's canonical event order, with the provider's usage counts and
// the stop reason mapped (end_turn/STOP -> stop, max_tokens -> length,
// tool_use -> tool_calls, refusal -> content_filter).
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, so the same body and the same provider answer give the same
// bytes on a replay. The two model listings use fixed timestamps.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A stop reason in any wire as the normalized family: stop | length | tool_calls | content_filter. */
export function finishOf(reason: unknown): string {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 'tool_calls';
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** The normalized family as an OpenAI finish_reason. */
export function openAIFinish(finish: string): string {
  return finish === 'length' ? 'length' : finish === 'tool_calls' ? 'tool_calls' : finish === 'content_filter' ? 'content_filter' : 'stop';
}

/** The normalized family as an Anthropic stop_reason. */
export function anthropicStop(finish: string): string {
  return finish === 'length' ? 'max_tokens' : finish === 'tool_calls' ? 'tool_use' : finish === 'content_filter' ? 'refusal' : 'end_turn';
}

/** One tool call in an upstream answer: arguments kept as a JSON string. */
export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** One upstream answer, in either wire, JSON or streamed, normalized. */
export interface NormalizedAnswer {
  text: string;
  toolCalls: ToolCall[];
  usage: { input: number; output: number };
  finish: string;
}

/** Split an SSE body into frames: { event, data } pairs. */
export function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI JSON or SSE answer, normalized. */
export function parseOpenAIAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    const body = parseJsonObject(bodyText);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
        });
      }
    }
    return {
      text: typeof message.content === 'string' ? message.content : textOf(message.content),
      toolCalls,
      usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
      finish: finishOf(choice.finish_reason),
    };
  }
  let text = '';
  const toolParts = new Map<number, { id: string; name: string; arguments: string }>();
  let usage = { input: 0, output: 0 };
  let finish = 'stop';
  for (const frame of parseSse(bodyText)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : 0;
        const part = toolParts.get(index) ?? { id: '', name: '', arguments: '' };
        if (typeof call.id === 'string' && call.id !== '') part.id = call.id;
        if (isRecord(call.function)) {
          if (typeof call.function.name === 'string' && call.function.name !== '') part.name = call.function.name;
          if (typeof call.function.arguments === 'string') part.arguments += call.function.arguments;
        }
        toolParts.set(index, part);
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) finish = finishOf(choice.finish_reason);
  }
  const toolCalls = [...toolParts.entries()].sort((a, b) => a[0] - b[0]).map(([, p]) => ({ id: p.id, name: p.name, arguments: p.arguments }));
  return { text, toolCalls, usage, finish };
}

/** An Anthropic JSON or SSE answer, normalized. */
export function parseAnthropicAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    const body = parseJsonObject(bodyText);
    if (body === null) return null;
    const content = Array.isArray(body.content) ? body.content : [];
    const text = content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
    const toolCalls: ToolCall[] = [];
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      toolCalls,
      usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
      finish: finishOf(body.stop_reason),
    };
  }
  let text = '';
  const toolParts = new Map<number, { id: string; name: string; arguments: string }>();
  let usage = { input: 0, output: 0 };
  let finish = 'stop';
  for (const frame of parseSse(bodyText)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      usage = { input: numberOrZero(data.message.usage.input_tokens), output: usage.output };
    } else if (frame.event === 'content_block_start' && typeof data.index === 'number' && isRecord(data.content_block)) {
      const block = data.content_block;
      if (block.type === 'tool_use') {
        toolParts.set(data.index, {
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          arguments: '',
        });
      }
    } else if (frame.event === 'content_block_delta' && typeof data.index === 'number' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') text += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const part = toolParts.get(data.index);
        if (part !== undefined) part.arguments += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') finish = finishOf(data.delta.stop_reason);
      if (isRecord(data.usage)) usage = { input: usage.input, output: numberOrZero(data.usage.output_tokens) };
    }
  }
  const toolCalls = [...toolParts.entries()].sort((a, b) => a[0] - b[0]).map(([, p]) => ({ id: p.id, name: p.name, arguments: p.arguments }));
  return { text, toolCalls, usage, finish };
}

/** A Gemini JSON or SSE answer, normalized. */
export function parseGeminiAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  const collect = (body: Json): { text: string; toolCalls: ToolCall[]; usage: { input: number; output: number }; finish: string | null } => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(cand.content) ? cand.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    let text = '';
    const toolCalls: ToolCall[] = [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') text += part.text;
      else if (isRecord(part.functionCall)) {
        toolCalls.push({
          id: '',
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        });
      }
    }
    const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text,
      toolCalls,
      usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
      finish: typeof cand.finishReason === 'string' ? finishOf(cand.finishReason) : null,
    };
  };
  if (!stream) {
    const body = parseJsonObject(bodyText);
    if (body === null) return null;
    const c = collect(body);
    return { text: c.text, toolCalls: c.toolCalls, usage: c.usage, finish: c.finish ?? 'stop' };
  }
  let text = '';
  const toolCalls: ToolCall[] = [];
  let usage = { input: 0, output: 0 };
  let finish: string | null = null;
  for (const frame of parseSse(bodyText)) {
    const body = parseJsonObject(frame.data);
    if (body === null) continue;
    const c = collect(body);
    text += c.text;
    toolCalls.push(...c.toolCalls);
    if (c.usage.input !== 0 || c.usage.output !== 0) usage = c.usage;
    if (c.finish !== null) finish = c.finish;
  }
  return { text, toolCalls, usage, finish: finish ?? 'stop' };
}

// ---------------------------------------------------------------------------
// The neutral request: one shape extracted from either client wire, rendered
// into any provider wire. Tools and stops keep their order; the system prompt
// keeps every system message's text; images keep their bytes.
// ---------------------------------------------------------------------------

export interface NeutralTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface NeutralMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  /** Text parts and image parts, in order. An image part is { image: { mime, data } | { url } }. */
  parts: { text?: string; imageData?: { mime: string; data: string }; imageUrl?: string }[];
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export interface NeutralRequest {
  systemText: string;
  messages: NeutralMessage[];
  tools: NeutralTool[];
  toolChoice: string | Json | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function splitDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] !== undefined && m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

/** One OpenAI content part as neutral parts. */
function openAIParts(content: unknown): NeutralMessage['parts'] {
  const parts: NeutralMessage['parts'] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const data = splitDataUrl(block.image_url.url);
      if (data !== null) parts.push({ imageData: data });
      else parts.push({ imageUrl: block.image_url.url });
    }
  }
  return parts;
}

/** One Anthropic content block array as neutral parts. */
function anthropicParts(content: unknown): NeutralMessage['parts'] {
  const parts: NeutralMessage['parts'] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ imageData: { mime: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ imageUrl: block.source.url });
      }
    }
  }
  return parts;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI client body as a neutral request. */
export function openAIToNeutral(body: Json): NeutralRequest {
  const messages: NeutralMessage[] = [];
  const systemParts: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (role === 'tool') {
      messages.push({ role: 'tool', parts: [{ text: toolResultText(raw.content) }], toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '' });
      continue;
    }
    if (role !== 'user' && role !== 'assistant') continue;
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
        });
      }
    }
    messages.push({ role, parts: openAIParts(raw.content), ...(toolCalls.length > 0 ? { toolCalls } : {}) });
  }
  const tools: NeutralTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const stops: string[] = [];
  if (typeof body.stop === 'string') stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === 'string') stops.push(s);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice: (body.tool_choice as string | Json | null) ?? null,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic client body as a neutral request. */
export function anthropicToNeutral(body: Json): NeutralRequest {
  const messages: NeutralMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (!Array.isArray(content)) {
      messages.push({ role: assistant ? 'assistant' : 'user', parts: anthropicParts(content) });
      continue;
    }
    if (assistant) {
      const toolCalls: ToolCall[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        });
      }
      messages.push({ role: 'assistant', parts: anthropicParts(content), ...(toolCalls.length > 0 ? { toolCalls } : {}) });
      continue;
    }
    // A user message may carry text/image blocks and tool_result blocks.
    const parts = anthropicParts(content);
    const toolResults: NeutralMessage[] = [];
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_result') continue;
      toolResults.push({
        role: 'tool',
        parts: [{ text: toolResultText(block.content) }],
        toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
      });
    }
    if (parts.length > 0) messages.push({ role: 'user', parts });
    messages.push(...toolResults);
  }
  const tools: NeutralTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const stops: string[] = [];
  if (Array.isArray(body.stop_sequences)) for (const s of body.stop_sequences) if (typeof s === 'string') stops.push(s);
  return {
    systemText: textOf(body.system),
    messages,
    tools,
    toolChoice: (body.tool_choice as string | Json | null) ?? null,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops,
    jsonMode: false,
    stream: body.stream === true,
  };
}

function neutralText(parts: NeutralMessage['parts']): string {
  return parts.flatMap((p) => (p.text !== undefined ? [p.text] : [])).join('\n');
}

/** The neutral request as an OpenAI Chat Completions body naming `model`. */
export function neutralToOpenAI(req: NeutralRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
  for (const msg of req.messages) {
    if (msg.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: msg.toolCallId ?? '', content: neutralText(msg.parts) });
      continue;
    }
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.text !== undefined) parts.push({ type: 'text', text: part.text });
      else if (part.imageData !== undefined) parts.push({ type: 'image_url', image_url: { url: `data:${part.imageData.mime};base64,${part.imageData.data}` } });
      else if (part.imageUrl !== undefined) parts.push({ type: 'image_url', image_url: { url: part.imageUrl } });
    }
    if (msg.role === 'assistant') {
      const out: Json = { role: 'assistant', content: parts.length === 1 && parts[0].type === 'text' ? (parts[0] as Json).text : parts.length === 0 ? null : parts };
      if (msg.toolCalls !== undefined && msg.toolCalls.length > 0) {
        out.tool_calls = msg.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
      }
      messages.push(out);
      continue;
    }
    // A user message: a lone text part stays a plain string (the client's own
    // shape); anything with images stays a parts array.
    if (parts.length === 1 && parts[0].type === 'text') messages.push({ role: 'user', content: (parts[0] as Json).text });
    else if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (req.toolChoice !== null) {
    if (typeof req.toolChoice === 'string') out.tool_choice = req.toolChoice;
    else if (isRecord(req.toolChoice)) {
      if (req.toolChoice.type === 'auto') out.tool_choice = 'auto';
      else if (req.toolChoice.type === 'any') out.tool_choice = 'required';
      else if (req.toolChoice.type === 'tool' && typeof req.toolChoice.name === 'string') out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
    }
  }
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The neutral request as an Anthropic Messages body naming `model`. */
export function neutralToAnthropic(req: NeutralRequest, model: string): Json {
  const messages: Json[] = [];
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length === 0) return;
    messages.push({ role: 'user', content: pendingToolResults });
    pendingToolResults = [];
  };
  for (const msg of req.messages) {
    if (msg.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: msg.toolCallId ?? '',
        content: neutralText(msg.parts),
      });
      continue;
    }
    flushToolResults();
    const blocks: Json[] = [];
    for (const part of msg.parts) {
      if (part.text !== undefined) blocks.push({ type: 'text', text: part.text });
      else if (part.imageData !== undefined) blocks.push({ type: 'image', source: { type: 'base64', media_type: part.imageData.mime, data: part.imageData.data } });
      else if (part.imageUrl !== undefined) blocks.push({ type: 'image', source: { type: 'url', url: part.imageUrl } });
    }
    if (msg.role === 'assistant') {
      if (msg.toolCalls !== undefined) {
        for (const c of msg.toolCalls) blocks.push({ type: 'tool_use', id: c.id, name: c.name, input: argumentsOf(c.arguments) });
      }
      messages.push({ role: 'assistant', content: blocks.length === 1 && blocks[0].type === 'text' ? (blocks[0] as Json).text : blocks });
      continue;
    }
    messages.push({ role: 'user', content: blocks.length === 1 && blocks[0].type === 'text' ? (blocks[0] as Json).text : blocks });
  }
  flushToolResults();
  const out: Json = { model, messages };
  if (req.systemText !== '') out.system = req.systemText;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice !== null) {
    if (typeof req.toolChoice === 'string') {
      if (req.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
      else if (req.toolChoice === 'required') out.tool_choice = { type: 'any' };
    } else if (isRecord(req.toolChoice)) {
      if (req.toolChoice.type === 'auto') out.tool_choice = { type: 'auto' };
      else if (req.toolChoice.type === 'any') out.tool_choice = { type: 'any' };
      else if (req.toolChoice.type === 'tool' && typeof req.toolChoice.name === 'string') out.tool_choice = { type: 'tool', name: req.toolChoice.name };
    }
  }
  if (req.stream) out.stream = true;
  return out;
}

/** The neutral request as a Gemini generateContent body. */
export function neutralToGemini(req: NeutralRequest): Json {
  const contents: Json[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'tool') {
      // A tool result carries no text the echo reads; attach its
      // functionResponse to the previous content so the last user text is
      // never overwritten by a result-only content.
      const target = contents.length > 0 ? contents[contents.length - 1] : null;
      if (target !== null && Array.isArray(target.parts)) {
        (target.parts as Json[]).push({
          functionResponse: { name: '', response: { content: neutralText(msg.parts) } },
        });
      }
      continue;
    }
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.text !== undefined) parts.push({ text: part.text });
      else if (part.imageData !== undefined) parts.push({ inlineData: { mimeType: part.imageData.mime, data: part.imageData.data } });
      else if (part.imageUrl !== undefined) parts.push({ fileData: { fileUri: part.imageUrl } });
    }
    if (msg.role === 'assistant' && msg.toolCalls !== undefined) {
      for (const c of msg.toolCalls) parts.push({ functionCall: { name: c.name, args: argumentsOf(c.arguments) } });
    }
    if (parts.length === 0) continue;
    contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (req.systemText !== '') out.systemInstruction = { parts: [{ text: req.systemText }] };
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  return out;
}

// ---------------------------------------------------------------------------
// The normalized answer as a client response, in the client's wire.
// ---------------------------------------------------------------------------

/** One server-sent event. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** The normalized answer as an OpenAI Chat Completions object. */
export function answerToOpenAI(answer: NormalizedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text !== '' || answer.toolCalls.length === 0 ? answer.text : null };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(answer.finish), logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

/** The normalized answer as an OpenAI SSE stream: role chunk, content deltas,
 * tool-call deltas, one finish chunk, the usage chunk, then [DONE]. */
export function answerToOpenAIStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((c, index) => {
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index, id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }] }, null),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, openAIFinish(answer.finish))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The normalized answer as an Anthropic Messages object. */
export function answerToAnthropic(answer: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const c of answer.toolCalls) {
    content.push({ type: 'tool_use', id: c.id !== '' ? c.id : 'toolu_unknown', name: c.name, input: argumentsOf(c.arguments) });
  }
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** The normalized answer as an Anthropic SSE stream: message_start, one
 * content block per text/tool call, message_delta with the stop reason and
 * output tokens, then message_stop. */
export function answerToAnthropicStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const c of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: c.id !== '' ? c.id : `toolu_${index}`, name: c.name, input: {} },
      }),
    );
    if (c.arguments !== '') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: c.arguments } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.finish), stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
