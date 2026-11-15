// The three wires: translating requests and answers between the OpenAI
// Chat Completions API, the Anthropic Messages API and the Gemini
// generateContent API.
//
// The one rule that governs every translation: the provider answers with
// `ANSWER ECHO{...}`, an echo of the request it parsed, and the env's
// client compares that echo with the echo of the request it sent. So a
// translation must preserve, exactly, the system text, the last user
// message's text, the tool names in order, max_tokens, the stop sequences,
// the image bytes and the JSON-mode flag. Anything else (temperature,
// tool_choice) rides along for faithfulness but is never checked.
//
// A provider's answer is always plain text with a family-0 stop reason
// (OpenAI `stop`, Anthropic `end_turn`, Gemini `STOP`): no tool calls ever
// come back, so no translation invents one. Usage is always the provider's
// own numbers, remapped to the client's field names.

import { isRecord } from './brief.ts';
import type { JsonObject } from './brief.ts';

/** A provider's answer, reduced to what the client wire needs. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

const FIXED_CREATED = 1767225600;

/** The text of a string, or of a block array's text blocks joined with
 *  newlines; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** One SSE frame: `event: <name>` when named, then the data line, then the
 *  blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into delta-sized pieces for streaming. */
function piecesOf(text: string, size = 64): string[] {
  if (text === '') return [''];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

// ---------------------------------------------------------------------------
// Request building: the client's body as each upstream wire wants it.
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an
 *  inline image; null for the rest. */
function anthropicBlockToPart(block: JsonObject): JsonObject | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
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

/** An OpenAI content part as an Anthropic content block: text or an inline
 *  image; null for the rest. */
function openAIPartToBlock(part: JsonObject): JsonObject | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const url = part.image_url.url;
    if (typeof url !== 'string') return null;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** An OpenAI message's image as a Gemini part: inline data for a data URL,
 *  a file reference for a plain URL; null when it holds no image. */
function openAIImageToGeminiPart(part: JsonObject): JsonObject | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = part.image_url.url;
  if (typeof url !== 'string') return null;
  const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
  }
  return { fileData: { mimeType: 'image/png', fileUri: url } };
}

/** An Anthropic image block as a Gemini part. */
function anthropicImageToGeminiPart(block: JsonObject): JsonObject | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  }
  return null;
}

/** The client's tools as OpenAI function tools. */
function toolsAsOpenAI(body: JsonObject, clientApi: 'openai' | 'anthropic'): JsonObject[] {
  if (!Array.isArray(body.tools)) return [];
  const out: JsonObject[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (clientApi === 'anthropic') {
      if (typeof raw.name !== 'string') continue;
      out.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    } else if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      out.push(raw);
    }
  }
  return out;
}

/** The client's tools as Anthropic tools. */
function toolsAsAnthropic(body: JsonObject, clientApi: 'openai' | 'anthropic'): JsonObject[] {
  if (!Array.isArray(body.tools)) return [];
  const out: JsonObject[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (clientApi === 'openai') {
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      out.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else if (typeof raw.name === 'string') {
      out.push(raw);
    }
  }
  return out;
}

/** The client's tools as Gemini function declarations. */
function toolsAsGemini(body: JsonObject, clientApi: 'openai' | 'anthropic'): JsonObject[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations: JsonObject[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (clientApi === 'anthropic') {
      if (typeof raw.name !== 'string') continue;
      declarations.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    } else if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      declarations.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  return declarations;
}

/** The system text: the client's system message(s) for OpenAI, the `system`
 *  field for Anthropic. A top-level `system` field on an OpenAI body is
 *  not part of that wire and the client's parser ignores it, so it is
 *  dropped here too: carrying it would put text in the echo the client
 *  never asked for. */
function systemTextOf(body: JsonObject, clientApi: 'openai' | 'anthropic'): string {
  if (clientApi === 'anthropic') return textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = textOf(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The stop sequences the client asked for, in either wire's spelling. */
function stopsOf(body: JsonObject, clientApi: 'openai' | 'anthropic'): string[] {
  if (clientApi === 'anthropic') return stringList(body.stop_sequences);
  return stringList(body.stop);
}

/** The output cap the client asked for. */
function maxTokensOf(body: JsonObject, clientApi: 'openai' | 'anthropic'): number | null {
  if (clientApi === 'anthropic') {
    const v = numberOrZero(body.max_tokens);
    return typeof body.max_tokens === 'number' ? v : null;
  }
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** Whether the client asked for JSON mode (an OpenAI body only; the
 *  Anthropic wire has no such flag). */
export function wantsJsonMode(body: JsonObject, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, stop sequences and streaming pass through.
 *
 * One guard for the echo: a user message that held only tool_result blocks
 * has no text, so the client's parser reads its text as ''. The OpenAI
 * translation would otherwise drop that message entirely (it becomes only
 * tool messages), leaving an earlier user message as the last one. An
 * empty user message after the tool messages keeps the last-user-text ''.
 */
export function anthropicToOpenAI(body: JsonObject, model: string): JsonObject {
  const messages: JsonObject[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: JsonObject[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: JsonObject[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    else messages.push({ role: 'user', content: '' });
  }

  const out: JsonObject = { model, messages };
  const maxTokens = maxTokensOf(body, 'anthropic');
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body, 'anthropic');
  if (stops.length > 0) out.stop = stops;
  const tools = toolsAsOpenAI(body, 'anthropic');
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`: system messages become the system prompt, image_url parts
 * become image blocks, tool messages become tool_result blocks, tools and
 * tool_choice are rewritten, and the output cap, temperature, stop
 * sequences and streaming pass through.
 *
 * One guard for the echo: a tool message becomes a user message holding
 * only tool_result blocks, whose text the client's parser reads as ''.
 * When the tool message is the last one, that would clobber the last user
 * text the client sent. A text block carrying the most recent user text
 * rides in the same message, so the last user text survives; a later user
 * message overwrites it either way.
 */
export function openAIToAnthropic(body: JsonObject, model: string): JsonObject {
  const messages: JsonObject[] = [];
  let lastUserText = '';
  let sawUser = false;
  const system = systemTextOf(body, 'openai');
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'tool') {
      const blocks: JsonObject[] = [
        {
          type: 'tool_result',
          tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
          content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
        },
      ];
      if (sawUser && lastUserText !== '') blocks.push({ type: 'text', text: lastUserText });
      messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      const blocks: JsonObject[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: JsonObject = {};
        const args = call.function.arguments;
        if (typeof args === 'string' && args.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(args);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        } else if (isRecord(args)) {
          input = args;
        }
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message.
    const content = raw.content;
    if (typeof content === 'string') {
      lastUserText = content;
      sawUser = true;
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: JsonObject[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) {
      lastUserText = textOf(content);
      sawUser = true;
      messages.push({ role: 'user', content: blocks });
    }
  }

  const out: JsonObject = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = maxTokensOf(body, 'openai');
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body, 'openai');
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = toolsAsAnthropic(body, 'openai');
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * Either client wire as a Gemini generateContent request. The system text
 * becomes the system instruction, user messages become user contents,
 * assistant messages become model contents, tool messages become user
 * contents carrying their result text, tools become function declarations,
 * and the output cap, stop sequences, temperature and JSON mode become the
 * generation config.
 *
 * One guard for the echo: Gemini's parser reads the user text from the
 * last non-model content. A trailing tool message would clobber the last
 * user text, so when the last content is not a user content carrying that
 * text, a final user content with exactly the last user text is appended;
 * it changes nothing else the parser reads.
 */
export function toGemini(body: JsonObject, clientApi: 'openai' | 'anthropic'): JsonObject {
  const system = systemTextOf(body, clientApi);
  const contents: JsonObject[] = [];
  let lastUserText = '';
  let sawUser = false;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'assistant') {
      const text = textOf(raw.content);
      const parts: JsonObject[] = [];
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ text: `call ${typeof call.function.name === 'string' ? call.function.name : ''}` });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({ role: 'user', parts: [{ text }] });
      continue;
    }
    // A user message.
    const content = raw.content;
    if (typeof content === 'string') {
      lastUserText = content;
      sawUser = true;
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: JsonObject[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (clientApi === 'anthropic') {
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else {
          const geminiPart = anthropicImageToGeminiPart(part);
          if (geminiPart !== null) parts.push(geminiPart);
        }
      } else {
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else {
          const geminiPart = openAIImageToGeminiPart(part);
          if (geminiPart !== null) parts.push(geminiPart);
        }
      }
    }
    if (parts.length > 0) {
      lastUserText = textOf(content);
      sawUser = true;
      contents.push({ role: 'user', parts });
    }
  }
  // Keep the last non-model content's text equal to the last user text.
  const last = contents[contents.length - 1];
  const lastText = last !== undefined && isRecord(last) ? geminiPartsText(last.parts) : null;
  if (sawUser && lastText !== lastUserText) {
    contents.push({ role: 'user', parts: [{ text: lastUserText }] });
  } else if (!sawUser && contents.length > 0 && lastText !== '') {
    contents.push({ role: 'user', parts: [{ text: '' }] });
  }

  const out: JsonObject = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = toolsAsGemini(body, clientApi);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: JsonObject = {};
  const maxTokens = maxTokensOf(body, clientApi);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = stopsOf(body, clientApi);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (wantsJsonMode(body, clientApi)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini content's parts. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('\n');
}

// ---------------------------------------------------------------------------
// Response parsing: a provider's answer (JSON or SSE) as normalized text
// and usage. A provider's answer is always one text with a family-0 stop,
// so the stop reason is never carried: every emitter below writes it.
// ---------------------------------------------------------------------------

/** A Chat Completions JSON answer. */
export function parseOpenAIJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
  };
}

/** The `data:` payloads of an SSE body, in order, `[DONE]` included. */
export function ssePayloads(raw: string): string[] {
  const out: string[] = [];
  const normalized = raw.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      out.push(trimmed.slice(5).trim());
    }
  }
  return out;
}

/** A Chat Completions SSE stream: content deltas joined, usage from the
 *  usage chunk (the provider only sends one when include_usage was set). */
export function parseOpenAIStream(raw: string): NormalizedAnswer | null {
  let text = '';
  let input = 0;
  let output = 0;
  let saw = false;
  for (const payload of ssePayloads(raw)) {
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    saw = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
  }
  return saw ? { text, inputTokens: input, outputTokens: output } : null;
}

/** A Messages JSON answer: its text blocks joined, usage as reported. */
export function parseAnthropicJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body)) return null;
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: texts.join(''),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
  };
}

/** A Messages SSE stream: message_start carries the input tokens,
 *  message_delta the output tokens, text deltas the answer. */
export function parseAnthropicStream(raw: string): NormalizedAnswer | null {
  let text = '';
  let input = 0;
  let output = 0;
  let saw = false;
  for (const payload of ssePayloads(raw)) {
    let event: unknown;
    try {
      event = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    saw = true;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      input = numberOrZero(event.message.usage.input_tokens);
    } else if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta') {
      if (typeof event.delta.text === 'string') text += event.delta.text;
    } else if (event.type === 'message_delta' && isRecord(event.usage)) {
      output = numberOrZero(event.usage.output_tokens);
    }
  }
  return saw ? { text, inputTokens: input, outputTokens: output } : null;
}

/** A Gemini generateContent JSON answer. */
export function parseGeminiJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const text = geminiPartsText(content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
  };
}

/** A Gemini streamGenerateContent SSE stream: text pieces in order, the
 *  usage metadata on the final frame. */
export function parseGeminiStream(raw: string): NormalizedAnswer | null {
  let text = '';
  let input = 0;
  let output = 0;
  let saw = false;
  for (const payload of ssePayloads(raw)) {
    let frame: unknown;
    try {
      frame = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(frame)) continue;
    saw = true;
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    text += geminiPartsText(content.parts);
    if (isRecord(frame.usageMetadata)) {
      input = numberOrZero(frame.usageMetadata.promptTokenCount);
      output = numberOrZero(frame.usageMetadata.candidatesTokenCount);
    }
  }
  return saw ? { text, inputTokens: input, outputTokens: output } : null;
}

// ---------------------------------------------------------------------------
// Response emitting: a normalized answer in the client's wire.
// ---------------------------------------------------------------------------

/** A Chat Completions JSON answer. The stop reason is always `stop`: the
 *  provider's answer carries no other. */
export function openAIJsonAnswer(answer: NormalizedAnswer, model: string): JsonObject {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A Chat Completions SSE stream: a role chunk, content deltas, the finish
 *  chunk, the usage chunk, then `[DONE]` last. */
export function openAIStreamAnswer(answer: NormalizedAnswer, model: string): string {
  const chunk = (delta: JsonObject, finish: string | null): JsonObject => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of piecesOf(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A Messages JSON answer. The stop reason is always `end_turn`. */
export function anthropicJsonAnswer(answer: NormalizedAnswer, model: string): JsonObject {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A Messages SSE stream: message_start first, one text block, the deltas,
 *  then message_delta with the stop reason and output tokens, then
 *  message_stop. The input tokens ride in message_start, as the wire
 *  requires. */
export function anthropicStreamAnswer(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of piecesOf(answer.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** An error in the OpenAI shape. */
export function openAIError(message: string, type: string): JsonObject {
  return { error: { message, type, code: null } };
}

/** An error in the Anthropic shape. */
export function anthropicError(type: string, message: string): JsonObject {
  return { type: 'error', error: { type, message } };
}
