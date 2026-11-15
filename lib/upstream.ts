// One upstream attempt: a single call to one provider on its own wire.
//
// A call that answers 2xx is a success; 402, 429 or a 5xx is a refusal
// the caller fails over from; anything else (a 400 for a bad model, a 401
// for a wrong key) is a bug, never a schedule, and is reported the same
// way so the caller can move on. A call that cannot be reached at all
// counts as status 0.
//
// When the client and the provider speak the same wire and the request
// streams, the provider's event stream is passed through untouched: it is
// already the exact event order the client expects. Every other answer is
// read fully and normalized to text plus usage, then re-emitted in the
// client's wire by the caller.

import { authHeaders, joinBase } from './brief.ts';
import type { DirectoryProvider } from './brief.ts';
import {
  anthropicToOpenAI,
  openAIToAnthropic,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  toGemini,
} from './translate.ts';
import type { NormalizedAnswer } from './translate.ts';
import type { JsonObject } from './brief.ts';
import type { Candidate } from './plan.ts';

export interface Attempt {
  ok: boolean;
  status: number;
  /** A same-wire event stream to pipe straight to the client, when set. */
  passthrough: ReadableStream<Uint8Array> | null;
  /** The normalized answer, when the call succeeded and was read fully. */
  answer: NormalizedAnswer | null;
  /** The provider's error body, for diagnostics only. */
  errorText: string;
}

const CALL_TIMEOUT_MS = 10000;

function failure(status: number, errorText: string): Attempt {
  return { ok: false, status, passthrough: null, answer: null, errorText };
}

/** The upstream request for one provider: its URL, headers and body. */
function upstreamRequest(
  envUrl: string,
  provider: DirectoryProvider,
  model: string,
  clientApi: 'openai' | 'anthropic',
  body: JsonObject,
  stream: boolean,
): { url: string; headers: Record<string, string>; payload: JsonObject } {
  const base = joinBase(envUrl, provider.path);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    const payload =
      clientApi === 'openai'
        ? { ...body, model }
        : anthropicToOpenAI(body, model);
    if (stream) {
      headers.accept = 'text/event-stream';
      payload.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload =
      clientApi === 'anthropic'
        ? { ...body, model }
        : openAIToAnthropic(body, model);
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers, payload };
  }
  // Gemini: the model rides in the path, the stream flag in the verb.
  const encoded = encodeURIComponent(model);
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encoded}:${verb}${stream ? '?alt=sse' : ''}`;
  if (stream) headers.accept = 'text/event-stream';
  return { url, headers, payload: toGemini(body, clientApi) };
}

/** Parse a 2xx answer of one wire into text plus usage. */
function parseAnswer(wire: string, stream: boolean, text: string): NormalizedAnswer | null {
  if (wire === 'openai') {
    if (stream) return parseOpenAIStream(text);
    try {
      return parseOpenAIJson(JSON.parse(text) as unknown);
    } catch {
      return null;
    }
  }
  if (wire === 'anthropic') {
    if (stream) return parseAnthropicStream(text);
    try {
      return parseAnthropicJson(JSON.parse(text) as unknown);
    } catch {
      return null;
    }
  }
  if (stream) return parseGeminiStream(text);
  try {
    return parseGeminiJson(JSON.parse(text) as unknown);
  } catch {
    return null;
  }
}

/**
 * One call to one provider's model. The caller records the call against
 * the provider's budget whether it succeeds or refuses.
 */
export async function callOnce(
  envUrl: string,
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: JsonObject,
  stream: boolean,
): Promise<Attempt> {
  const provider = candidate.probe.provider;
  const sameWire =
    (clientApi === 'openai' && provider.wire === 'openai') ||
    (clientApi === 'anthropic' && provider.wire === 'anthropic');
  const request = upstreamRequest(envUrl, provider, candidate.modelId, clientApi, body, stream);
  let res: Response;
  try {
    res = await fetch(request.url, {
      method: 'POST',
      headers: request.headers,
      body: JSON.stringify(request.payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    return failure(0, `the provider could not be reached: ${(err as Error).message}`);
  }
  if (!res.ok) {
    let errorText = '';
    try {
      errorText = (await res.text()).slice(0, 500);
    } catch {
      // The body is gone; the status is what matters.
    }
    return failure(res.status, errorText);
  }
  // A same-wire stream goes straight to the client untouched.
  if (stream && sameWire && res.body !== null) {
    return { ok: true, status: res.status, passthrough: res.body, answer: null, errorText: '' };
  }
  let raw = '';
  try {
    raw = await res.text();
  } catch (err) {
    return failure(0, `the provider's answer could not be read: ${(err as Error).message}`);
  }
  const answer = parseAnswer(provider.wire, stream, raw);
  if (answer === null) return failure(502, 'the provider answered in a shape this server could not read');
  return { ok: true, status: res.status, passthrough: null, answer, errorText: '' };
}
