export type ChatStatus = 'answered' | 'out_of_scope' | 'escalated'
export type AnswerLanguage = 'ar' | 'en' | 'mixed'
export interface Citation { doc_id: string; title: string; section: string; snippet: string }
export interface ChatResponse {
  answer: string; citations: Citation[]; status: ChatStatus; language: AnswerLanguage
}
export type ErrorKind = 'validation' | 'service' | 'network' | 'timeout' | 'invalid_response'

export class ChatError extends Error {
  constructor(public kind: ErrorKind, public status?: number) {
    super(kind)
    this.name = 'ChatError'
  }
}

const text = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object'

export function isChatResponse(value: unknown): value is ChatResponse {
  if (!record(value) || !text(value.answer) || !Array.isArray(value.citations)
      || !['answered', 'out_of_scope', 'escalated'].includes(String(value.status))
      || !['ar', 'en', 'mixed'].includes(String(value.language))) return false
  if ((value.status === 'answered') !== (value.citations.length > 0)) return false
  return value.citations.every(citation => record(citation)
    && ['doc_id', 'title', 'section', 'snippet'].every(key => text(citation[key])))
}

const baseUrl = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')

export async function requestChat(message: string, signal: AbortSignal): Promise<ChatResponse> {
  const controller = new AbortController()
  let timedOut = false
  const cancel = () => controller.abort()
  signal.addEventListener('abort', cancel, { once: true })
  if (signal.aborted) controller.abort()
  const timeout = window.setTimeout(() => { timedOut = true; controller.abort() }, 40_000)
  try {
    const response = await fetch(`${baseUrl}/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ message, session_id: null }), signal: controller.signal,
    })
    if (!response.ok) throw new ChatError(response.status === 422 ? 'validation' : 'service', response.status)
    const body: unknown = await response.json().catch(() => { throw new ChatError('invalid_response') })
    if (!isChatResponse(body)) throw new ChatError('invalid_response')
    return body
  } catch (error) {
    if (signal.aborted) throw new DOMException('Request cancelled', 'AbortError')
    if (timedOut) throw new ChatError('timeout')
    if (error instanceof ChatError) throw error
    throw new ChatError('network')
  } finally {
    clearTimeout(timeout)
    signal.removeEventListener('abort', cancel)
  }
}
