import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatError, requestChat } from './api'
import type { ChatResponse } from './api'

export interface Turn { id: string; question: string; pending: boolean; response?: ChatResponse; error?: ChatError }

export function useChat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [pending, setPending] = useState(false)
  const active = useRef<AbortController | null>(null)
  const version = useRef(0)
  const nextId = useRef(0)

  useEffect(() => () => active.current?.abort(), [])

  const send = useCallback(async (question: string, retryId?: string) => {
    if (active.current || !question.trim()) return
    const controller = new AbortController()
    active.current = controller
    const requestVersion = ++version.current
    const id = retryId || `turn-${++nextId.current}`
    const turn: Turn = { id, question: question.trim(), pending: true }
    setTurns(previous => retryId ? previous.map(item => item.id === id ? turn : item) : [...previous, turn])
    setPending(true)
    try {
      const response = await requestChat(turn.question, controller.signal)
      if (version.current === requestVersion) {
        setTurns(previous => previous.map(item => item.id === id ? { ...turn, pending: false, response } : item))
      }
    } catch (error) {
      if (version.current === requestVersion && !controller.signal.aborted) {
        const failure = error instanceof ChatError ? error : new ChatError('network')
        setTurns(previous => previous.map(item => item.id === id ? { ...turn, pending: false, error: failure } : item))
      }
    } finally {
      if (version.current === requestVersion) { active.current = null; setPending(false) }
    }
  }, [])

  const clear = useCallback(() => {
    ++version.current
    active.current?.abort()
    active.current = null
    setTurns([])
    setPending(false)
  }, [])

  return { turns, pending, send, clear }
}
