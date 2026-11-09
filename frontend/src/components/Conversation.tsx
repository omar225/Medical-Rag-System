import { AlertCircle, AlertTriangle, Check, Copy as CopyIcon, MessageCircle, LoaderCircle, RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Copy } from '../copy'
import type { Turn } from '../useChat'
import { BidiText } from './BidiText'
import { Brand } from './Brand'
import { Sources } from './Sources'

export function Conversation({ turns, copy, retry, pending, announce }: {
  turns: Turn[]; copy: Copy; retry: (turn: Turn) => void; pending: boolean; announce: (message: string) => void
}) {
  const latest = useRef<HTMLLIElement>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const copyTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(copyTimer.current), [])
  useEffect(() => {
    if (turns.length > 1) latest.current?.scrollIntoView({ block: 'start', behavior: 'instant' })
  }, [turns.length])

  async function copyAnswer(turn: Turn) {
    try {
      await navigator.clipboard.writeText(turn.response!.answer)
      setCopied(turn.id); announce(copy.copied)
      clearTimeout(copyTimer.current)
      copyTimer.current = setTimeout(() => setCopied(null), 2000)
    } catch { announce(copy.copyFailed) }
  }

  return <section className="conversation" aria-labelledby="conversation-title">
    <h1 className="visually-hidden" id="conversation-title">{copy.conversation}</h1>
    <ol className="turns">
      {turns.map((turn, index) => <li className="turn" key={turn.id} ref={index === turns.length - 1 ? latest : undefined}>
        <div className="user-message"><span className="visually-hidden">{copy.you}: </span><BidiText text={turn.question} /></div>
        <article className={`assistant-message ${turn.response?.status === 'escalated' ? 'assistant-message--escalated' : ''}`} aria-label={copy.assistant}>
          <div className="assistant-identity"><Brand compact /><h2>{copy.assistant}</h2>
            {turn.response && <button type="button" className="copy-answer" onClick={() => void copyAnswer(turn)} aria-label={copied === turn.id ? copy.copied : copy.copy} title={copy.copy}>
              {copied === turn.id ? <Check size={16} aria-hidden="true" /> : <CopyIcon size={16} aria-hidden="true" />}
            </button>}
          </div>
          {turn.pending && <div className="pending-response" aria-busy="true"><LoaderCircle className="spinner" size={18} aria-hidden="true" /><span>{copy.waiting}</span></div>}
          {turn.error && <div className="request-error" role="alert" aria-live="polite">
            <AlertCircle size={20} aria-hidden="true" /><div><h3>{copy.errorTitle}</h3><p>{copy[turn.error.kind]}</p>
              <button type="button" className="retry-button" disabled={pending} onClick={() => retry(turn)}><RotateCcw size={14} aria-hidden="true" />{copy.retry}</button></div>
          </div>}
          {turn.response && <>
            <div className={`answer-status answer-status--${turn.response.status}`}>
              {turn.response.status === 'escalated' ? <AlertTriangle size={17} aria-hidden="true" /> : turn.response.status === 'out_of_scope' ? <MessageCircle size={17} aria-hidden="true" /> : <span className="status-dot" aria-hidden="true" />}
              <span>{copy[turn.response.status]}</span>
            </div>
            <BidiText className="answer-text" text={turn.response.answer} language={turn.response.language} role={turn.response.status === 'escalated' ? 'alert' : undefined} />
            {turn.response.citations.length > 0 && <Sources citations={turn.response.citations} copy={copy} />}
          </>}
        </article>
      </li>)}
    </ol>
  </section>
}
