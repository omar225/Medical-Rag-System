import { ArrowUp, CornerDownLeft, LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { RefObject } from 'react'
import type { Copy } from '../copy'

export function Composer({ copy, draft, setDraft, pending, submit, inputRef }: {
  copy: Copy; draft: string; setDraft: (value: string) => void; pending: boolean;
  submit: () => void; inputRef: RefObject<HTMLTextAreaElement | null>
}) {
  const [invalid, setInvalid] = useState(false)
  useEffect(() => {
    const element = inputRef.current
    if (element) { element.style.height = 'auto'; element.style.height = `${Math.min(element.scrollHeight, 176)}px` }
    if (draft.trim()) setInvalid(false)
  }, [draft, inputRef])

  function send() {
    if (pending) return
    if (!draft.trim()) {
      setInvalid(true)
      inputRef.current?.focus()
      return
    }
    submit()
  }

  return <div className="composer-area">
    <form className="composer" onSubmit={event => { event.preventDefault(); send() }}>
      <label htmlFor="medical-question">{copy.composerLabel}</label>
      <div className="composer-input-row">
        <textarea id="medical-question" name="message" ref={inputRef} value={draft}
          onChange={event => setDraft(event.target.value)} dir="auto" rows={2}
          autoComplete="off" aria-invalid={invalid || undefined} aria-describedby="question-hint composer-error" placeholder={copy.placeholder}
          onKeyDown={event => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
              event.preventDefault(); send()
            }
          }} />
        <button className="send-button" type="submit" disabled={pending} aria-label={pending ? copy.sending : copy.send}>
          {pending ? <LoaderCircle className="spinner" size={22} aria-hidden="true" /> : <ArrowUp size={23} aria-hidden="true" />}
        </button>
      </div>
      <div className="composer-footer"><span id="question-hint">{copy.inputHint}</span>
        <span className="keyboard-hint" title={copy.newlineHint}><CornerDownLeft size={12} aria-hidden="true" />{copy.keyboardHint}</span>
      </div>
      <p className="input-error" id="composer-error" aria-live="polite">{invalid ? copy.emptyError : ''}</p>
    </form>
    <p className="medical-disclaimer">{copy.safety}</p>
  </div>
}
