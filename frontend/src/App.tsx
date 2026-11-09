import { BookOpen, Globe2, Plus, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { translations } from './copy'
import type { Locale } from './copy'
import { useChat } from './useChat'
import { Brand } from './components/Brand'
import { Composer } from './components/Composer'
import { Conversation } from './components/Conversation'
import { EmptyState } from './components/EmptyState'

export function App() {
  const [locale, setLocale] = useState<Locale>('en')
  const [draft, setDraft] = useState('')
  const [announcement, announce] = useState('')
  const { turns, pending, send, clear } = useChat()
  const input = useRef<HTMLTextAreaElement>(null)
  const dialog = useRef<HTMLDialogElement>(null)
  const newButton = useRef<HTMLButtonElement>(null)
  const wasPending = useRef(false)
  const copy = translations[locale]

  useEffect(() => {
    document.documentElement.lang = locale
    document.documentElement.dir = locale === 'ar' ? 'rtl' : 'ltr'
  }, [locale])
  useEffect(() => {
    if (pending) announce(copy.questionSent)
    else if (wasPending.current) announce(copy.received)
    wasPending.current = pending
  }, [pending, copy])

  function submit() {
    if (pending || !draft.trim()) return
    const question = draft.trim()
    setDraft('')
    void send(question)
  }
  function startNew() {
    clear(); setDraft(''); announce(copy.cleared)
    dialog.current?.close()
    input.current?.focus()
  }

  return <div className="app-shell">
    <a className="skip-link" href="#workspace">{copy.skip}</a>
    <header className="app-header"><div className="header-content">
      <div className="brand"><Brand /><div><span className="brand-name" dir="ltr">{copy.brand}</span><p>{copy.tagline}</p></div></div>
      <nav className="header-actions" aria-label={locale === 'ar' ? 'أدوات المحادثة' : 'Conversation controls'}>
        <button className="language-button" type="button" onClick={() => setLocale(locale === 'en' ? 'ar' : 'en')} aria-label={copy.switchLanguage}>
          <Globe2 size={17} aria-hidden="true" /><span lang={locale === 'en' ? 'ar' : 'en'}>{locale === 'en' ? 'العربية' : 'English'}</span>
        </button>
        <span className="header-divider" aria-hidden="true" />
        <button className="new-conversation" ref={newButton} type="button" disabled={turns.length === 0} onClick={() => dialog.current?.showModal()}>
          <Plus size={18} aria-hidden="true" /><span>{copy.newChat}</span>
        </button>
      </nav>
    </div></header>

    <main className={`workspace ${turns.length ? 'workspace--active' : ''}`} id="workspace" tabIndex={-1}>
      {turns.length === 0 ? <EmptyState copy={copy} choose={question => { setDraft(question); input.current?.focus() }} />
        : <Conversation turns={turns} copy={copy} pending={pending} announce={announce} retry={turn => void send(turn.question, turn.id)} />}
      <Composer copy={copy} draft={draft} setDraft={setDraft} pending={pending} submit={submit} inputRef={input} />
      {turns.length === 0 && <aside className="source-promise"><span className="source-promise-icon"><BookOpen size={21} strokeWidth={1.7} aria-hidden="true" /></span><div><h2>{copy.sourceHeading}</h2><p>{copy.sourceHelp}</p></div></aside>}
      {turns.length > 0 && <p className="session-note">{copy.sessionNote}</p>}
    </main>
    <footer className="app-footer"><span dir="ltr">MTI Medical</span><span>{locale === 'ar' ? 'معرفة واضحة. مصادر شفافة.' : 'Thoughtful answers. Transparent sources.'}</span></footer>
    <div className="visually-hidden" role="status" aria-live="polite" aria-atomic="true">{announcement}</div>

    <dialog className="clear-dialog" ref={dialog} aria-labelledby="clear-title" aria-describedby="clear-description" onClose={() => { if (document.activeElement === document.body) newButton.current?.focus() }}>
      <button className="dialog-close" type="button" onClick={() => dialog.current?.close()} aria-label={copy.cancel}><X size={20} aria-hidden="true" /></button>
      <h2 id="clear-title">{copy.clearTitle}</h2><p id="clear-description">{copy.clearText}</p>
      <div className="dialog-actions"><button type="button" className="secondary-button" onClick={() => dialog.current?.close()} autoFocus>{copy.cancel}</button><button type="button" className="primary-button" onClick={startNew}>{copy.clear}</button></div>
    </dialog>
  </div>
}
