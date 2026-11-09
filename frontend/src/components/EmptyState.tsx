import { ArrowUpRight, BookOpen, Pill, ShieldCheck, Sparkles } from 'lucide-react'
import type { Copy } from '../copy'
import { examples } from '../copy'
import { BidiText } from './BidiText'

export function EmptyState({ copy, choose }: { copy: Copy; choose: (text: string) => void }) {
  return <section className="welcome" aria-labelledby="welcome-title">
    <div className="welcome-intro">
      <div>
        <h1 id="welcome-title">{copy.title}</h1>
        <p className="welcome-description">{copy.intro}</p>
      </div>
      <div className="evidence-illustration" aria-hidden="true">
        <div className="illustration-ring" />
        <div className="paper paper--back" />
        <div className="paper paper--front">
          <BookOpen size={30} strokeWidth={1.5} />
          <span className="paper-line paper-line--short" /><span className="paper-line" /><span className="paper-line" />
          <div className="paper-reference"><span /><span /><span /></div>
        </div>
        <div className="illustration-seal"><ShieldCheck size={23} strokeWidth={1.7} /></div>
      </div>
    </div>
    <div className="examples-heading"><h2>{copy.examples}</h2><span>{copy.examplesHint}</span></div>
    <div className="examples">
      {examples.map(({ kind, question }) => {
        const Icon = kind === 'ingredient' ? Pill : kind === 'cautions' ? ShieldCheck : Sparkles
        return <button className="example" key={kind} onClick={() => choose(question)} type="button">
          <span className="example-symbol"><Icon size={20} strokeWidth={1.6} aria-hidden="true" /></span>
          <span className="example-content"><span className="example-kind">{copy[kind]}</span><BidiText text={question} inline /></span>
          <ArrowUpRight className="example-arrow" size={18} aria-hidden="true" />
        </button>
      })}
    </div>
  </section>
}
