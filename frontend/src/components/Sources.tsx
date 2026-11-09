import { ChevronDown, FileText } from 'lucide-react'
import type { Citation } from '../api'
import type { Copy } from '../copy'
import { BidiText } from './BidiText'

export function Sources({ citations, copy }: { citations: Citation[]; copy: Copy }) {
  return <details className="sources">
    <summary><FileText size={17} aria-hidden="true" /><span>{copy.sources}</span>
      <span className="source-count">{citations.length}</span><ChevronDown className="disclosure-chevron" size={16} aria-hidden="true" /></summary>
    <div className="sources-body">
      {citations.map((citation, index) => <section className="source-document" key={`${citation.doc_id}-${index}`} aria-label={`${copy.source} ${index + 1}`}>
        <div className="source-title"><BidiText text={citation.title} /><bdi className="source-id" dir="ltr">{citation.doc_id}</bdi></div>
        <span className="source-field-label">{copy.section}</span><BidiText className="source-section" text={citation.section} />
        <span className="source-field-label">{copy.excerpt}</span><blockquote><BidiText text={citation.snippet} /></blockquote>
      </section>)}
      <p className="source-note">{copy.sourceNote}</p>
    </div>
  </details>
}
