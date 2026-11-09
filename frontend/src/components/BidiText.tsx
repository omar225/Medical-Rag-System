import { Fragment } from 'react'

function direction(text: string): 'rtl' | 'ltr' {
  const firstLetter = text.match(/\p{Letter}/u)?.[0]
  return firstLetter && /\p{Script=Arabic}/u.test(firstLetter) ? 'rtl' : 'ltr'
}

const scriptRuns = /([A-Za-z][A-Za-z0-9]*(?:[ \t/()+.%:'’\-]+[A-Za-z0-9]+)*|[\p{Script=Arabic}\p{Mark}]+(?:[ \t]+[\p{Script=Arabic}\p{Mark}]+)*)/gu

// Determine paragraph direction from the original text, before isolating Latin runs.
// dir=auto ignores nested <bdi>, which otherwise misclassifies English-first mixed lines.
// Text nodes/newlines are preserved exactly; no reversal or invisible character insertion.
export function BidiText({ text, className = '', language, inline = false, role }: {
  text: string; className?: string; language?: string; inline?: boolean; role?: 'alert'
}) {
  const Element = inline ? 'span' : 'div'
  return <Element className={`bidi-text ${className}`} dir={direction(text)} lang={language === 'mixed' ? undefined : language} role={role}>
    {text.split('\n').map((line, lineIndex) => <Fragment key={lineIndex}>
      {lineIndex > 0 ? '\n' : null}<span className="bidi-line" dir={direction(line)}>
        {line.split(scriptRuns).map((part, index) => index % 2 === 1
          ? <bdi dir={direction(part)} lang={direction(part) === 'rtl' ? 'ar' : 'en'} key={index}>{part}</bdi> : part)}
      </span>
    </Fragment>)}
  </Element>
}
