export function Brand({ compact = false }: { compact?: boolean }) {
  return <span className={`brand-mark ${compact ? 'brand-mark--small' : ''}`} aria-hidden="true">
    <svg viewBox="0 0 40 40" fill="none"><path d="M9 11h9a4 4 0 0 1 4 4v15a7 7 0 0 0-6-3H9V11Zm22 0h-5v16h5V11Z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
      <path d="m14 20 3-4 4 8 3-5h8" stroke="#70D5CE" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" /></svg>
  </span>
}
