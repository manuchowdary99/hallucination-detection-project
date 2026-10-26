import { ChevronDown, ExternalLink, GitCompare } from 'lucide-react'
import { useState } from 'react'
import type { ClaimResult } from '../types/api'
import { StatusBadge } from './ui'
import { fmtPct } from './format'

function NliBars({ e, c, n }: { e: number; c: number; n: number }) {
  return (
    <div className="flex flex-wrap gap-3 text-xs text-slate-600">
      <span>Entailment <strong className="text-emerald-700">{fmtPct(e)}</strong></span>
      <span>Contradiction <strong className="text-rose-700">{fmtPct(c)}</strong></span>
      <span>Neutral <strong className="text-slate-700">{fmtPct(n)}</strong></span>
    </div>
  )
}

export default function ClaimCard({ claim, showChanged }: { claim: ClaimResult; showChanged?: boolean }) {
  const [open, setOpen] = useState(claim.status !== 'SUPPORTED')
  return (
    <article className="card p-4" data-testid="claim-card">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <p className="flex-1 font-medium text-slate-900">{claim.text}</p>
        <div className="flex items-center gap-2">
          {showChanged && claim.changed_in_correction && (
            <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-700 ring-1 ring-inset ring-indigo-200">
              <GitCompare className="h-3 w-3" aria-hidden /> Changed in correction
            </span>
          )}
          <StatusBadge status={claim.status} />
        </div>
      </div>
      <p className="mt-2 text-sm text-slate-600">{claim.explanation}</p>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
        <span>
          Claim confidence: <strong className="text-slate-700">{claim.confidence == null ? 'n/a' : `${claim.confidence.toFixed(1)}%`}</strong>
        </span>
        {claim.nli && <NliBars e={claim.nli.entailment} c={claim.nli.contradiction} n={claim.nli.neutral} />}
        {claim.conflicting_evidence && <span className="font-medium text-amber-700">Sources disagree</span>}
      </div>
      {claim.entities.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {claim.entities.map((e) => (
            <span key={e.text + e.label} className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">
              {e.text} <span className="text-slate-400">{e.label}</span>
            </span>
          ))}
        </div>
      )}
      <button className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-indigo-700" onClick={() => setOpen(!open)} aria-expanded={open}>
        <ChevronDown className={`h-4 w-4 transition ${open ? 'rotate-180' : ''}`} aria-hidden />
        {claim.evidence.length} evidence passage{claim.evidence.length === 1 ? '' : 's'}
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {claim.evidence.length === 0 && <li className="text-sm text-slate-500">No relevant evidence was retrieved.</li>}
          {claim.evidence.map((ev, i) => (
            <li key={i} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <a href={ev.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-sm font-medium text-indigo-700 hover:underline">
                  {ev.source_name}: {ev.title} <ExternalLink className="h-3.5 w-3.5" aria-hidden />
                </a>
                {ev.label && <StatusBadge status={ev.label} />}
              </div>
              <blockquote className="mt-1 text-sm text-slate-700">“{ev.snippet}”</blockquote>
              <div className="mt-1 flex flex-wrap gap-3 text-xs text-slate-500">
                <span>Relevance {fmtPct(ev.relevance)}</span>
                {ev.nli && <NliBars e={ev.nli.entailment} c={ev.nli.contradiction} n={ev.nli.neutral} />}
                {ev.retrieved_at && <span>Retrieved {new Date(ev.retrieved_at).toLocaleString()}</span>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </article>
  )
}
