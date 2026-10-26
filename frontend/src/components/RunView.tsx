import { ArrowRight, Bot, ExternalLink, Wand2 } from 'lucide-react'
import type { ClaimChange, RunResponse, VerificationResult } from '../types/api'
import ClaimCard from './ClaimCard'
import ScoreCard from './ScoreCard'
import { Alert, Spinner, StatusBadge } from './ui'
import { fmtMs } from './format'

function Claims({ result, showChanged }: { result: VerificationResult; showChanged?: boolean }) {
  if (result.claims.length === 0)
    return <p className="text-sm text-slate-500">No verifiable factual claims were detected in this text.</p>
  return (
    <div className="space-y-3">
      {result.claims.map((c) => (
        <ClaimCard key={c.index + c.text} claim={c} showChanged={showChanged} />
      ))}
    </div>
  )
}

function Changes({ changes }: { changes: ClaimChange[] }) {
  return (
    <div className="card p-4" data-testid="correction-changes">
      <h3 className="text-sm font-medium text-slate-500">What changed (each rewritten claim was re-verified)</h3>
      <ul className="mt-2 space-y-3">
        {changes.map((ch, i) => (
          <li key={i} className="text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className={ch.original_text ? 'text-slate-500 line-through' : 'italic text-slate-400'}>
                {ch.original_text ?? 'new claim'}
              </span>
              {ch.original_status && <StatusBadge status={ch.original_status} />}
              <ArrowRight className="h-4 w-4 text-slate-400" aria-hidden />
              <span className={ch.corrected_text ? 'font-medium text-slate-900' : 'italic text-slate-400'}>
                {ch.corrected_text ?? 'removed'}
              </span>
              {ch.corrected_status && <StatusBadge status={ch.corrected_status} />}
            </div>
            {ch.sources.length > 0 ? (
              <div className="mt-1 flex flex-wrap gap-3 text-xs">
                {ch.sources.map((s) => (
                  <a key={s.url + s.title} href={s.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-indigo-700 hover:underline">
                    {s.source_name}: {s.title} <ExternalLink className="h-3 w-3" aria-hidden />
                  </a>
                ))}
              </div>
            ) : ch.corrected_text ? (
              <p className="mt-1 text-xs text-amber-700">No retrieved evidence supports this rewritten claim.</p>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  )
}

interface Props {
  run: RunResponse
  onCorrect?: () => void
  correcting?: boolean
}

export default function RunView({ run, onCorrect, correcting }: Props) {
  const corrected = run.corrected_verification
  const canCorrect = run.original_verification.claims.some((c) => c.status === 'CONTRADICTED' || c.status === 'UNVERIFIABLE')
  return (
    <div className="space-y-6" data-testid="run-view">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
        <span>Run #{run.id}</span>
        <span>{new Date(run.created_at).toLocaleString()}</span>
        <span>Mode: {run.mode === 'generate' ? 'Generated + verified' : 'Verified existing answer'}</span>
        {run.provider && (
          <span className="inline-flex items-center gap-1">
            <Bot className="h-3.5 w-3.5" aria-hidden /> {run.provider.name}
            {run.provider.is_demo && <strong className="text-amber-700"> (demo provider)</strong>}
          </span>
        )}
        <span>Processing time: {fmtMs(run.processing_time_ms)}</span>
      </div>
      {run.warnings.length > 0 && (
        <div className="space-y-2">
          {run.warnings.map((w) => (
            <Alert key={w}>{w}</Alert>
          ))}
        </div>
      )}
      {run.question && (
        <div className="card p-4">
          <h3 className="text-sm font-medium text-slate-500">Question</h3>
          <p className="mt-1 text-slate-800">{run.question}</p>
        </div>
      )}

      <div className={`grid gap-4 ${corrected ? 'lg:grid-cols-2' : ''}`}>
        <div className="card p-4">
          <h3 className="text-sm font-medium text-slate-500">Original answer</h3>
          <p className="mt-1 whitespace-pre-wrap text-slate-800">{run.original_answer}</p>
        </div>
        {corrected && (
          <div className="card border-indigo-200 p-4">
            <h3 className="text-sm font-medium text-indigo-700">Corrected answer (re-verified)</h3>
            <p className="mt-1 whitespace-pre-wrap text-slate-800">{run.final_answer}</p>
          </div>
        )}
      </div>

      {run.correction.status !== 'not_requested' && (
        <Alert kind={run.correction.status === 'failed' ? 'error' : 'info'}>
          <strong>Correction: {run.correction.status.replace('_', ' ')}.</strong> {run.correction.message}
          {run.correction.status === 'corrected' && (
            <> Score {run.correction.score_before ?? '—'} → {run.correction.score_after ?? '—'}; {run.correction.changed_claims} claim(s) changed.</>
          )}
        </Alert>
      )}

      {!!run.correction.unsupported_changed_claims && (
        <Alert>
          {run.correction.unsupported_changed_claims} rewritten or added claim(s) are not supported by retrieved evidence. Treat them as unverified.
        </Alert>
      )}

      {corrected && run.correction.changes && run.correction.changes.length > 0 && <Changes changes={run.correction.changes} />}

      {onCorrect && canCorrect && (
        <div className="flex items-center gap-3">
          <button className="btn-secondary" onClick={onCorrect} disabled={correcting}>
            <Wand2 className="h-4 w-4" aria-hidden /> {corrected ? 'Re-run correction' : 'Correct unreliable claims'}
          </button>
          {correcting && <Spinner label="Correcting and re-verifying…" />}
        </div>
      )}

      <div className={`grid gap-6 ${corrected ? 'lg:grid-cols-2' : ''}`}>
        <section className="space-y-3">
          <ScoreCard score={run.original_verification.score} title={corrected ? 'Original reliability' : 'Reliability score'} />
          <h3 className="font-semibold text-slate-800">Claims in original answer</h3>
          <Claims result={run.original_verification} showChanged={!!corrected} />
        </section>
        {corrected && (
          <section className="space-y-3">
            <ScoreCard score={corrected.score} title="Corrected reliability" />
            <h3 className="font-semibold text-slate-800">Claims in corrected answer</h3>
            <Claims result={corrected} showChanged />
          </section>
        )}
      </div>
    </div>
  )
}
