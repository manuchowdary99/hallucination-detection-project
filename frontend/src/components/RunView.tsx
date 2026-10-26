import { Bot, Wand2 } from 'lucide-react'
import type { RunResponse, VerificationResult } from '../types/api'
import ClaimCard from './ClaimCard'
import ScoreCard from './ScoreCard'
import { Alert, Spinner } from './ui'
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
