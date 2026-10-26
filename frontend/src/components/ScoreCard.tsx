import type { ScoreBreakdown } from '../types/api'
import { scoreColor } from './format'

export default function ScoreCard({ score, title = 'Reliability score' }: { score: ScoreBreakdown; title?: string }) {
  const c = score.counts
  return (
    <div className="card p-4" data-testid="score-card">
      <h3 className="text-sm font-medium text-slate-500">{title}</h3>
      <div className="mt-1 flex items-baseline gap-2">
        <span className={`text-4xl font-bold ${scoreColor(score.score)}`}>{score.score == null ? '—' : score.score.toFixed(0)}</span>
        {score.score != null && <span className="text-slate-400">/ 100</span>}
        <span className={`ml-2 text-sm font-semibold ${scoreColor(score.score)}`}>{score.label}</span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
        <Count n={c.supported} label="Supported" cls="text-emerald-700" />
        <Count n={c.contradicted} label="Contradicted" cls="text-rose-700" />
        <Count n={c.unverifiable} label="Unverifiable" cls="text-amber-700" />
        <Count n={c.unavailable} label="Unavailable" cls="text-slate-600" />
      </div>
      <details className="mt-3 text-xs text-slate-500">
        <summary className="cursor-pointer">How is this calculated?</summary>
        <p className="mt-1 font-mono">{score.formula}</p>
        {Object.keys(score.components).length > 0 && (
          <p className="mt-1">
            {Object.entries(score.components).map(([k, v]) => `${k} = ${v}`).join(' · ')}
          </p>
        )}
        <p className="mt-1">This is an evidence-coverage estimate, not a calibrated probability that the answer is true.</p>
      </details>
    </div>
  )
}

function Count({ n, label, cls }: { n: number; label: string; cls: string }) {
  return (
    <div className="rounded-lg bg-slate-50 px-2 py-1.5">
      <div className={`text-lg font-semibold ${cls}`}>{n}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}
