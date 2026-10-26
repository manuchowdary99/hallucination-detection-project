import { Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Alert, Spinner } from '../components/ui'
import { fmtDate, fmtMs, scoreColor } from '../components/format'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'

export default function History() {
  const { data, error, loading, setData } = useApi(api.history)
  const [actionError, setActionError] = useState<string | null>(null)

  async function remove(id: number) {
    if (!window.confirm(`Delete verification run #${id}?`)) return
    try {
      await api.deleteRun(id)
      setData((d) => d?.filter((r) => r.id !== id) ?? null)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    }
  }

  async function clearAll() {
    if (!window.confirm('Delete ALL verification history? This cannot be undone.')) return
    try {
      await api.clearHistory()
      setData([])
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">History</h1>
          <p className="text-sm text-slate-600">Every verification run is stored locally in SQLite.</p>
        </div>
        {data && data.length > 0 && (
          <button className="btn-danger" onClick={clearAll}>
            <Trash2 className="h-4 w-4" aria-hidden /> Clear history
          </button>
        )}
      </div>
      {loading && <Spinner label="Loading history…" />}
      {(error || actionError) && <Alert kind="error">{error || actionError}</Alert>}
      {data && data.length === 0 && (
        <div className="card p-8 text-center text-slate-600">
          No saved runs. <Link to="/verify" className="font-medium text-indigo-700">Start verifying</Link>.
        </div>
      )}
      {data && data.length > 0 && (
        <ul className="space-y-3">
          {data.map((r) => (
            <li key={r.id} className="card flex flex-wrap items-center gap-4 p-4">
              <Link to={`/history/${r.id}`} className="min-w-0 flex-1">
                <p className="truncate font-medium text-slate-900 hover:text-indigo-700">{r.query_summary}</p>
                <p className="mt-1 text-xs text-slate-500">
                  #{r.id} · {fmtDate(r.created_at)} · {r.mode === 'generate' ? 'Generated' : 'Verified'}
                  {r.provider && ` · ${r.provider}${r.is_demo ? ' (demo)' : ''}`} · {r.counts.total} claims ({r.counts.supported} S / {r.counts.contradicted} C / {r.counts.unverifiable} U) · correction: {r.correction_status.replace('_', ' ')} · {fmtMs(r.processing_time_ms)}
                </p>
              </Link>
              <div className="text-right">
                <div className={`text-2xl font-bold ${scoreColor(r.final_reliability_score)}`}>{r.final_reliability_score ?? '—'}</div>
                <div className="text-xs text-slate-500">{r.score_label}</div>
              </div>
              <button className="rounded-lg p-2 text-slate-400 hover:bg-rose-50 hover:text-rose-600" onClick={() => remove(r.id)} aria-label={`Delete run ${r.id}`}>
                <Trash2 className="h-4 w-4" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
