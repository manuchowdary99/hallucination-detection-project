import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from 'recharts'
import { Link } from 'react-router-dom'
import { Alert, Spinner } from '../components/ui'
import { STATUS_META, fmtMs, scoreColor } from '../components/format'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'

function Stat({ label, value, cls = 'text-slate-900' }: { label: string; value: string | number; cls?: string }) {
  return (
    <div className="card p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${cls}`}>{value}</div>
    </div>
  )
}

export default function Dashboard() {
  const { data, error, loading } = useApi(api.stats)
  if (loading) return <Spinner label="Loading statistics…" />
  if (error) return <Alert kind="error">{error}</Alert>
  if (!data) return null
  const c = data.counts
  const pie = [
    { name: 'Supported', value: c.supported, color: STATUS_META.SUPPORTED.color },
    { name: 'Contradicted', value: c.contradicted, color: STATUS_META.CONTRADICTED.color },
    { name: 'Unverifiable', value: c.unverifiable, color: STATUS_META.UNVERIFIABLE.color },
    { name: 'Unavailable', value: c.unavailable, color: STATUS_META.VERIFICATION_UNAVAILABLE.color },
  ].filter((d) => d.value > 0)
  const trend = [...data.recent].reverse().map((r) => ({ name: `#${r.id}`, score: r.final_reliability_score ?? r.reliability_score ?? 0, hasScore: (r.final_reliability_score ?? r.reliability_score) != null }))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-sm text-slate-600">Computed live from the verification runs stored in the database.</p>
      </div>
      {data.total_runs === 0 ? (
        <div className="card p-8 text-center text-slate-600">
          No verification runs yet. <Link to="/verify" className="font-medium text-indigo-700">Verify your first answer</Link>.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Stat label="Responses verified" value={data.total_runs} />
            <Stat label="Claims checked" value={data.total_claims} />
            <Stat label="Avg reliability" value={data.average_reliability == null ? '—' : data.average_reliability.toFixed(1)} cls={scoreColor(data.average_reliability)} />
            <Stat label="Avg processing time" value={fmtMs(data.average_latency_ms)} />
            <Stat label="Supported" value={`${c.supported} (${data.percentages.supported ?? 0}%)`} cls="text-emerald-600" />
            <Stat label="Contradicted" value={`${c.contradicted} (${data.percentages.contradicted ?? 0}%)`} cls="text-rose-600" />
            <Stat label="Unverifiable" value={`${c.unverifiable} (${data.percentages.unverifiable ?? 0}%)`} cls="text-amber-600" />
            <Stat label="Corrections attempted" value={data.corrections_attempted} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="card p-4">
              <h2 className="mb-2 font-semibold text-slate-800">Claim verdict distribution</h2>
              {pie.length === 0 ? (
                <p className="text-sm text-slate-500">No claims yet.</p>
              ) : (
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={pie} dataKey="value" nameKey="name" innerRadius={55} outerRadius={95} paddingAngle={2}>
                      {pie.map((d) => <Cell key={d.name} fill={d.color} />)}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </div>
            <div className="card p-4">
              <h2 className="mb-2 font-semibold text-slate-800">Final reliability of recent runs</h2>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={trend}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" fontSize={12} />
                  <YAxis domain={[0, 100]} fontSize={12} />
                  <Tooltip formatter={(v, _n, p) => ((p as { payload?: { hasScore?: boolean } })?.payload?.hasScore ? String(v) : 'insufficient evidence')} />
                  <Bar dataKey="score" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card overflow-x-auto p-4">
            <h2 className="mb-2 font-semibold text-slate-800">Recent verifications</h2>
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-slate-500">
                <tr><th className="py-2">Run</th><th>Query</th><th>Score</th><th>Claims (S/C/U)</th><th>Time</th></tr>
              </thead>
              <tbody>
                {data.recent.map((r) => (
                  <tr key={r.id} className="border-t border-slate-100">
                    <td className="py-2"><Link className="text-indigo-700" to={`/history/${r.id}`}>#{r.id}</Link></td>
                    <td className="max-w-sm truncate">{r.query_summary}</td>
                    <td className={scoreColor(r.final_reliability_score)}>{r.final_reliability_score ?? '—'}</td>
                    <td>{r.counts.supported}/{r.counts.contradicted}/{r.counts.unverifiable}</td>
                    <td>{fmtMs(r.processing_time_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
