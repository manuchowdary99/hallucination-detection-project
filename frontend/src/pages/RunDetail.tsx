import { ArrowLeft } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import RunView from '../components/RunView'
import { Alert, Spinner } from '../components/ui'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'

export default function RunDetail() {
  const id = Number(useParams().id)
  const { data, error, loading, setData } = useApi(() => api.run(id), [id])
  const [correcting, setCorrecting] = useState(false)
  const [corrError, setCorrError] = useState<string | null>(null)

  async function correct() {
    setCorrecting(true)
    setCorrError(null)
    try {
      setData(await api.correct(id))
    } catch (e) {
      setCorrError(e instanceof Error ? e.message : String(e))
    } finally {
      setCorrecting(false)
    }
  }

  return (
    <div className="space-y-4">
      <Link to="/history" className="inline-flex items-center gap-1 text-sm text-indigo-700"><ArrowLeft className="h-4 w-4" aria-hidden /> Back to history</Link>
      {loading && <Spinner label="Loading run…" />}
      {error && <Alert kind="error">{error}</Alert>}
      {corrError && <Alert kind="error">{corrError}</Alert>}
      {data && <RunView run={data} onCorrect={correct} correcting={correcting} />}
    </div>
  )
}
