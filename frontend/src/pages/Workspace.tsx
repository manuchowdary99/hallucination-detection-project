import { Sparkles, ShieldCheck } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import RunView from '../components/RunView'
import { Alert, Spinner } from '../components/ui'
import { api } from '../services/api'
import type { HealthResponse, InputMode, RunResponse } from '../types/api'

const STAGES = [
  'Extracting factual claims…',
  'Recognising entities…',
  'Retrieving evidence from Wikipedia and Wikidata…',
  'Running natural-language inference…',
  'Computing reliability score…',
]

const EXAMPLES = [
  'Tell me about the Eiffel Tower.',
  'Who was Albert Einstein?',
  'Describe the Apollo 11 mission.',
  'How tall is Mount Everest?',
]

export default function Workspace() {
  const [mode, setMode] = useState<InputMode>('generate')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [provider, setProvider] = useState('demo')
  const [autoCorrect, setAutoCorrect] = useState(false)
  const [busy, setBusy] = useState(false)
  const [correcting, setCorrecting] = useState(false)
  const [stage, setStage] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [run, setRun] = useState<RunResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const timer = useRef<number>()

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    if (!busy) return
    setStage(0)
    timer.current = window.setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 2500)
    return () => window.clearInterval(timer.current)
  }, [busy])

  const openaiReady = !!health?.llm.openai_configured
  const text = mode === 'generate' ? question : answer
  const tooLong = text.length > 8000

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    if (!text.trim()) {
      setError(mode === 'generate' ? 'Enter a question to generate an answer.' : 'Paste the AI response you want to verify.')
      return
    }
    setBusy(true)
    try {
      const r =
        mode === 'generate'
          ? await api.generateAndVerify(question.trim(), provider, autoCorrect)
          : await api.verify(answer.trim(), question.trim() || null, autoCorrect)
      setRun(r)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function correct() {
    if (!run) return
    setCorrecting(true)
    setError(null)
    try {
      setRun(await api.correct(run.id, mode === 'generate' ? provider : undefined))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setCorrecting(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Verification workspace</h1>
        <p className="text-sm text-slate-600">Generate an answer and verify it, or paste an existing AI response.</p>
      </div>
      {health && ['unavailable', 'disabled'].includes(health.nli.status) && (
        <Alert>NLI model status: {health.nli.status}. {health.nli.detail} Claims with evidence will be marked VERIFICATION_UNAVAILABLE.</Alert>
      )}
      <form onSubmit={submit} className="card space-y-4 p-5" aria-label="Verification form">
        <div role="tablist" className="inline-flex rounded-lg bg-slate-100 p-1">
          {(['generate', 'verify'] as InputMode[]).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              onClick={() => setMode(m)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${mode === m ? 'bg-white text-indigo-700 shadow' : 'text-slate-600'}`}
            >
              {m === 'generate' ? 'Generate answer' : 'Verify existing answer'}
            </button>
          ))}
        </div>

        <div>
          <label htmlFor="question" className="label">
            {mode === 'generate' ? 'Question' : 'Original question (optional)'}
          </label>
          <input id="question" className="input" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. Tell me about the Eiffel Tower." />
          {mode === 'generate' && provider === 'demo' && (
            <div className="mt-2 flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button key={ex} type="button" className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-700 hover:bg-slate-200" onClick={() => setQuestion(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          )}
        </div>

        {mode === 'verify' && (
          <div>
            <label htmlFor="answer" className="label">AI response to verify</label>
            <textarea id="answer" rows={7} className="input" value={answer} onChange={(e) => setAnswer(e.target.value)} placeholder="Paste the AI-generated answer here…" />
          </div>
        )}

        <div className="flex flex-wrap items-end gap-6">
          {mode === 'generate' && (
            <div>
              <label htmlFor="provider" className="label">Answer provider</label>
              <select id="provider" className="input" value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="demo">Demo provider (offline, deterministic)</option>
                <option value="openai" disabled={!openaiReady}>
                  OpenAI-compatible {openaiReady ? `(${health?.llm.openai_model})` : '(not configured)'}
                </option>
              </select>
            </div>
          )}
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input type="checkbox" className="h-4 w-4 rounded border-slate-300" checked={autoCorrect} onChange={(e) => setAutoCorrect(e.target.checked)} />
            Verify and correct (rewrite unreliable claims, then re-verify)
          </label>
        </div>
        {mode === 'generate' && provider === 'demo' && (
          <p className="text-xs text-slate-500">
            The demo provider returns canned answers that intentionally contain some errors so the verifier has something to catch. It is not a real LLM.
          </p>
        )}
        {tooLong && <Alert kind="error">Input is {text.length} characters; the limit is 8000.</Alert>}
        <div className="flex items-center gap-4">
          <button type="submit" className="btn-primary" disabled={busy || tooLong}>
            {mode === 'generate' ? <Sparkles className="h-4 w-4" aria-hidden /> : <ShieldCheck className="h-4 w-4" aria-hidden />}
            {mode === 'generate' ? 'Generate & verify' : 'Verify'}
          </button>
          {busy && <Spinner label={STAGES[stage]} />}
        </div>
        {busy && <p className="text-xs text-slate-500">The first request can take longer while the NLI model loads.</p>}
      </form>

      {error && <Alert kind="error">{error}</Alert>}
      {run && !busy && <RunView run={run} onCorrect={correct} correcting={correcting} />}
    </div>
  )
}
