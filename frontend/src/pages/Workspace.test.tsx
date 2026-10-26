import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Workspace from './Workspace'
import type { RunResponse } from '../types/api'

const score = (s: number | null, sup: number, con: number) => ({
  score: s, label: s == null ? 'Insufficient evidence' : 'Moderate', formula: 'f', components: {},
  counts: { supported: sup, contradicted: con, unverifiable: 0, unavailable: 0, total: sup + con },
})

const run: RunResponse = {
  id: 7, created_at: '2026-01-01T00:00:00Z', mode: 'verify', question: null, provider: null,
  original_answer: 'The Eiffel Tower is in Paris. It was completed in 1925.',
  final_answer: 'The Eiffel Tower is in Paris. It was completed in 1925.',
  original_verification: {
    latency_ms: 10, warnings: [], score: score(48, 1, 1),
    claims: [
      { index: 0, text: 'The Eiffel Tower is in Paris.', source_sentence: null, entities: [], status: 'SUPPORTED', confidence: 92, model_confidence: 97, nli: { entailment: 0.97, contradiction: 0.01, neutral: 0.02 }, explanation: 'Entailed by Wikipedia.', conflicting_evidence: false, changed_in_correction: false,
        evidence: [{ source_name: 'Wikipedia', title: 'Eiffel Tower', url: 'https://en.wikipedia.org/wiki/Eiffel_Tower', snippet: 'The Eiffel Tower is in Paris.', retrieved_at: null, relevance: 0.9, nli: null, label: 'SUPPORTED' }] },
      { index: 1, text: 'The Eiffel Tower was completed in 1925.', source_sentence: null, entities: [], status: 'CONTRADICTED', confidence: 90, model_confidence: null, nli: null, explanation: 'Contradicted: completed in 1889.', conflicting_evidence: false, changed_in_correction: false, evidence: [] },
    ],
  },
  corrected_verification: null,
  correction: { status: 'not_requested', message: null, provider: null, changed_claims: 0, score_before: null, score_after: null },
  processing_time_ms: 1234, warnings: [],
}

function mockFetch(handler: (url: string, init?: RequestInit) => Response) {
  const fn = vi.fn((url: string, init?: RequestInit) => Promise.resolve(handler(url, init)))
  vi.stubGlobal('fetch', fn)
  return fn
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

afterEach(() => vi.unstubAllGlobals())

describe('Workspace', () => {
  it('verifies an existing answer and renders claims with statuses and evidence links', async () => {
    const fetchMock = mockFetch((url) => (url.includes('/api/health') ? json({ nli: { status: 'ok' }, llm: { openai_configured: false } }) : json(run, 201)))
    render(<MemoryRouter><Workspace /></MemoryRouter>)
    fireEvent.click(screen.getByRole('tab', { name: 'Verify existing answer' }))
    fireEvent.change(screen.getByLabelText('AI response to verify'), { target: { value: run.original_answer } })
    fireEvent.click(screen.getByRole('button', { name: /^Verify$/ }))

    expect(await screen.findAllByTestId('claim-card')).toHaveLength(2)
    expect(screen.getByText('CONTRADICTED')).toBeInTheDocument()
    expect(screen.getByText('48')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /1 evidence passage$/ }))
    expect(screen.getByRole('link', { name: /Wikipedia: Eiffel Tower/ })).toHaveAttribute('href', 'https://en.wikipedia.org/wiki/Eiffel_Tower')
    const verifyCall = fetchMock.mock.calls.find(([u]) => u.includes('/api/verify'))!
    expect(JSON.parse(verifyCall[1]!.body as string)).toMatchObject({ response_text: run.original_answer, auto_correct: false })
  })

  it('shows validation error for empty input without calling the API', async () => {
    const fetchMock = mockFetch(() => json({ nli: { status: 'ok' }, llm: { openai_configured: false } }))
    render(<MemoryRouter><Workspace /></MemoryRouter>)
    fireEvent.click(screen.getByRole('button', { name: /Generate & verify/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Enter a question')
    await waitFor(() => expect(fetchMock.mock.calls.every(([u]) => u.includes('/api/health'))).toBe(true))
  })

  it('surfaces structured backend errors', async () => {
    mockFetch((url) =>
      url.includes('/api/health')
        ? json({ nli: { status: 'ok' }, llm: { openai_configured: false } })
        : json({ error: { code: 'provider_error', message: 'LLM provider request failed.' } }, 502),
    )
    render(<MemoryRouter><Workspace /></MemoryRouter>)
    fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'Who was Einstein?' } })
    fireEvent.click(screen.getByRole('button', { name: /Generate & verify/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('LLM provider request failed.')
  })
})
