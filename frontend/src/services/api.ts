import type { HealthResponse, HistoryItem, RunResponse, StatsResponse } from '../types/api'

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiError(0, 'network_error', 'Cannot reach the VeriAI backend. Is it running on port 8000?')
  }
  if (res.status === 204) return undefined as T
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const err = body?.error
    const detail = Array.isArray(err?.details) && err.details.length ? ` (${err.details[0].msg})` : ''
    throw new ApiError(res.status, err?.code ?? 'http_error', (err?.message ?? `Request failed with HTTP ${res.status}`) + detail)
  }
  return body as T
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),
  verify: (response_text: string, question: string | null, auto_correct: boolean) =>
    request<RunResponse>('/api/verify', { method: 'POST', body: JSON.stringify({ response_text, question, auto_correct }) }),
  generateAndVerify: (question: string, provider: string, auto_correct: boolean) =>
    request<RunResponse>('/api/generate-and-verify', { method: 'POST', body: JSON.stringify({ question, provider, auto_correct }) }),
  correct: (run_id: number, provider?: string) =>
    request<RunResponse>('/api/correct', { method: 'POST', body: JSON.stringify({ run_id, provider }) }),
  history: () => request<HistoryItem[]>('/api/history?limit=200'),
  run: (id: number) => request<RunResponse>(`/api/history/${id}`),
  deleteRun: (id: number) => request<void>(`/api/history/${id}`, { method: 'DELETE' }),
  clearHistory: () => request<{ deleted: number }>('/api/history?confirm=true', { method: 'DELETE' }),
  stats: () => request<StatsResponse>('/api/stats'),
}
