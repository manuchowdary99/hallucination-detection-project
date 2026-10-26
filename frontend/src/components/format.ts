import { Ban, CheckCircle2, HelpCircle, XCircle } from 'lucide-react'
import type { ClaimStatus } from '../types/api'

export const STATUS_META: Record<ClaimStatus, { label: string; cls: string; color: string; Icon: typeof CheckCircle2 }> = {
  SUPPORTED: { label: 'Supported', cls: 'bg-emerald-50 text-emerald-800 ring-emerald-200', color: '#10b981', Icon: CheckCircle2 },
  CONTRADICTED: { label: 'Contradicted', cls: 'bg-rose-50 text-rose-800 ring-rose-200', color: '#f43f5e', Icon: XCircle },
  UNVERIFIABLE: { label: 'Unverifiable', cls: 'bg-amber-50 text-amber-800 ring-amber-200', color: '#f59e0b', Icon: HelpCircle },
  VERIFICATION_UNAVAILABLE: { label: 'Verification unavailable', cls: 'bg-slate-100 text-slate-700 ring-slate-300', color: '#94a3b8', Icon: Ban },
}

export function scoreColor(score: number | null) {
  if (score === null) return 'text-slate-500'
  return score >= 75 ? 'text-emerald-600' : score >= 50 ? 'text-amber-600' : 'text-rose-600'
}

export const fmtPct = (v: number) => `${(v * 100).toFixed(0)}%`
export const fmtMs = (v: number | null | undefined) => (v == null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`)
export const fmtDate = (s: string) => new Date(s).toLocaleString()
