import { AlertTriangle, Loader2 } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ClaimStatus } from '../types/api'
import { STATUS_META } from './format'

export function StatusBadge({ status }: { status: ClaimStatus }) {
  const m = STATUS_META[status]
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${m.cls}`}>
      <m.Icon className="h-3.5 w-3.5" aria-hidden />
      {status}
    </span>
  )
}

export function Alert({ kind = 'warning', children }: { kind?: 'warning' | 'error' | 'info'; children: ReactNode }) {
  const cls = {
    warning: 'border-amber-200 bg-amber-50 text-amber-900',
    error: 'border-rose-200 bg-rose-50 text-rose-900',
    info: 'border-indigo-200 bg-indigo-50 text-indigo-900',
  }[kind]
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={`flex gap-2 rounded-lg border px-3 py-2 text-sm ${cls}`}>
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>{children}</div>
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-600">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {label}
    </span>
  )
}
