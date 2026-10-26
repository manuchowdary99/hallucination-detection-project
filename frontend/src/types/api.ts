export type ClaimStatus = 'SUPPORTED' | 'CONTRADICTED' | 'UNVERIFIABLE' | 'VERIFICATION_UNAVAILABLE'
export type InputMode = 'verify' | 'generate'
export type CorrectionStatus = 'not_requested' | 'not_needed' | 'corrected' | 'no_changes' | 'failed'

export interface NLIScores { entailment: number; contradiction: number; neutral: number }
export interface Entity { text: string; label: string }

export interface Evidence {
  source_name: string
  title: string
  url: string
  snippet: string
  retrieved_at: string | null
  relevance: number
  nli: NLIScores | null
  label: ClaimStatus | null
}

export interface ClaimResult {
  index: number
  text: string
  source_sentence: string | null
  entities: Entity[]
  status: ClaimStatus
  confidence: number | null
  model_confidence: number | null
  nli: NLIScores | null
  explanation: string
  evidence: Evidence[]
  conflicting_evidence: boolean
  changed_in_correction: boolean
}

export interface StatusCounts { supported: number; contradicted: number; unverifiable: number; unavailable: number; total: number }

export interface ScoreBreakdown {
  score: number | null
  label: string
  counts: StatusCounts
  components: Record<string, number>
  formula: string
}

export interface VerificationResult { claims: ClaimResult[]; score: ScoreBreakdown; warnings: string[]; latency_ms: number }
export interface ProviderInfo { name: string; model: string | null; is_demo: boolean }

export interface EvidenceRef { source_name: string; title: string; url: string }

export interface ClaimChange {
  original_text: string | null
  original_status: ClaimStatus | null
  corrected_text: string | null
  corrected_status: ClaimStatus | null
  sources: EvidenceRef[]
}

export interface CorrectionInfo {
  status: CorrectionStatus
  message: string | null
  provider: ProviderInfo | null
  changed_claims: number
  score_before: number | null
  score_after: number | null
  changes?: ClaimChange[]
  unsupported_changed_claims?: number
}

export interface RunResponse {
  id: number
  created_at: string
  mode: InputMode
  question: string | null
  provider: ProviderInfo | null
  original_answer: string
  final_answer: string
  original_verification: VerificationResult
  corrected_verification: VerificationResult | null
  correction: CorrectionInfo
  processing_time_ms: number
  warnings: string[]
}

export interface HistoryItem {
  id: number
  created_at: string
  mode: InputMode
  query_summary: string
  reliability_score: number | null
  final_reliability_score: number | null
  score_label: string
  counts: StatusCounts
  correction_status: CorrectionStatus
  processing_time_ms: number
  provider: string | null
  is_demo: boolean
}

export interface StatsResponse {
  total_runs: number
  total_claims: number
  counts: StatusCounts
  percentages: Record<string, number>
  average_latency_ms: number | null
  average_reliability: number | null
  corrections_attempted: number
  recent: HistoryItem[]
}

export interface ComponentHealth { status: string; detail: string | null }
export interface HealthResponse {
  status: string
  llm: { default_provider: string; openai_configured: boolean; openai_model: string | null; demo_available: boolean }
  nli: ComponentHealth
  nlp: ComponentHealth
  knowledge_providers: string[]
  database: ComponentHealth
}
