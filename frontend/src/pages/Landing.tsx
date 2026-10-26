import { ArrowRight, BookOpenCheck, Gauge, SearchCheck, ShieldCheck, Wand2 } from 'lucide-react'
import { Link } from 'react-router-dom'

const features = [
  { Icon: SearchCheck, title: 'Claim verification', text: 'Answers are split into atomic factual claims and each claim is judged independently with a natural-language-inference model.' },
  { Icon: BookOpenCheck, title: 'Evidence retrieval', text: 'Named entities drive searches against Wikipedia and Wikidata. Every verdict links to the passage that produced it.' },
  { Icon: Gauge, title: 'Confidence scoring', text: 'A transparent, documented formula turns claim verdicts and evidence quality into a 0–100 reliability estimate.' },
  { Icon: Wand2, title: 'Self-correction', text: 'Contradicted or unsupported claims are rewritten using only retrieved evidence, then the new answer is verified again.' },
]

export default function Landing() {
  return (
    <div className="space-y-16">
      <section className="mx-auto max-w-3xl pt-8 text-center">
        <ShieldCheck className="mx-auto h-14 w-14 text-indigo-600" aria-hidden />
        <h1 className="mt-4 text-5xl font-extrabold tracking-tight text-slate-900">VeriAI</h1>
        <p className="mt-3 text-xl font-medium text-indigo-700">Trust Every Claim. Verify Every Answer.</p>
        <p className="mt-6 text-slate-600">
          Large language models sometimes <strong>hallucinate</strong>: they produce fluent, confident statements that are
          not supported by any real source — a wrong date, an invented award, a misplaced landmark. VeriAI is a
          verification layer that sits between the model and the reader. It extracts each factual claim, retrieves
          evidence from trusted knowledge sources, classifies the claim as supported, contradicted or unverifiable, and
          shows you exactly why.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link to="/verify" className="btn-primary px-6 py-3 text-base">
            Start verifying <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
          <Link to="/about" className="btn-secondary px-6 py-3 text-base">How it works</Link>
        </div>
      </section>
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {features.map(({ Icon, title, text }) => (
          <div key={title} className="card p-5">
            <Icon className="h-8 w-8 text-indigo-600" aria-hidden />
            <h2 className="mt-3 font-semibold text-slate-900">{title}</h2>
            <p className="mt-1 text-sm text-slate-600">{text}</p>
          </div>
        ))}
      </section>
      <section className="card mx-auto max-w-3xl p-6 text-sm text-slate-600">
        <h2 className="text-base font-semibold text-slate-900">What VeriAI does not claim</h2>
        <p className="mt-2">
          Missing evidence is reported as <em>unverifiable</em>, never as false. Scores are heuristic evidence-coverage
          estimates, not guarantees, and the knowledge sources themselves can be incomplete or outdated.
        </p>
      </section>
    </div>
  )
}
