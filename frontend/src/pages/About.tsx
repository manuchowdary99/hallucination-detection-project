const steps = [
  ['Input', 'You paste an AI response (optionally with the question), or ask a question. Inputs are validated and limited to 8,000 characters.'],
  ['Optional generation', 'In generate mode an answer is produced by the configured provider: an OpenAI-compatible API if a key is set, otherwise a deterministic offline demo provider that is clearly labelled as such.'],
  ['Claim extraction', 'spaCy segments sentences and uses the dependency parse to split coordinated clauses into atomic claims, resolve leading pronouns to the previous subject, and skip questions, opinions and fragments.'],
  ['NER and retrieval queries', 'Named entities (people, places, organisations, dates, quantities) are extracted from each claim and used to build search queries.'],
  ['Wikipedia / Wikidata evidence', 'The MediaWiki search and extracts APIs and Wikidata (entity search + SPARQL) are queried with timeouts, retries and caching. Passages are ranked by lexical relevance to the claim and sanitised; retrieved text is treated as untrusted data, never as instructions.'],
  ['NLI classification', 'A DeBERTa-v3 NLI model scores each (evidence → claim) pair for entailment, contradiction and neutrality. Strong entailment from relevant evidence ⇒ SUPPORTED; strong contradiction that clearly outweighs any entailment ⇒ CONTRADICTED; otherwise or with no evidence ⇒ UNVERIFIABLE. If the model cannot load, claims are marked VERIFICATION_UNAVAILABLE rather than guessed.'],
  ['Confidence scoring', 'Reliability = 100 × clamp(Σsupported(entailment·q)/N + 0.25·U/N − 0.5·Σcontradicted(contradiction·q)/N, 0, 1), where q = 0.5 + 0.5·evidence relevance and N is the number of claims. When there are no claims or no claim could be decided, the score is reported as unavailable instead of a number.'],
  ['Optional correction', 'Contradicted and unverifiable claims are sent, with their delimited evidence, to the provider, which is instructed to change only those claims, use only the evidence, and remove claims it cannot support.'],
  ['Re-verification', 'The corrected answer goes through the full pipeline again so you can compare scores and see which claims changed.'],
]

export default function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Methodology</h1>
        <p className="mt-1 text-slate-600">How VeriAI turns a free-text answer into claim-level, evidence-backed verdicts.</p>
      </div>
      <ol className="space-y-4">
        {steps.map(([title, text], i) => (
          <li key={title} className="card flex gap-4 p-4">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-sm font-bold text-white">{i + 1}</span>
            <div>
              <h2 className="font-semibold text-slate-900">{title}</h2>
              <p className="mt-1 text-sm text-slate-600">{text}</p>
            </div>
          </li>
        ))}
      </ol>
      <section className="card space-y-2 p-5 text-sm text-slate-600">
        <h2 className="text-base font-semibold text-slate-900">Limitations</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>Wikipedia and Wikidata can be incomplete, outdated or wrong; absence of evidence is reported as UNVERIFIABLE, not false.</li>
          <li>NLI models make mistakes, especially with numbers, negation, long passages and paraphrase.</li>
          <li>Claim extraction is rule-based and can split or merge claims imperfectly.</li>
          <li>The reliability score is a heuristic, not a calibrated probability.</li>
          <li>Not validated for medical, legal or financial use.</li>
        </ul>
      </section>
    </div>
  )
}
