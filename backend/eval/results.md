# VeriAI evaluation results

Run at 2026-10-26T19:18:18+00:00 on 36 labelled claims (live Wikipedia/Wikidata + MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

- Accuracy: **0.889**
- Macro F1: **0.868**
- Hallucination (CONTRADICTED) detection — precision 1.0, recall 0.846, F1 0.917
- Latency per claim — mean 2583 ms, median 2361 ms, max 5986 ms

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTED | 1.0 | 0.882 | 0.938 | 17 |
| CONTRADICTED | 1.0 | 0.846 | 0.917 | 13 |
| UNVERIFIABLE | 0.6 | 1.0 | 0.75 | 6 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTED | CONTRADICTED | UNVERIFIABLE | VERIFICATION_UNAVAILABLE |
|---|---|---|---|---|
| SUPPORTED | 15 | 0 | 2 | 0 |
| CONTRADICTED | 0 | 11 | 2 | 0 |
| UNVERIFIABLE | 0 | 0 | 6 | 0 |

## Self-correction (demo provider)

4 of 4 demo answers containing contradicted claims had fewer contradicted claims after correction (success rate 1.0).

| Question | Status | Score before | Score after | Contradicted before → after |
|---|---|---|---|---|
| Tell me about the Eiffel Tower. | corrected | 44.3 | 81.0 | 1 → 0 |
| Who was Albert Einstein? | corrected | 61.7 | 97.4 | 1 → 0 |
| Describe the Apollo 11 mission. | no_changes | 58.2 | None | 0 → 0 |
| What is Python? | corrected | 57.7 | 92.9 | 1 → 0 |
| Tell me about the Great Wall of China. | corrected | 21.1 | 70.0 | 1 → 0 |
| How tall is Mount Everest? | not_needed | 99.5 | None | 0 → 0 |

## Misclassified claims

| Claim | Gold | Predicted |
|---|---|---|
| Python was created by James Gosling. | CONTRADICTED | UNVERIFIABLE |
| Isaac Newton was born in 1643. | SUPPORTED | UNVERIFIABLE |
| The Titanic sank in 1912. | SUPPORTED | UNVERIFIABLE |
| The Titanic sank in 1925. | CONTRADICTED | UNVERIFIABLE |

Small, hand-built dataset: treat these numbers as a sanity check, not a benchmark result.
