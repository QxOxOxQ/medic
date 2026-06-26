# Qdrant Benchmark Report

- Generated at: `2026-06-02T12:25:11+02:00`
- Source collection: `hybrid_medic_documents`
- Test queries: `20`

## Recommendation

- Recommended quantized method: `scalar` with score `0.953`.
- Fastest average latency: `binary_2bit` (43.18ms).
- Highest measured accuracy: `scalar` (99.00%).

## Comparison

| Method | Avg | P95 | Avg speedup | P95 speedup | Accuracy | Rescore acc. | Score | Compression |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 43.70ms | 45.95ms | 1.00x | 1.00x | 100.00% | - | 1.000 | - |
| scalar | 45.41ms | 53.62ms | 0.96x | 0.86x | 99.00% | - | 0.953 | 4x |
| binary | 43.55ms | 46.30ms | 1.00x | 0.99x | 86.00% | 94.00% | 0.943 | 32x |
| binary_2bit | 43.18ms | 45.44ms | 1.01x | 1.01x | 87.00% | 93.50% | 0.946 | 16x |

## How To Read This

- Lower latency is better.
- Higher speedup is better; `1.00x` means equal to baseline.
- Accuracy is overlap with baseline search results.
- Score is `average speedup x best measured accuracy`; use it as a balanced starting point, not as an absolute truth.
