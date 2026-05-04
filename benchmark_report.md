# Qdrant Benchmark Report

- Generated at: `2026-05-30T01:31:34+02:00`
- Source collection: `dbpedia_100K`
- Test queries: `20`

## Recommendation

- Recommended quantized method: `scalar` with score `0.730`.
- Fastest average latency: `scalar` (84.75ms).
- Highest measured accuracy: `scalar` (80.50%).

## Comparison

| Method | Avg | P95 | Avg speedup | P95 speedup | Accuracy | Rescore acc. | Score | Compression |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 76.83ms | 98.72ms | 1.00x | 1.00x | 100.00% | - | 1.000 | - |
| scalar | 84.75ms | 99.80ms | 0.91x | 0.99x | 80.50% | - | 0.730 | 4x |
| binary | 86.23ms | 99.83ms | 0.89x | 0.99x | 8.00% | 10.50% | 0.094 | 32x |
| binary_2bit | 89.34ms | 99.68ms | 0.86x | 0.99x | 6.00% | 9.50% | 0.082 | 16x |

## How To Read This

- Lower latency is better.
- Higher speedup is better; `1.00x` means equal to baseline.
- Accuracy is overlap with baseline search results.
- Score is `average speedup x best measured accuracy`; use it as a balanced starting point, not as an absolute truth.
