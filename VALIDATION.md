# Validation summary - FP-Match v1.0.0

Test date: 18 Nov 2026. Command: `python evaluate.py <folder>`.

## Synthetic set (40 fingers x 3 plain impressions + 1 latent)
Generated with `fpmatch/synth.py`. Impressions differ by rotation (±25°), translation, elastic skin distortion,
pressure, noise and contact area. Latents are small partial prints (rotation up to ±60°) on a textured background.

| Metric | Result |
|---|---|
| Genuine / impostor comparisons | 120 / 7,020 |
| Equal error rate | ~2.6 % (score 7.8) |
| Genuine score | 5th pct 19.0, median 40.0 |
| Impostor score | 99th pct 9.1, max 15.1 |
| Impostor supported pairs (max) | 6 |
| "Strong" band | 87.5 % of genuine, 0 % of impostors |
| "Possible" or better | 97.5 % of genuine, 0.014 % of impostors |
| Latent 1:N rank-1 / 5 / 10 | 90.0 % / 95.0 % / 97.5 % |
| Speed (laptop CPU) | ~0.12 s to process a print, ~5 ms per comparison |

Nearly all genuine misses come from one finger where only about 5 minutiae were extracted (failure to extract).
The app warns whenever a print has fewer than 12 minutiae.

## Real prints (small check)
Two FVC-style genuine pairs and four impostor pairs (4 prints): both genuine pairs "Strong" (scores 41.9 and 28.4; 10 and 9 supported pairs). All impostor pairs scored 0.5-2.1 with 0 supported pairs.

## Caveats
Synthetic prints are cleaner and more alike than real casework prints. Before operational use, re-run
`evaluate.py` on FVC2002/2004, SOCOFing, or your lab's own ground-truth prints and adjust `STRONG_SCORE`,
`STRONG_SUPPORT` and `POSSIBLE_SCORE` in `fpmatch/matcher.py`.
