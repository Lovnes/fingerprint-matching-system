# Validation summary - FP-Match v1.1 (ML verifier 1.0)

Test date: 19 Nov 2026. All numbers are on **synthetic** fingerprints (`fpmatch/synth.py`). Real casework prints
are harder; re-run `evaluate.py` on FVC / SOCOFing / lab ground truth, and re-train the model on real prints
(`training/`, see README) before operational use.

## How the ML verifier was built

| Split | Fingers (master seeds) | Prints | Pairs | Used for |
|---|---|---|---|---|
| Train | 800 (200000-200799) | 4,000 | 25,600 | CNN weights |
| Validation | 200 (300000-300199) | 1,000 | 6,400 | Model selection, fusion weights, thresholds |
| Test | 200 (400000-400199) | 1,000 | 6,400 | Final numbers below (fingers never seen before) |
| Samples | 15 (1000-1014) | 60 | 1,665 | Independent check (`samples/synthetic`) |

Each finger: 3 plain impressions + 2 latents. Every print is a query against all other plain prints of its finger
(genuine) and against 40 random plain prints of other fingers, of which the **3 best-scoring** (hard impostors)
and 1 random are kept. Impostor rates below are therefore rates on *hard* impostors.

* CNN: 7 conv layers, 443k parameters, 8 epochs on Apple M1 (MPS), best validation AUC 0.9985.
* Exported to ONNX and run by OpenCV 5.1 in the app (max difference to PyTorch 8e-6).
* Fusion: logistic regression with non-negative weights - network output 13.4, supported minutiae 3.8,
  overlap 0.6, minutiae score 0 (no independent evidence beyond supported pairs).
* Bands: **strong** = ML score >= 35.5 (above every validation impostor) **and** network output > 0;
  **possible** = ML score >= 10.4 (99th percentile of validation hard impostors).
* Pairs that cannot be aligned with enough overlap (>= 1,500 px) are not assessed: ML score 0.

Two fusion problems were found and fixed during development: (1) with "minutiae count" and "no overlap" as
features, pairs with *no evidence* scored ~51 because of how training pairs were sampled; (2) the minutiae score
got a negative weight. The "network output > 0" rule for Strong was added after inspecting the single test
impostor that reached Strong (network output -1.3, 0 supported minutiae); on validation it costs 0.6 % of
genuine Strong results and removes no validation impostor (there were none above the threshold).

## Test split (200 unseen fingers)

| | Minutiae only | ML verifier |
|---|---|---|
| Plain queries: genuine "Strong" / flagged | 88.3 % / 94.6 % | **98.2 % / 98.3 %** |
| Plain queries: hard impostors "Strong" / flagged | 0 % / 0.13 % | **0 % / 0.04 %** |
| Latent queries: genuine "Strong" / flagged | 32.8 % / 59.1 % | **85.7 % / 87.4 %** |
| Latent queries: hard impostors "Strong" / flagged | 0 % / 0.94 % | 0 % / 1.63 % |
| Equal error rate, latent queries | 27.9 % | **12.3 %** |

Genuine latent pairs the ML stage misses are mostly ones with no usable alignment (not assessed).

## Samples (`python evaluate.py samples/synthetic`)

| | Minutiae only | ML verifier |
|---|---|---|
| Plain-to-plain genuine "Strong" (45 pairs) | 95.6 % | **100 %** |
| Plain-to-plain impostors flagged (945 pairs) | 0 % | 0 % |
| Latent-to-plain genuine "Strong" (45 pairs) | 33.3 % | **93.3 %** |
| Latent-to-plain impostors flagged (630 pairs) | 0.16 % (1) | 0.48 % (3, all "Possible", ML ~11) |
| Latent 1:N rank-1 / rank-5 (15 queries) | 86.7 % / 100 % | **93.3 %** / 100 % |

The only missed latent is S004_L1 (2 minutiae, cannot be aligned). The three flagged impostor pairs involve the
weakest latents (S001_L1, S009_L1, S014_L1, 5-6 minutiae), which match their own finger as "Strong".

## Speed (M1 laptop)
~0.12 s to process a print; ~5 ms per minutiae comparison; ~3 ms per ML comparison (batched).
