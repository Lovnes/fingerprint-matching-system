# FP-Match: fingerprint screening tool

FP-Match is a local tool for comparing fingerprints and searching a reference database. It helps investigators
**shortlist candidates faster**. A qualified examiner must still verify every candidate (ACE-V).

Everything runs offline on your own machine. No print leaves the computer.

## Features

| | |
|---|---|
| **Sample vs Suspect** | Two slots side by side: the sample (scene/exhibit) print and the suspect's print, with a score, supported/paired minutiae and a side-by-side chart of the paired minutiae |
| **ML verifier** | A neural network compares the ridge images of both prints after alignment; a learned fusion with the minutiae evidence gives the **ML match score** that decides the indication. Runs offline through OpenCV (no GPU, no PyTorch needed) |
| **1:N search** | Search one print against the whole reference database and get ranked candidates. Can be limited to one finger position |
| **Latent mode** | Background flattening, CLAHE, non-local-means denoising, stronger orientation smoothing and stricter quality filters |
| **Image controls** | Resolution (ppi) correction, invert for light ridges, crop to region of interest |
| **Enrolment** | Upload files, or point it at a whole folder (e.g. an FVC or SOCOFing dataset). Subject ID is taken from the file name |
| **Chain of custody** | SHA-256 of every input file, and an audit log of every enrol, delete, compare and search |
| **Reports** | Self-contained HTML case report with case/FIR no., examiner, hashes, settings, results and charts (print to PDF from a browser) |

## Install and run

FP-Match runs on **macOS** (Apple silicon and Intel) and **Windows 10/11**. It opens in your web browser,
but it is only served on this computer (`localhost`) and sends no telemetry.

### Option A: ready-made app (no Python needed)

| | Download | Start |
|---|---|---|
| **macOS** | `FP-Match-macOS-arm64.zip` (Apple silicon) or `FP-Match-macOS-x86_64.zip` (Intel) | Unzip, move **FP-Match.app** to Applications, double-click |
| **Windows** | `FP-Match-Windows.zip` | Unzip (*Extract all*, not just open), double-click **FP-Match.exe** inside the folder |

The app opens your browser at http://localhost:8501.
* **Stop it:** click **⏻ Quit FP-Match** at the bottom of the sidebar. On Windows you can also close the black console window.
* **Closed the tab by mistake?** Open http://localhost:8501 again (on Windows, double-clicking FP-Match.exe also works).
* **First launch warnings.** The app is not code-signed. On **macOS**, right-click FP-Match.app, choose **Open**, then
  **Open** again. If macOS says the app is "damaged", run `xattr -dr com.apple.quarantine /Applications/FP-Match.app`.
  On **Windows**, at "Windows protected your PC", click **More info**, then **Run anyway**.

### Option B: run from this folder (needs Python 3.9+)

| | Start |
|---|---|
| **macOS** | Double-click **`FP-Match.command`** (or run `./run.sh`). If macOS blocks it, right-click it and choose **Open**. |
| **Windows** | Install Python from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH**. Then double-click **`FP-Match.bat`**. |

The first start creates `.venv` and installs the packages (a few minutes, needs internet). Later starts are fast,
and packages are reinstalled automatically if `requirements.txt` changes.

Manual way, on any OS:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # or: python launcher.py
```

### Where the database is kept

| OS | Default location |
|---|---|
| macOS | `~/Library/Application Support/FP-Match/fingerprints.db` |
| Windows | `%LOCALAPPDATA%\FP-Match\fingerprints.db` |

You can change it in the sidebar, or set the environment variable `FPMATCH_DB`. Older installs that already have
`data/fingerprints.db` in this folder keep using it. Back up this file. It holds all enrolled references and
the audit log. The packaged Mac app writes its log (`fpmatch.log`) to the same folder.

### Building the apps

PyInstaller cannot cross-compile, so build each app on its own OS:

```bash
./packaging/build_mac.sh            # macOS   -> dist/FP-Match.app, dist/FP-Match-macOS-<arch>.zip
packaging\build_windows.bat         # Windows -> dist\FP-Match\FP-Match.exe, dist\FP-Match-Windows.zip
```

A Mac build runs only on the CPU type it was built on (arm64 or x86_64). To build all three zips without owning
each machine, push this folder to GitHub and run **Actions → Build apps** (`.github/workflows/build.yml`). Pushing
a tag such as `v1.0.0` also attaches the zips to a GitHub release.

## Quick demo

`samples/synthetic/` already holds 15 synthetic fingers (3 impressions + 1 latent each). You can make more with:

```bash
python -m fpmatch.synth samples/synthetic --fingers 50 --impressions 3 --latents 1
```

1. **Enroll references**, choose *Folder on this computer*, enter `samples/synthetic` (works in the packaged
   app too) and enroll. Subject IDs
   come from the file names.
2. **1:N Search**: upload `samples/synthetic/S005_L1.png` (a simulated latent). Latent mode is on by default.
3. **Sample vs Suspect**: put `S003_L1.png` in the sample slot and `S003_1.png` in the suspect slot (same finger),
   then swap the suspect for `S004_1.png` (different finger).

For real prints, download FVC2002/2004 DB*_B or SOCOFing and enroll the folder.

## How it works

```
image -> resample to 500 ppi -> [latent: flatten + CLAHE + denoise] -> normalise
      -> ROI mask -> ridge orientation (gradient tensor) -> ridge frequency
      -> oriented Gabor filter bank -> binarise -> thin -> prune spurs
      -> minutiae (crossing number) -> remove false minutiae -> template
template A, template B
      -> local structure of each minutia (6 nearest neighbours; rotation/translation invariant)
      -> candidate pairs -> align on each candidate -> one-to-one pairing (12 px, 20°)
      -> least-squares refinement -> weight pairs by neighbourhood agreement -> minutiae score
ML stage (fpmatch/ml.py)
      -> warp the query onto the reference with the matcher's alignment
      -> 192 px window around the overlap: [query ridges, reference ridges, overlap mask] -> 96x96
      -> CNN (7 conv layers, 443k parameters, fpmatch/models/verifier.onnx) -> ridge-agreement logit
      -> logistic-regression fusion with minutiae score / supported / paired / overlap -> ML score 0-100
```

In a 1:N search every reference is scored by the minutiae matcher; the ML stage then re-scores all of them
(databases up to 500 prints) or the best 200, and the list is ranked by ML score.

Code layout:

| File | Purpose |
|---|---|
| `fpmatch/enhance.py` | Segmentation, orientation, frequency, Gabor enhancement, thinning |
| `fpmatch/minutiae.py` | Minutiae extraction and false-minutiae removal |
| `fpmatch/matcher.py` | Matcher, score, decision bands |
| `fpmatch/ml.py` | ML verifier: pair input, OpenCV inference, fusion, ML bands |
| `fpmatch/models/` | Trained model (`verifier.onnx`) and its card (`verifier.json`: training data, fusion weights, thresholds, test results, SHA-256) |
| `training/` | Dataset builder and training script (PyTorch) |
| `fpmatch/database.py` | SQLite reference store, parallel 1:N search, audit log |
| `fpmatch/report.py` | HTML report |
| `fpmatch/synth.py` | Synthetic fingerprint generator for testing |
| `evaluate.py` | EER / FMR / rank-N on any labelled folder |
| `app.py` | Streamlit interface |
| `launcher.py` | Starts the app on a free local port and opens the browser (used by the launchers and the packaged app) |
| `FP-Match.command` / `FP-Match.bat` | Double-click launchers for macOS / Windows |
| `packaging/` | PyInstaller spec and build scripts for the standalone apps |

## Reading the result

* **ML match score** (0-100): decides the indication when the model is loaded (sidebar shows "ML verifier active").
  The bands are stored in `fpmatch/models/verifier.json` and shown on the Help page. It is a similarity score,
  not a probability of identity.
* **Minutiae score** (0-100): the share of the overlapping area that agrees, weighted by neighbourhood agreement.
* **Supported pairs**: paired minutiae whose neighbourhoods also agree. Treat these as the corresponding points.
* Without the model (files missing), the minutiae bands are used: 🟢 *Strong candidate*: score ≥ 25 and ≥ 7
  supported. 🟠 *Possible*: score ≥ 15 or ≥ 7 supported. ⚪ *Not supported*: below both.

## Re-training the ML model

The shipped model was trained on **synthetic** prints only (see `VALIDATION.md`). Re-train it on real prints
before operational use. Files must be named `<subject>_<impression>`; impressions starting with `L` are latents.
Use different subjects for the three sets.

```bash
pip install -r training/requirements-train.txt
python training/build_dataset.py --folder /data/train --out training/work/train.npz
python training/build_dataset.py --folder /data/val   --out training/work/val.npz
python training/build_dataset.py --folder /data/test  --out training/work/test.npz
python training/train.py --train training/work/train.npz --val training/work/val.npz \
                         --test training/work/test.npz --out fpmatch/models
python evaluate.py /data/test                     # minutiae-only vs ML, side by side
```

Synthetic data instead: `--synthetic 800 --seed-base 200000` (train), `--synthetic 200 --seed-base 300000` (val),
`--synthetic 200 --seed-base 400000` (test). Then rebuild the apps so they ship the new model.

## Measured accuracy (synthetic prints)

Held-out test set of 200 synthetic fingers the model never saw (full numbers in `VALIDATION.md`):

| | Minutiae only | With ML verifier |
|---|---|---|
| Plain prints: same finger rated "Strong" | 88.3 % | **98.2 %** |
| Latent prints: same finger rated "Strong" | 32.8 % | **85.7 %** |
| Different fingers rated "Strong" (hardest impostors) | 0 % | 0 % |

**Run `evaluate.py` on real data before relying on the thresholds** (it prints both methods side by side):

```bash
python evaluate.py /path/to/FVC2002/DB1_B            # files named <subject>_<impression>.tif
python evaluate.py /path/to/folder --csv scores.csv   # every comparison, for your own ROC/DET plots
```

Files named like `S003_L1.png` are treated as latent queries for a rank-1/5/10 test.

## Limitations

* **The ML model was trained and tested on synthetic prints only.** Real prints differ (sensors, skin, surfaces,
  development techniques). Re-train it on real data (see *Re-training the ML model*) before operational use.
* Prints that cannot be aligned (very poor latents, roughly fewer than 5 minutiae) get no ML judgement.
* The bands were calibrated on synthetic prints. Re-calibrate them on real prints from your lab.
* Minutiae cannot be edited by hand yet. An examiner-marked latent would give much better results on hard cases.
* WSQ images are not read directly. Convert them to PNG first (e.g. NBIS `dwsq`).
* Results are investigative leads, not identifications.
