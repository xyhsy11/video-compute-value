# View the experiments with VidFit

The project-local browser presents the existing E0, E1 and E2 evidence in tabs:
experiment designs, accuracy charts, E0 descriptive timing and memory, E1's full
anonymous correctness matrix, E2A's held-out folds and frozen advancement gate,
source hashes, and the historical resource snapshot. No new scientific collection
is performed. E0 remains `MEASUREMENT_UNSTABLE`; E2B remains
`NOT_ADMITTED_E2A_ADVANCEMENT`.

From this repository on the existing Windows installation:

```powershell
.\scripts\open_vidfit_experiments.ps1
```

Open <http://127.0.0.1:7861>. The launcher uses the separate VidFit environment at
`C:\Vidfit\.venv` and its local checkout. It does not install packages or change
either experiment runtime. Override `-VidfitRoot` or `-Port` when needed. Stop the
server with Ctrl+C in its terminal.

For a portable report without UI dependencies:

```powershell
.\.venv\Scripts\python.exe scripts\vidfit_experiments.py
```

Open `build/vidfit-experiments/index.html`. The HTML is self-contained; its source
links point to the frozen public evidence commit. `evidence.json` contains the
public chart data and source hashes. These generated files stay in ignored
`build/`. A new export replaces only these presentation outputs.

The viewer reads an explicit allowlist of public result files. It verifies saved
E1/E2 summary hashes and the E1 correctness hash, recomputes E1/E2 accuracy from
anonymous public correctness, and checks the frozen E2A advancement rule. Missing
or inconsistent records block the view instead of becoming zero-valued results.
This is saved-record verification, not a fresh processor/GPU validation.

The **VidFit · Inspect frames** tab calls VidFit's real `VideoAnalyzer.preview`
with the `inspect` backend. It displays actual source frames, timestamps, indices
and RGB hashes from a local uploaded video. It loads no model weights or processor
and offers no inference action. This preview uses VidFit's uniform sampling and
does not reproduce the frozen E0/E1/E2 preprocessing. The gallery is separate from
the historical export. Uploaded media remains local and must not be committed.

The resource bars are explicitly dated historical closeout observations, not a
live resource balance. Presentation work is accounted in a separate local receipt,
without editing any historical experiment ledger. E2A's scientific gate continues
to block E2B regardless of the remaining budget.

The [validation receipt](vidfit-validation.json) records the saved-record checks,
targeted CPU tests, real VidFit CPU preview, browser interactions, resource lineage
and historical preservation. This presentation does not exercise a new model,
processor or GPU path. Any future project work must inherit the cumulative charge
in `.local/vidfit_presentation_resource_ledger.json` as well as earlier history.
