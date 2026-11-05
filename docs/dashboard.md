# FraudNet dashboard

The existing FastAPI app serves the dashboard at `/`. Start it from the project
root with `.venv/bin/python -m uvicorn api.main:app --port 8000`.
There is no frontend package installation or build step.

## Workspace

- **Investigations:** enter an internal graph node ID, review the serving model's
  classification and score, and inspect incoming/outgoing connections and known
  neighbor-label availability. Alternatively, supply a complete JSON feature
  array or object. Feature-only analysis has no graph report.
- **Reports:** generate a report explicitly after a node analysis; cancel, retry,
  regenerate, or download a completed report as Markdown. Partial streams and
  unavailable services have explicit states. Report content is supplied by the
  existing investigation backend; this redesign does not change its reasoning.
- **Batch screening:** upload a CSV with all 166 named feature columns and an
  optional `reference` column. The client validates finite values, quoted CSV,
  required fields, a 100-row limit and a 2 MB limit. Requests use `/predict` one
  row at a time, with cancellation, progress, per-row errors and CSV export.
- **Model information:** displays the currently serving model and actual service
  availability from `/health`. Dataset counts also come from the loaded graph.
- **Recent analyses:** up to eight successful analyses in the current page
  session. They are held in memory; no case database or persistent history is
  implied. Clicking a node reference performs a fresh analysis.

Classification is taken directly from the server. The dashboard does not apply
another classification threshold, display hardcoded model-performance claims,
or treat unknown neighbor labels as legitimate. The serving model remains
GraphSAGE; this interface does not activate the Random Forest research candidate.

## Implementation

`api/static/index.html` defines accessible views, forms and controls.
`dashboard.css` provides the neutral/forest colour system, desktop sidebar,
compact-screen navigation and responsive report/evidence layout.
`dashboard.mjs` handles the live workflow. `dashboard-core.mjs` contains the pure
input, CSV, prediction-display, export and streamed-event parsing functions.

Report Markdown is rendered with text nodes and a small set of formatting
elements. Generated HTML and URLs are never inserted into the page. File names,
references, errors and graph flags also use text nodes. CSV exports escape quotes
and neutralize spreadsheet-formula prefixes. No remote fonts or runtime CDNs are
required. Existing report-accuracy limitations remain separate from rendering.

Run the frontend checks with Node.js 24:

```bash
node --check api/static/dashboard.mjs
node --test tests/dashboard.test.mjs
```

The CI workflow also runs these checks. Backend tests remain available with
`.venv/bin/python -m pytest tests/ -q`. Browser QA covers real node/feature
predictions, reports and cancellation, CSV validation/screening/export, service
information, and desktop/mobile layouts. The redesign is a local source change;
it does not deploy to the historical AWS host.
