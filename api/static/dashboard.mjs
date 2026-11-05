import { FEATURE_NAMES, parseFeatureVector, parseCSV, predictionView, createEventParser, csvCell } from './dashboard-core.mjs';

const $ = id => document.getElementById(id);
const number = value => new Intl.NumberFormat('en-US').format(value);
const time = value => new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(value);
const state = { health: null, mode: 'node', node: null, recent: [], report: '', fileRows: null, fileId: 0, batchResults: [], analysisId: 0, analysisController: null, reportController: null, batchController: null };
const nodeButtonHTML = $('analyze-node').innerHTML;
const featureButtonHTML = $('analyze-features').innerHTML;

function show(id, visible = true) { $(id).hidden = !visible; }
function badge(id, text, className = 'neutral') { $(id).textContent = text; $(id).className = `badge ${className}`; }
function error(id, message = '') { $(id).textContent = message; show(id, Boolean(message)); }
function reportNotice(message) { $('report-notice-text').textContent = message; show('report-notice', Boolean(message)); }
function cell(row, text, className = '') { const td = row.insertCell(); td.textContent = text; if (className) td.className = className; return td; }
function labelCell(row, view) { const td = row.insertCell(); const span = document.createElement('span'); span.className = `badge ${view.className}`; span.textContent = view.label; td.append(span); }

async function request(url, options = {}) {
  try { return await fetch(url, options); }
  catch (e) { if (e.name === 'AbortError' || options.signal?.aborted) throw e; throw new Error('The service could not be reached. Check the connection and try again.'); }
}
async function requestJSON(url, options = {}) {
  const response = await request(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error('The service returned an unreadable response. Please try again.'); }
  if (!response.ok) {
    if (response.status >= 500) throw new Error(response.status === 503 ? 'This service is currently unavailable. Please try again later.' : 'The service could not complete this request. Please try again.');
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Some input values were rejected. Check the supplied values.');
  }
  return data;
}

async function loadHealth() {
  $('refresh-health').disabled = true;
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const health = await requestJSON('/health', { signal: controller.signal });
    if (health.status !== 'ok' || !health.dataset || !Number.isSafeInteger(health.dataset.nodes) || health.dataset.nodes < 1) throw new Error('Unavailable');
    state.health = health;
    document.querySelectorAll('[data-model]').forEach(element => { element.textContent = health.model; });
    document.querySelectorAll('[data-stat]').forEach(element => { const value = health.dataset[element.dataset.stat]; element.textContent = Number.isFinite(value) ? number(value) : '—'; });
    $('node-help').textContent = `Enter an internal graph node ID from 0 to ${number(health.dataset.nodes - 1)}.`;
    $('sidebar-status').textContent = 'API connected'; $('sidebar-dot').className = 'status-dot green';
    badge('model-connection', 'Connected', 'licit'); badge('service-prediction', 'Available', 'licit');
    badge('service-investigation', health.investigation_available ? 'Available' : 'Unavailable', health.investigation_available ? 'licit' : 'neutral');
    badge('service-rag', health.rag_available ? 'Available' : 'Unavailable', health.rag_available ? 'licit' : 'neutral');
    badge('service-llm', health.llm_available ? 'Configured' : health.investigation_available ? 'Local summary' : 'Unavailable', health.llm_available ? 'licit' : 'neutral');
    $('health-checked').textContent = `Last checked at ${time(new Date())}.`;
    show('connection-alert', false);
  } catch {
    state.health = null;
    $('sidebar-status').textContent = 'Service unavailable'; $('sidebar-dot').className = 'status-dot offline';
    ['model-connection', 'service-prediction', 'service-investigation', 'service-rag', 'service-llm'].forEach(id => badge(id, 'Unavailable'));
    document.querySelectorAll('[data-stat], [data-model]').forEach(element => { element.textContent = '—'; });
    $('health-checked').textContent = `Could not connect at ${time(new Date())}.`;
    show('connection-alert');
  } finally { clearTimeout(timeout); $('refresh-health').disabled = false; updateReportAvailability(); }
}

function stopReport(message = 'Report generation cancelled. You can generate it again.') {
  if (!state.reportController) return;
  state.reportController.abort(); state.reportController = null;
  show('report-progress', false); reportNotice(message);
  if (state.report) renderReport();
  show('report-empty', !state.report); updateReportAvailability();
}
function resetReport() {
  stopReport(); state.report = ''; $('report-content').replaceChildren();
  show('report-empty'); show('report-content', false); show('report-progress', false); show('report-complete', false); show('report-notice', false);
  $('export-report').disabled = true;
  updateReportAvailability();
}
function updateReportAvailability() {
  const allowed = state.node !== null && Boolean(state.health?.investigation_available) && !state.reportController;
  $('generate-report').disabled = !allowed; $('regenerate-report').disabled = !allowed;
  $('retry-report').disabled = !allowed;
  $('report-availability').textContent = state.node === null ? 'Available after a node analysis' : !state.health ? 'Connect to the service to generate a report' : !state.health.investigation_available ? 'Investigation service is currently unavailable' : state.health.llm_available ? `Ready for node ${number(state.node)}` : `Structured local summary for node ${number(state.node)}`;
}
function analysisBusy(busy) {
  if (busy) $('analysis-empty').setAttribute('aria-busy', 'true'); else $('analysis-empty').removeAttribute('aria-busy');
  $('analyze-node').disabled = busy; $('analyze-features').disabled = busy;
  $('analyze-node').innerHTML = busy && state.mode === 'node' ? '<span class="spinner" aria-hidden="true"></span>Analyzing…' : nodeButtonHTML;
  $('analyze-features').innerHTML = busy && state.mode === 'features' ? '<span class="spinner" aria-hidden="true"></span>Analyzing…' : featureButtonHTML;
  document.querySelectorAll('[data-node]').forEach(button => { button.disabled = busy; });
}

async function analyze(kind) {
  let payload, node;
  const errorId = kind === 'node' ? 'node-error' : 'feature-error';
  const input = kind === 'node' ? $('node-id') : $('feature-input');
  error(errorId); input.removeAttribute('aria-invalid');
  try {
    if (kind === 'node') {
      const value = input.value.trim();
      if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value))) throw new Error('Enter a valid, whole-number graph node ID.');
      node = Number(value);
      if (state.health && node >= state.health.dataset.nodes) throw new Error(`Choose a node from 0 to ${number(state.health.dataset.nodes - 1)}.`);
    } else payload = parseFeatureVector(input.value);
  } catch (e) { error(errorId, e.message); input.setAttribute('aria-invalid', 'true'); input.focus(); return; }
  state.analysisController?.abort(); const controller = new AbortController(); state.analysisController = controller;
  const operation = ++state.analysisId; const timeout = setTimeout(() => controller.abort('timeout'), 30000);
  state.node = null; resetReport(); show('analysis-result', false); show('analysis-empty'); badge('analysis-state', 'Analyzing'); analysisBusy(true);
  $('analysis-empty').setAttribute('aria-busy', 'true');
  try {
    const prediction = requestJSON(kind === 'node' ? `/predict/${node}` : '/predict', kind === 'node' ? { signal: controller.signal } : { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const graph = kind === 'node' ? requestJSON(`/node/${node}`, { signal: controller.signal }) : Promise.resolve(null);
    const [p, g] = await Promise.allSettled([prediction, graph]);
    if (operation !== state.analysisId) return;
    if (p.status === 'rejected') throw p.reason;
    const view = predictionView(p.value);
    const date = new Date(); const reference = kind === 'node' ? `Node ${number(node)}` : 'Feature vector';
    $('model-score').textContent = view.percent; badge('class-badge', view.label, view.className);
    $('score-fill').style.width = `${view.probability * 100}%`;
    const scoreColor = view.className === 'illicit' ? 'var(--danger)' : view.className === 'uncertain' ? 'var(--amber)' : 'var(--accent)';
    $('score-fill').style.background = scoreColor; $('model-score').parentElement.style.color = scoreColor;
    $('score-track').setAttribute('aria-label', `Model score: ${view.percent} percent. Classification: ${view.label}.`);
    $('result-reference').textContent = reference; $('result-time').textContent = time(date);
    state.node = kind === 'node' ? node : null; updateReportAvailability();
    show('network-details', false); show('network-message', false);
    if (kind === 'node' && g.status === 'fulfilled' && g.value) renderGraph(g.value);
    else { $('network-message').textContent = kind === 'node' ? 'The prediction is available, but network context could not be retrieved.' : 'Feature-only prediction. Network context and graph investigation reports require a graph node.'; show('network-message'); }
    show('analysis-empty', false); show('analysis-result'); badge('analysis-state', 'Complete', 'licit');
    state.recent.unshift({ reference, kind, node: kind === 'node' ? node : null, view, date }); state.recent = state.recent.slice(0, 8); renderRecent();
    if (window.matchMedia('(max-width: 960px)').matches) document.querySelector('.analysis-panel').scrollIntoView({ block: 'nearest', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
  } catch (e) {
    if (operation !== state.analysisId) return;
    error(errorId, controller.signal.reason === 'timeout' ? 'The request timed out. Please try again.' : e.name === 'AbortError' ? 'Analysis cancelled. Please try again.' : e.message);
    badge('analysis-state', 'Unavailable'); show('analysis-empty'); show('analysis-result', false);
  } finally {
    clearTimeout(timeout);
    if (operation === state.analysisId) { state.analysisController = null; analysisBusy(false); $('analysis-empty').removeAttribute('aria-busy'); }
  }
}

function renderGraph(graph) {
  $('incoming').textContent = number(graph.in_degree); $('outgoing').textContent = number(graph.out_degree); $('neighbors').textContent = number(graph.neighbors_1hop.total);
  const neighbors = graph.neighbors_1hop;
  $('neighbor-labels').textContent = neighbors.total ? `${number(neighbors.illicit + neighbors.licit)} of ${number(neighbors.total)} neighbors have an available label; ${number(neighbors.unknown)} are unknown.` : 'No directly connected neighbors were found.';
  $('structural-flags').replaceChildren();
  for (const flag of graph.structural_flags || []) { const span = document.createElement('span'); span.textContent = String(flag).replaceAll('_', ' '); $('structural-flags').append(span); }
  show('network-details');
}
function renderRecent() {
  $('recent-rows').replaceChildren();
  for (const item of state.recent) {
    const row = $('recent-rows').insertRow(); const reference = row.insertCell();
    if (item.node !== null) { const button = document.createElement('button'); button.type = 'button'; button.className = 'table-reference'; button.textContent = item.reference; button.addEventListener('click', () => { setMode('node'); $('node-id').value = String(item.node); analyze('node'); }); reference.append(button); }
    else reference.textContent = item.reference;
    cell(row, item.kind === 'node' ? 'Graph node' : 'Features'); labelCell(row, item.view); cell(row, `${item.view.percent}%`, 'numeric'); cell(row, time(item.date), 'numeric');
  }
}

// Markdown is constructed from text nodes. Generated HTML, scripts and URLs are never inserted.
function inline(parent, text) {
  const tokens = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g; let position = 0;
  for (const match of text.matchAll(tokens)) {
    parent.append(document.createTextNode(text.slice(position, match.index)));
    const value = match[0]; const element = document.createElement(value.startsWith('**') ? 'strong' : value.startsWith('`') ? 'code' : 'em');
    element.textContent = value.slice(value.startsWith('**') ? 2 : 1, value.startsWith('**') ? -2 : -1); parent.append(element); position = match.index + value.length;
  }
  parent.append(document.createTextNode(text.slice(position)));
}
function renderReport() {
  const fragment = document.createDocumentFragment(); let list = null, table = null;
  const lines = state.report.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) { list = null; table = null; continue; }
    if (/^\|.*\|$/.test(line)) {
      if (/^[|:\-\s]+$/.test(line)) continue;
      if (!table) { const wrap = document.createElement('div'); wrap.className = 'table-scroll'; table = document.createElement('table'); wrap.append(table); fragment.append(wrap); }
      const row = table.insertRow(); for (const text of line.slice(1, -1).split('|')) inline(row.insertCell(), text.trim()); continue;
    }
    table = null;
    const heading = line.match(/^(#{1,6})\s+(.+)/);
    const bullet = line.match(/^([-*]|\d+\.)\s+(.+)/);
    if (heading) { list = null; const h = document.createElement(`h${Math.min(4, Math.max(2, heading[1].length))}`); inline(h, heading[2]); fragment.append(h); }
    else if (bullet) { const type = /\d/.test(bullet[1]) ? 'ol' : 'ul'; if (!list || list.tagName.toLowerCase() !== type) { list = document.createElement(type); fragment.append(list); } const li = document.createElement('li'); inline(li, bullet[2]); list.append(li); }
    else if (/^[-*_]{3,}$/.test(line)) { list = null; fragment.append(document.createElement('hr')); }
    else { list = null; const p = document.createElement(line.startsWith('> ') ? 'blockquote' : 'p'); inline(p, line.startsWith('> ') ? line.slice(2) : line); fragment.append(p); }
  }
  $('report-content').replaceChildren(fragment);
}

async function generateReport() {
  if (state.node === null || !state.health?.investigation_available || state.reportController) return;
  const node = state.node; resetReport();
  const controller = new AbortController(); state.reportController = controller; updateReportAvailability();
  show('report-empty', false); show('report-progress'); $('report-status').textContent = `Preparing investigation for node ${number(node)}…`;
  const timeout = setTimeout(() => controller.abort('timeout'), 180000);
  let finished = false, renderQueued = false;
  const update = () => { if (renderQueued) return; renderQueued = true; requestAnimationFrame(() => { renderQueued = false; if (state.reportController === controller) renderReport(); }); };
  const consume = event => {
    if (state.reportController !== controller) return;
    if (event.type === 'status' && typeof event.message === 'string') $('report-status').textContent = event.message;
    else if (event.type === 'chunk' && typeof event.text === 'string') { state.report += event.text; show('report-content'); update(); }
    else if (event.type === 'done') finished = true;
    else if (event.type === 'error') throw new Error(typeof event.message === 'string' ? event.message : 'Report generation could not be completed.');
  };
  try {
    const response = await request(`/investigate/stream/${node}`, { signal: controller.signal, headers: { Accept: 'text/event-stream' } });
    if (!response.ok) throw new Error(response.status === 503 ? 'Investigation reports are currently unavailable. Your prediction is still available.' : 'The investigation could not be started. Please try again.');
    if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body) throw new Error('The service did not return a report stream.');
    const reader = response.body.getReader(), decoder = new TextDecoder(), parseEvents = createEventParser();
    try {
      while (true) { const { value, done } = await reader.read(); if (done) break; for (const event of parseEvents(decoder.decode(value, { stream: true }))) consume(event); }
      for (const event of parseEvents(decoder.decode(), true)) consume(event);
    } finally { reader.releaseLock(); }
    if (state.reportController !== controller) return;
    if (!finished || !state.report.trim()) throw new Error('The report ended before it was complete. You can retry; any partial text is shown below.');
    renderReport(); show('report-complete'); $('export-report').disabled = false;
  } catch (e) {
    if (state.reportController !== controller) return;
    reportNotice(controller.signal.reason === 'timeout' ? 'Report generation timed out. You can retry; any partial text is shown below.' : e.name === 'AbortError' ? 'Report generation cancelled.' : e.message);
    if (state.report) { renderReport(); show('report-content'); } else show('report-empty');
  } finally {
    clearTimeout(timeout);
    if (state.reportController === controller) { state.reportController = null; show('report-progress', false); updateReportAvailability(); }
  }
}

function download(name, text, type) { const url = URL.createObjectURL(new Blob([text], { type })); const a = document.createElement('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
async function chooseFile(file) {
  if (!file || state.batchController) return;
  const operation = ++state.fileId;
  state.fileRows = null; state.batchResults = []; $('run-batch').disabled = true; show('batch-results', false); error('batch-error');
  try {
    if (!/\.csv$/i.test(file.name)) throw new Error('Choose a CSV file.');
    if (file.size > 2 * 1024 * 1024) throw new Error('Choose a file smaller than 2 MB.');
    const rows = parseCSV(await file.text()); if (operation !== state.fileId) return; state.fileRows = rows;
    $('file-description').textContent = `${file.name} · ${number(rows.length)} transaction${rows.length === 1 ? '' : 's'} · ${(file.size / 1024).toFixed(1)} KB`;
    $('run-batch').disabled = false; show('clear-file');
  } catch (e) { if (operation !== state.fileId) return; $('csv-file').value = ''; $('file-description').textContent = 'Up to 100 transactions · Maximum 2 MB'; show('clear-file', false); error('batch-error', e.message); }
}
function renderBatch() {
  $('batch-rows').replaceChildren();
  const counts = { Screened: state.batchResults.length, Illicit: 0, Licit: 0, Uncertain: 0, Errors: 0 };
  for (const item of state.batchResults) {
    const row = $('batch-rows').insertRow(); cell(row, item.reference);
    if (item.error) { counts.Errors++; labelCell(row, { label: 'Error', className: 'error' }); cell(row, '—', 'numeric'); cell(row, item.error, 'error-cell'); }
    else { counts[item.view.label]++; labelCell(row, item.view); cell(row, `${item.view.percent}%`, 'numeric'); cell(row, 'Complete'); }
  }
  $('batch-summary').replaceChildren();
  for (const [label, count] of Object.entries(counts)) { const div = document.createElement('div'), strong = document.createElement('strong'), span = document.createElement('span'); strong.textContent = number(count); span.textContent = label; div.append(strong, span); $('batch-summary').append(div); }
  show('batch-results');
}
async function runBatch() {
  if (!state.fileRows || state.batchController) return;
  const rows = state.fileRows, controller = new AbortController(); state.batchController = controller; state.batchResults = [];
  $('run-batch').disabled = true; $('clear-file').disabled = true; $('csv-file').disabled = true; show('batch-progress'); error('batch-error');
  $('batch-progress-bar').max = rows.length; $('batch-progress-bar').value = 0; $('export-batch').disabled = true;
  let cancelled = false;
  try {
    for (const item of rows) {
      if (controller.signal.aborted) { cancelled = true; break; }
      $('batch-status').textContent = `Screening ${state.batchResults.length + 1} of ${rows.length} transactions…`;
      let timeout;
      try {
        timeout = setTimeout(() => controller.abort('timeout'), 30000);
        const data = await requestJSON('/predict', { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(item.features) });
        state.batchResults.push({ reference: item.reference, view: predictionView(data) });
      } catch (e) {
        if (controller.signal.aborted) { cancelled = true; break; }
        state.batchResults.push({ reference: item.reference, error: e.message });
      } finally { clearTimeout(timeout); }
      $('batch-progress-bar').value = state.batchResults.length; renderBatch();
    }
    $('batch-status').textContent = cancelled ? `Screening ${controller.signal.reason === 'timeout' ? 'timed out' : 'cancelled'}. ${state.batchResults.length} of ${rows.length} transactions completed.` : `${rows.length} transactions screened.`;
  } finally { state.batchController = null; $('run-batch').disabled = !state.fileRows; $('clear-file').disabled = false; $('csv-file').disabled = false; show('cancel-batch', false); $('export-batch').disabled = !state.batchResults.length; }
}

function setMode(mode) {
  if (state.mode !== mode) { state.analysisId++; state.analysisController?.abort(); state.analysisController = null; analysisBusy(false); badge('analysis-state', $('analysis-result').hidden ? 'Awaiting input' : 'Complete', $('analysis-result').hidden ? 'neutral' : 'licit'); }
  state.mode = mode;
  document.querySelectorAll('[data-mode]').forEach(tab => { const selected = tab.dataset.mode === mode; tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1; });
  show('input-node', mode === 'node'); show('input-features', mode === 'features');
}
function setView() {
  const requested = location.hash.slice(1), view = ['investigations', 'batch', 'model'].includes(requested) ? requested : 'investigations';
  const names = { investigations: 'Investigations', batch: 'Batch screening', model: 'Model information' };
  document.querySelectorAll('[data-view]').forEach(link => { const selected = link.dataset.view === view; link.classList.toggle('active', selected); if (selected) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  for (const name of Object.keys(names)) show(`view-${name}`, name === view);
  $('breadcrumb-current').textContent = names[view]; document.title = `FraudNet · ${names[view]}`;
  window.scrollTo({ top: 0, behavior: 'instant' });
}

$('node-form').addEventListener('submit', event => { event.preventDefault(); analyze('node'); });
$('feature-form').addEventListener('submit', event => { event.preventDefault(); analyze('features'); });
document.querySelectorAll('[data-node]').forEach(button => button.addEventListener('click', () => { $('node-id').value = button.dataset.node; analyze('node'); }));
document.querySelectorAll('[data-mode]').forEach(tab => {
  tab.addEventListener('click', () => setMode(tab.dataset.mode));
  tab.addEventListener('keydown', event => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) { event.preventDefault(); const mode = event.key === 'Home' ? 'node' : event.key === 'End' ? 'features' : state.mode === 'node' ? 'features' : 'node'; setMode(mode); $(`tab-${mode}`).focus(); } });
});
$('example-vector').addEventListener('click', () => { $('feature-input').value = JSON.stringify(Array(166).fill(0)); error('feature-error'); $('feature-input').removeAttribute('aria-invalid'); });
for (const [field, errorId] of [['node-id', 'node-error'], ['feature-input', 'feature-error']]) $(field).addEventListener('input', () => { error(errorId); $(field).removeAttribute('aria-invalid'); });
$('generate-report').addEventListener('click', generateReport); $('regenerate-report').addEventListener('click', generateReport);
$('retry-report').addEventListener('click', generateReport);
$('cancel-report').addEventListener('click', () => stopReport());
$('export-report').addEventListener('click', () => { if (state.report && state.node !== null) download(`fraudnet-node-${state.node}.md`, state.report, 'text/markdown;charset=utf-8'); });
$('csv-file').addEventListener('change', event => chooseFile(event.target.files[0]));
for (const type of ['dragenter', 'dragover']) $('drop-zone').addEventListener(type, event => { event.preventDefault(); if (!state.batchController) $('drop-zone').classList.add('dragging'); });
for (const type of ['dragleave', 'drop']) $('drop-zone').addEventListener(type, event => { event.preventDefault(); $('drop-zone').classList.remove('dragging'); if (type === 'drop') chooseFile(event.dataTransfer.files[0]); });
$('clear-file').addEventListener('click', () => { if (state.batchController) return; state.fileId++; state.fileRows = null; state.batchResults = []; $('csv-file').value = ''; $('file-description').textContent = 'Up to 100 transactions · Maximum 2 MB'; $('run-batch').disabled = true; show('clear-file', false); show('batch-results', false); show('batch-progress', false); error('batch-error'); });
$('run-batch').addEventListener('click', () => { show('cancel-batch'); runBatch(); });
$('cancel-batch').addEventListener('click', () => state.batchController?.abort('user'));
$('download-template').addEventListener('click', () => download('fraudnet-template.csv', `${FEATURE_NAMES.join(',')}\r\n${FEATURE_NAMES.map(() => '0').join(',')}\r\n`, 'text/csv;charset=utf-8'));
$('export-batch').addEventListener('click', () => download('fraudnet-screening.csv', ['reference,classification,model_score,error', ...state.batchResults.map(item => [item.reference, item.view?.label || 'Error', item.view?.probability ?? '', item.error || ''].map(csvCell).join(','))].join('\r\n'), 'text/csv;charset=utf-8'));
$('refresh-health').addEventListener('click', loadHealth); $('reconnect').addEventListener('click', loadHealth);
window.addEventListener('hashchange', () => { if (location.hash !== '#main') setView(); }); setView(); loadHealth();
