export const FEATURE_COUNT = 166;
export const MAX_BATCH_ROWS = 100;
export const FEATURE_NAMES = Array.from({ length: FEATURE_COUNT }, (_, i) => `feature_${i + 1}`);

function finiteNumber(value, reference) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${reference} must be a finite number.`);
  return value;
}

export function parseFeatureVector(text) {
  let data;
  try { data = JSON.parse(text); } catch { throw new Error('Enter a valid JSON array or feature object.'); }
  if (Array.isArray(data)) {
    if (data.length !== FEATURE_COUNT) throw new Error(`Provide exactly ${FEATURE_COUNT} feature values; received ${data.length}.`);
    return Object.fromEntries(FEATURE_NAMES.map((name, i) => [name, finiteNumber(data[i], name)]));
  }
  if (!data || typeof data !== 'object') throw new Error('Provide a JSON array or an object with all 166 named features.');
  const unexpected = Object.keys(data).filter(key => !FEATURE_NAMES.includes(key));
  if (unexpected.length) throw new Error(`Unexpected field: ${unexpected[0]}. Use feature_1 through feature_166.`);
  return Object.fromEntries(FEATURE_NAMES.map(name => [name, finiteNumber(data[name], name)]));
}

export function parseCSV(text) {
  text = text.replace(/^\uFEFF/, '');
  const rows = []; let row = [], field = '', quoted = false, closed = false;
  function addField() { row.push(field.trim()); field = ''; closed = false; }
  function addRow() { addField(); if (row.some(value => value !== '')) rows.push(row); row = []; }
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') { quoted = false; closed = true; }
      else field += char;
    } else if (char === '"' && !field.trim() && !closed) { field = ''; quoted = true; }
    else if (char === ',') addField();
    else if (char === '\n' || char === '\r') { if (char === '\r' && text[i + 1] === '\n') i++; addRow(); }
    else if (closed && char.trim()) throw new Error('A quoted CSV field contains unexpected characters.');
    else if (char === '"') throw new Error('Check the quotation marks in the CSV file.');
    else if (!closed) field += char;
  }
  if (quoted) throw new Error('A quoted CSV field was not closed.');
  if (field || row.length || closed) addRow();
  if (rows.length < 2) throw new Error('Include a header and at least one transaction.');
  const headers = rows.shift();
  if (new Set(headers).size !== headers.length) throw new Error('CSV column names must be unique.');
  const missing = FEATURE_NAMES.filter(name => !headers.includes(name));
  if (missing.length) throw new Error(`Missing ${missing.length} feature column${missing.length === 1 ? '' : 's'}, starting with ${missing[0]}. Download the template for the required format.`);
  if (rows.length > MAX_BATCH_ROWS) throw new Error(`Screen up to ${MAX_BATCH_ROWS} transactions at a time. This file contains ${rows.length}.`);
  const indices = FEATURE_NAMES.map(name => headers.indexOf(name));
  const referenceIndex = headers.indexOf('reference');
  return rows.map((values, rowIndex) => {
    if (values.length !== headers.length) throw new Error(`Row ${rowIndex + 2} has ${values.length} fields; expected ${headers.length}.`);
    const features = Object.fromEntries(FEATURE_NAMES.map((name, i) => {
      const value = values[indices[i]];
      if (!value) throw new Error(`Row ${rowIndex + 2}: ${name} is empty.`);
      return [name, finiteNumber(Number(value), `Row ${rowIndex + 2}: ${name}`)];
    }));
    return { reference: referenceIndex >= 0 && values[referenceIndex] ? values[referenceIndex] : `Row ${rowIndex + 1}`, features };
  });
}

export function predictionView(data) {
  const probability = data?.fraud_probability;
  if (typeof probability !== 'number' || !Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error('The service returned an invalid model score.');
  const labels = { licit: 'Licit', illicit: 'Illicit', uncertain: 'Uncertain' };
  if (!Object.hasOwn(labels, data.label)) throw new Error('The service returned an unrecognized classification.');
  // Classification comes from the serving model; do not apply a second threshold in the UI.
  return { label: labels[data.label], className: data.label, probability, percent: (probability * 100).toFixed(1) };
}

export function createEventParser() {
  let buffer = '';
  return (chunk, final = false) => {
    buffer = (buffer + chunk).replace(/\r\n/g, '\n');
    if (buffer.length > 2_000_000) throw new Error('The report stream exceeded the supported response size.');
    const frames = buffer.split('\n\n'); buffer = frames.pop();
    if (final && buffer.trim()) { frames.push(buffer); buffer = ''; }
    return frames.flatMap(frame => {
      const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
      if (!data) return [];
      let event;
      try { event = JSON.parse(data); } catch { throw new Error('The report stream returned an unreadable response. Please try again.'); }
      if (!event || typeof event.type !== 'string') throw new Error('The report stream returned an invalid event.');
      return [event];
    });
  };
}

export function csvCell(value) {
  let text = String(value ?? '');
  if (/^[\s]*[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}
