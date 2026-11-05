import test from 'node:test';
import assert from 'node:assert/strict';
import { FEATURE_NAMES, parseFeatureVector, parseCSV, predictionView, createEventParser, csvCell } from '../api/static/dashboard-core.mjs';

const header = FEATURE_NAMES.join(',');
const values = FEATURE_NAMES.map(() => '0').join(',');

test('feature vectors require all finite numeric values', () => {
  const vector = parseFeatureVector(JSON.stringify(Array(166).fill(0)));
  assert.equal(Object.keys(vector).length, 166);
  assert.equal(vector.feature_166, 0);
  assert.throws(() => parseFeatureVector('[1,2]'), /exactly 166/);
  assert.throws(() => parseFeatureVector(JSON.stringify(Array(166).fill(null))), /finite number/);
  assert.throws(() => parseFeatureVector('{"feature_1":0}'), /feature_2/);
  assert.throws(() => parseFeatureVector(JSON.stringify({ ...vector, other: 0 })), /Unexpected field/);
});

test('CSV handles BOM, CRLF, quoted headers and references', () => {
  const rows = parseCSV(`\uFEFFreference,${FEATURE_NAMES.map(name => `"${name}"`).join(',')}\r\n"case, \"\"one\"\"",${values}\r\n`);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].reference, 'case, "one"');
  assert.equal(rows[0].features.feature_166, 0);
});

test('CSV validates missing columns, missing values, invalid numbers and size', () => {
  assert.throws(() => parseCSV('feature_1\n0'), /Missing 165/);
  assert.throws(() => parseCSV(`${header}\n${values},1`), /167 fields/);
  assert.throws(() => parseCSV(`${header}\n${values.replace(/^0/, '')}`), /feature_1 is empty/);
  assert.throws(() => parseCSV(`${header}\n${values.replace(/^0/, 'NaN')}`), /finite number/);
  assert.throws(() => parseCSV(`${header}\n${values.replace(/^0/, 'Infinity')}`), /finite number/);
  assert.throws(() => parseCSV(`${header}\n${Array(101).fill(values).join('\n')}`), /up to 100/);
  assert.throws(() => parseCSV(`${header}\n"${values}`), /not closed/);
});

test('CSV handles shuffled columns and preserves negative values', () => {
  const names = [...FEATURE_NAMES].reverse();
  const [row] = parseCSV(`${names.join(',')}\n${names.map((name, i) => i === 0 ? '-1.25' : '0').join(',')}`);
  assert.equal(row.features.feature_166, -1.25);
  assert.equal(row.features.feature_1, 0);
});

test('display uses the server label, including middle-range predictions', () => {
  const result = predictionView({ fraud_probability: .6, label: 'illicit' });
  assert.equal(result.label, 'Illicit');
  assert.equal(result.percent, '60.0');
  assert.equal(predictionView({ fraud_probability: .4, label: 'uncertain' }).label, 'Uncertain');
  assert.throws(() => predictionView({ fraud_probability: -1, label: 'licit' }), /invalid model score/);
  assert.throws(() => predictionView({ fraud_probability: .5, label: 'error' }), /unrecognized/);
});

test('SSE parser handles chunk boundaries and CRLF', () => {
  const parse = createEventParser();
  assert.deepEqual(parse('data: {"type":"sta'), []);
  assert.deepEqual(parse('tus","message":"Starting"}\r'), []);
  assert.deepEqual(parse('\n\r\ndata: {"type":"chunk","text":"Hello"}\n\n'), [{ type: 'status', message: 'Starting' }, { type: 'chunk', text: 'Hello' }]);
  assert.deepEqual(parse('data: {"type":"done"}', true), [{ type: 'done' }]);
});

test('SSE ignores keep-alive comments and rejects malformed events', () => {
  const parse = createEventParser();
  assert.deepEqual(parse(': keep-alive\n\n'), []);
  assert.throws(() => parse('data: not-json\n\n'), /unreadable response/);
});

test('CSV export quotes fields and neutralizes spreadsheet formulas', () => {
  assert.equal(csvCell('case,"one"'), '"case,""one"""');
  assert.equal(csvCell('=1+2'), '"\'=1+2"');
  assert.equal(csvCell('  @SUM(A1)'), '"\'  @SUM(A1)"');
  assert.equal(csvCell(.5), '"0.5"');
});
