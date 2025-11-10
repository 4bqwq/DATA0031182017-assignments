const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');
const { createObjectCsvWriter } = require('csv-writer');

const EDGE_PATH   = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PROFILE_DIR = path.resolve('./edge-profile');
const CSV_DIR     = path.resolve('./csv');
const START_URL   = 'https://esi.clarivate.com/IndicatorsAction.action?Init=Yes';
const API_PATH    = '/IndicatorsDataAction.action';

const LIMIT    = 100;   // server page size cap
const SLEEP_MS = 700;   // gentle pacing

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
function ensureDir(p) { if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true }); }
function safeName(s) {
  return s.replace(/&/g, ' and ').replace(/\//g, ' or ').replace(/,/g, ' _ ')
          .replace(/[^A-Za-z0-9_\-\.]+/g, '_').replace(/^_+|_+$/g, '');
}
function makeCsvWriter(filename) {
  ensureDir(path.dirname(filename));
  return createObjectCsvWriter({
    path: filename,
    header: [
      { id: 'row', title: 'row' },
      { id: 'name', title: 'name' },
      { id: 'cites', title: 'cites' },
      { id: 'wosDocs', title: 'wos_docs' },
      { id: 'citesPerPaper', title: 'cites_per_paper' },
      { id: 'groupBy', title: 'groupBy' },
      { id: 'filterBy', title: 'filterBy' },
      { id: 'filterValues', title: 'filterValues' },
      { id: 'subject', title: 'subject' },
      { id: 'institution', title: 'institution' }
    ]
  });
}

async function launchBrowser() {
  ensureDir(PROFILE_DIR);
  ensureDir(CSV_DIR);
  const browser = await puppeteer.launch({
    headless: false,
    executablePath: EDGE_PATH,
    userDataDir: PROFILE_DIR,
    args: ['--lang=en-US']
  });
  const [page] = await browser.pages();
  return { browser, page };
}

async function ensureReady(page) {
  await page.goto(START_URL, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.waitForFunction(() => document.readyState === 'complete', { timeout: 60_000 });
  await page.waitForFunction(
    () => !document.querySelector('.x-mask') && !document.querySelector('.x-mask-msg'),
    { timeout: 60_000 }
  ).catch(() => {});
  await page.waitForNetworkIdle({ idleTime: 800, timeout: 20_000 }).catch(() => {});
}

async function attachXHRSpy(page) {
  page.on('request', (req) => {
    const url = req.url();
    if (url.includes('IndicatorsDataAction.action')) {
      const u = new URL(url);
      const p = Object.fromEntries(u.searchParams.entries());
      console.log('[ESI-XHR]', JSON.stringify(p));
    }
  });
}

async function esiQuery(page, {
  groupBy,
  filterBy,
  filterValues,
  docType = 'Top',
  pageNo = 1,
  start = 0,
  limit = LIMIT,
  sortProp = 'cites',
  sortDir = 'DESC'
}) {
  const MAX_TRIES = 5;

  for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
    try {
      const resText = await page.evaluate(async (params, apiPath) => {
        const url = new URL(apiPath, location.origin);
        url.searchParams.set('_dc', String(Date.now()));
        url.searchParams.set('type', 'grid');
        url.searchParams.set('groupBy', params.groupBy);
        url.searchParams.set('filterBy', params.filterBy);
        url.searchParams.set('filterValues', params.filterValues);
        url.searchParams.set('docType', params.docType);
        url.searchParams.set('page', String(params.pageNo));
        url.searchParams.set('start', String(params.start));
        url.searchParams.set('limit', String(params.limit));
        url.searchParams.set('sort', JSON.stringify([{ property: params.sortProp, direction: params.sortDir }]));

        const r = await fetch(url.toString(), {
          headers: { 'x-requested-with': 'XMLHttpRequest' },
          credentials: 'include',
          cache: 'no-store',
          redirect: 'follow'
        });
        return await r.text();
      }, { groupBy, filterBy, filterValues, docType, pageNo, start, limit, sortProp, sortDir }, API_PATH);

      // Try JSON parse
      try {
        return JSON.parse(resText);
      } catch {
        if (/No results/i.test(resText)) return { data: [], total: 0, raw: resText };
        if (/<!DOCTYPE|<html/i.test(resText)) throw new Error('Got HTML instead of JSON');
        throw new Error(`Unexpected response: ${resText.slice(0, 160)}`);
      }
    } catch (e) {
      const backoff = Math.min(2500, 400 * Math.pow(1.6, attempt - 1));
      console.warn(`  [retry ${attempt}/${MAX_TRIES}] ${e.message || e}. Waiting ${backoff}ms`);
      await page.waitForNetworkIdle({ idleTime: 600, timeout: 10_000 }).catch(() => {});
      await sleep(backoff);
      if (attempt === 3) await ensureReady(page).catch(() => {});
    }
  }
  throw new Error('ESI API fetch failed after retries');
}

async function fetchAll(page, q) {
  // stream through pages until a short page arrives
  const rows = [];
  let pageNo = 1, start = 0;
  for (;;) {
    console.log('[ESI-XHR]', JSON.stringify({
      _dc: String(Date.now()), type: 'grid',
      groupBy: q.groupBy, filterBy: q.filterBy, filterValues: q.filterValues,
      docType: q.docType || 'Top', page: String(pageNo), start: String(start),
      limit: String(LIMIT), sort: JSON.stringify([{ property: q.sortProp || 'cites', direction: q.sortDir || 'DESC' }])
    }));
    const chunk = await esiQuery(page, { ...q, pageNo, start, limit: LIMIT });
    const data = Array.isArray(chunk?.data) ? chunk.data : [];
    rows.push(...data);
    if (data.length < LIMIT) break; // last page
    pageNo += 1;
    start += LIMIT;
    await sleep(SLEEP_MS);
  }
  return rows;
}

// normalize to skinny rows
function normalizeRows(rows, opts) {
  return rows.map((r, i) => ({
    row: i + 1,
    name: r.institution || r.researchField || r.journal || r.territory || r.name || '',
    cites: r.cites ?? null,
    wosDocs: r.webOfScienceDocs ?? r.webOfScienceDocuments ?? null,
    citesPerPaper: r.citesPerPaper ?? null,
    groupBy: opts.groupBy,
    filterBy: opts.filterBy,
    filterValues: opts.filterValues,
    subject: opts.subject || '',
    institution: opts.institution || ''
  }));
}

// perspectives
async function institutionsByField(page, fieldName, docType = 'Top') {
  const filterValues = fieldName.toUpperCase();
  const q = { groupBy: 'Institutions', filterBy: 'ResearchFields', filterValues, docType };
  const rows = await fetchAll(page, q);
  return normalizeRows(rows, { ...q, subject: fieldName });
}

async function fieldsByInstitution(page, institutionName) {
  const filterValues = institutionName.toUpperCase(); // ESI is case-sensitive for institutions
  const q = { groupBy: 'ResearchFields', filterBy: 'Institutions', filterValues, docType: 'Top' };
  const rows = await fetchAll(page, q);
  return normalizeRows(rows, { ...q, institution: institutionName });
}

async function journalsByField(page, fieldName) {
  const q = { groupBy: 'Journals', filterBy: 'ResearchFields', filterValues: fieldName.toUpperCase(), docType: 'Top' };
  const rows = await fetchAll(page, q);
  return normalizeRows(rows, { ...q, subject: fieldName });
}

async function territoriesByField(page, fieldName) {
  const q = { groupBy: 'Territories', filterBy: 'ResearchFields', filterValues: fieldName.toUpperCase(), docType: 'Top' };
  const rows = await fetchAll(page, q);
  return normalizeRows(rows, { ...q, subject: fieldName });
}

async function main() {
  const wantLogin = process.argv.includes('--login');
  const { browser, page } = await launchBrowser();
  await attachXHRSpy(page);

  if (wantLogin) {
    await ensureReady(page);
    console.log('\nLogin mode: complete SSO in Edge, then press Enter here.');
    await new Promise((resolve) => {
      process.stdin.setRawMode(true);
      process.stdin.resume();
      process.stdin.on('data', () => resolve());
    });
    console.log('Thanks — profile saved to', PROFILE_DIR);
    await browser.close();
    return;
  }

  await ensureReady(page);

  const subjects = [
    'Agricultural Sciences','Biology & Biochemistry','Chemistry','Clinical Medicine','Computer Science',
    'Economics & Business','Engineering','Environment/Ecology','Geosciences','Immunology',
    'Materials Science','Mathematics','Microbiology','Molecular Biology & Genetics','Multidisciplinary',
    'Neuroscience & Behavior','Pharmacology & Toxicology','Physics','Plant & Animal Science',
    'Psychiatry/Psychology','Social Sciences, General','Space Science'
  ];
  const institution = 'East China Normal University';

  // A) Institutions by Field (Top)
  for (const s of subjects) {
    console.log(`\n[Institutions by Field] ${s}`);
    const rows = await institutionsByField(page, s, 'Top');
    const out = path.join(CSV_DIR, `esi_institutions_by_${safeName(s)}.csv`);
    await makeCsvWriter(out).writeRecords(rows);
    console.log(`  → ${rows.length} rows written to ${out}`);
  }

  // B) Fields for the institution (Top)
  console.log(`\n[Fields by Institution] ${institution}`);
  const rowsB = await fieldsByInstitution(page, institution);
  const outB = path.join(CSV_DIR, `esi_fields_of_${safeName(institution)}.csv`);
  await makeCsvWriter(outB).writeRecords(rowsB);
  console.log(`  → ${rowsB.length} rows written to ${outB}`);

  // C/D) Optional examples
  try {
    console.log(`\n[Journals by Field] Materials Science`);
    const rowsC = await journalsByField(page, 'Materials Science');
    const outC = path.join(CSV_DIR, `esi_journals_by_Materials_Science.csv`);
    await makeCsvWriter(outC).writeRecords(rowsC);
    console.log(`  → ${rowsC.length} rows written to ${outC}`);
  } catch (e) { console.log('  (Journals view may not be enabled; skipping)'); }

  try {
    console.log(`\n[Territories by Field] Chemistry`);
    const rowsD = await territoriesByField(page, 'Chemistry');
    const outD = path.join(CSV_DIR, `esi_territories_by_Chemistry.csv`);
    await makeCsvWriter(outD).writeRecords(rowsD);
    console.log(`  → ${rowsD.length} rows written to ${outD}`);
  } catch (e) { console.log('  (Territories/Countries view name may differ; skipping)'); }

  await browser.close();
}

main().catch(e => { console.error(e); process.exit(1); });
