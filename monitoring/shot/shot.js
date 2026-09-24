const { chromium } = require('playwright');
(async () => {
  const url = process.argv[2];
  const out = process.argv[3] || '/tmp/dash.png';
  const wait = parseInt(process.argv[4] || '25000', 10);
  const b = await chromium.launch({ channel: 'chromium', args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1200 }, deviceScaleFactor: 1 });
  const errs = [];
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 160)); });
  await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await p.waitForTimeout(wait);
  // 统计实际渲染出的面板数与"重复方块"数
  const info = await p.evaluate(() => {
    const panels = document.querySelectorAll('[data-panelid]');
    const titles = [...panels].map(e => (e.getAttribute('aria-label') || '').slice(0, 60));
    return { n: panels.length, cpuish: titles.filter(t => /核心|core/i.test(t)).length, titles: titles.slice(0, 8) };
  });
  await p.screenshot({ path: out, fullPage: true });
  console.log(JSON.stringify({ panels: info.n, cpuish: info.cpuish, sample: info.titles, errs: errs.slice(0, 5) }, null, 1));
  await b.close();
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });
