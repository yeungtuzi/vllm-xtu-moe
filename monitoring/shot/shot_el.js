const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ channel: 'chromium', args: ['--no-sandbox'] });
  const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  await p.goto(process.argv[2], { waitUntil: 'domcontentloaded', timeout: 90000 });
  await p.waitForTimeout(parseInt(process.argv[4] || '25000', 10));
  const h = await p.evaluate(() => document.body.scrollHeight);
  await p.setViewportSize({ width: 1600, height: Math.min(h, 4000) });
  await p.waitForTimeout(4000);
  await p.screenshot({ path: process.argv[3], fullPage: true });
  console.log('page height', h);
  await b.close();
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });
