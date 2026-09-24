const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ channel: 'chromium', args: ['--no-sandbox'] });
  const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  await p.goto(process.argv[2], { waitUntil: 'domcontentloaded', timeout: 90000 });
  await p.waitForTimeout(20000);
  const r = await p.evaluate(() => {
    const out = { iframes: document.querySelectorAll('iframe').length, imgs: document.querySelectorAll('img').length,
                  panels: document.querySelectorAll('[data-testid^="data-testid Panel"]').length,
                  hits: [] };
    for (const el of document.querySelectorAll('*')) {
      const t = (el.textContent || '').slice(0, 0);
      if (el.children.length === 0 && /核心地图/.test(el.textContent || '')) out.hits.push(el.textContent.slice(0, 60));
      if (el.tagName === 'IFRAME') out.hits.push('IFRAME src=' + el.getAttribute('src'));
    }
    // 找 text 面板的容器 HTML
    for (const el of document.querySelectorAll('[data-testid="data-testid Panel content"]')) {
      const h = el.innerHTML;
      if (/coremap|iframe/i.test(h)) out.hits.push('PANEL_HTML:' + h.slice(0, 200));
    }
    return out;
  });
  console.log(JSON.stringify(r, null, 1));
  await b.close();
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });
