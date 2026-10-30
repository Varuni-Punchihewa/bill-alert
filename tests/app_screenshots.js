// End-to-end UI check + screenshots for the report appendix.
// Start the app first:  streamlit run app/streamlit_app.py --server.port 8599
// Then:                  node tests/app_screenshots.js
const { chromium } = require('playwright');
const path = require('path');
const out = path.join(__dirname, '..', 'reports', 'figures');

async function pick(p, label) {
  await p.locator('[data-testid="stSelectbox"]').first().click();
  await p.keyboard.type(label); await p.keyboard.press('Enter'); await p.waitForTimeout(1500);
  await p.getByRole('button', { name: 'Check my month' }).click(); await p.waitForTimeout(6000);
}

(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1400, height: 1320 }, deviceScaleFactor: 1.5 });
  const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  await p.goto('http://localhost:8599', { waitUntil: 'networkidle' }); await p.waitForTimeout(3000);
  await pick(p, 'Example: family of four');
  await p.screenshot({ path: path.join(out, 'app_result_over.png') });
  await p.getByText('See the rules that matched your home').click(); await p.waitForTimeout(1200);
  const exp = p.locator('[data-testid="stExpander"]').filter({ hasText: 'See the rules that matched' });
  await exp.screenshot({ path: path.join(out, 'app_rules.png') });
  const nRules = await exp.locator('.reason').count();
  if (nRules < 1) errors.push('no matched rules shown');
  await pick(p, 'Example: careful two');
  await p.screenshot({ path: path.join(out, 'app_result_under.png') });
  await p.getByRole('tab', { name: 'How reliable is this?' }).click(); await p.waitForTimeout(1500);
  await p.screenshot({ path: path.join(out, 'app_reliability.png'), clip: { x: 0, y: 300, width: 1400, height: 760 } });
  await p.getByRole('tab', { name: 'About this tool' }).click(); await p.waitForTimeout(800);
  await p.getByText('See the whole rulebook').click(); await p.waitForTimeout(1500);
  if (await p.locator('[data-testid="stDataFrame"]').count() < 1) errors.push('rulebook table missing');
  const exc = await p.locator('[data-testid="stException"]').count();
  console.log(JSON.stringify({ pageErrors: errors, streamlitExceptions: exc }));
  await b.close();
  if (errors.length || exc) process.exit(1);
})();
