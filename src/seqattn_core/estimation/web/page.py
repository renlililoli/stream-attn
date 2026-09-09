"""Dependency-free local parameter editor; reports use the existing renderer."""

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 Block 仿真台 · SeqAttn</title>
<style>
:root{font:14px system-ui,-apple-system,"Segoe UI",sans-serif;color:#182f40;background:#f3f6f8;--teal:#127f78;--muted:#627b8b;--line:#dce5eb}
*{box-sizing:border-box}body{margin:0}header{height:86px;background:#fff;border-bottom:1px solid var(--line);padding:17px 25px;display:flex;align-items:center;justify-content:space-between;gap:20px}
.brand{font-size:11px;letter-spacing:2px;color:var(--teal);margin-bottom:4px}h1{font-size:22px;font-weight:650;margin:0}header p{color:var(--muted);margin:0;font-size:12px}.layout{display:grid;grid-template-columns:350px minmax(0,1fr);height:calc(100vh - 86px);height:calc(100dvh - 86px)}
aside{overflow-y:auto;padding:18px;border-right:1px solid var(--line);background:#fbfcfd}.actions{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px}button{font:inherit;cursor:pointer;border:1px solid #cfdce4;border-radius:7px;padding:7px 10px;background:white;color:#294758}button:hover{border-color:var(--teal)}button:disabled{opacity:.45;cursor:default}.primary{background:var(--teal);color:white;border-color:var(--teal)}
fieldset{border:0;padding:0;margin:0 0 18px}legend{font-size:12px;font-weight:650;letter-spacing:.5px;color:#476879;margin-bottom:12px;display:flex;align-items:center;gap:8px}legend .number{color:#91aaa9;font-size:11px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:11px}.field{display:flex;flex-direction:column;gap:5px;min-width:0}.field.wide{grid-column:1/-1}.field label{font-size:12px;color:#546d7e}input,select{width:100%;min-width:0;font:inherit;color:#173447;background:#fff;border:1px solid #d5e0e7;border-radius:6px;padding:8px;outline:none}input:focus,select:focus{border-color:var(--teal);box-shadow:0 0 0 2px #127f7814}input:disabled,select:disabled{background:#edf2f5;color:#7e929f}input:user-invalid{border-color:#c65f45}small{color:#8193a0;font-size:11px;line-height:1.5}.notice{font-size:12px;background:#eaf4f2;color:#36635f;padding:10px 12px;line-height:1.6;border-radius:7px;margin-bottom:14px}
details{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}summary{cursor:pointer;font-size:13px;color:#476879;margin-bottom:12px}.profile-label{font-size:12px;overflow-wrap:anywhere;margin:8px 0;color:var(--muted)}.muted{color:var(--muted)}
main{min-width:0;display:flex;flex-direction:column;position:relative}.result-bar{min-height:59px;padding:11px 20px;display:flex;align-items:center;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);background:#f9fbfc;flex-wrap:wrap}.state{display:flex;align-items:center;gap:8px;font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:#9dadb7}.dot.ready{background:#15877d}.dot.busy{background:#d09835;animation:pulse 1s infinite}.dot.error{background:#c45c42}.status-detail{font-size:11px;color:var(--muted);margin-top:4px}.result-actions{display:flex;align-items:center;gap:12px}.auto{font-size:12px;display:flex;gap:6px;align-items:center;white-space:nowrap}.auto input{width:auto;accent-color:var(--teal)}
#error{margin:0;padding:12px 20px;background:#fff1eb;color:#9e452e;line-height:1.6;white-space:pre-wrap;font-size:13px}#error:empty{display:none}#stale-note{display:none;padding:7px 20px;background:#fff7e7;color:#997037;font-size:12px}#report{width:100%;border:0;flex:1;min-height:400px;background:#f3f6f8}#report.stale{opacity:.55;pointer-events:none}.empty{position:absolute;left:0;right:0;top:180px;text-align:center;color:#8196a3;pointer-events:none}.empty strong{display:block;font-size:19px;font-weight:500;margin-bottom:9px}
@keyframes pulse{50%{opacity:.35}}@media(max-width:850px){header{height:auto;min-height:86px;padding:16px}header p{display:none}.layout{display:block;height:auto}aside{max-height:60vh;border-right:0;border-bottom:1px solid var(--line)}main{height:1000px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<header><div><div class="brand">SEQATTN / H3 SIMULATOR</div><h1>H3 Block 仿真台</h1></div><p>调整参数 → 重算完整 H3 路径 → 联动显示时间与内存</p></header>
<div class="layout"><aside>
<div class="actions"><button id="defaults">H3 默认参数</button><button id="import">导入参数</button><button id="save-settings">导出参数</button></div>
<div class="notice">初始吞吐与带宽是示例值，时间线是预测。默认打包并发路径已在 5090 观测，其他设备需验证。完整 H3 路径包含 RoPE 位置拷贝的 CPU 阻塞和 SwiGLU/FC2 联合调用。</div>
<form id="form" novalidate><div id="fields"></div></form>
<details id="profile-section"><summary>导入算子 Profile</summary>
<p class="muted" style="font-size:12px;line-height:1.6">使用现有的 H3DeviceProfile JSON。导入后由 Profile 决定各算子速率、工作区和打包并发关系，形状不匹配时会显示错误。</p>
<div class="actions"><button id="load-profile">选择 Profile JSON</button><button id="clear-profile" disabled>清除 Profile</button></div><div id="profile-label" class="profile-label">未导入，使用表单数值</div>
</details><small>参数保存在当前浏览器中。仿真由本地 Python 引擎计算。</small>
<input type="file" id="settings-file" accept="application/json,.json" hidden><input type="file" id="profile-file" accept="application/json,.json" hidden>
</aside><main>
<div class="result-bar"><div><div class="state"><span id="dot" class="dot"></span><span id="status" role="status" aria-live="polite">正在加载参数…</span></div><div id="status-detail" class="status-detail"></div></div>
<div class="result-actions"><label class="auto"><input id="auto" type="checkbox" checked>自动更新</label><button id="run" class="primary">计算</button><button id="save-report" disabled>下载 HTML</button></div></div>
<div id="error" role="alert"></div><div id="stale-note">参数已变化，当前图表为上一次结果。</div>
<div id="empty" class="empty"><strong>计算、传输与激活内存</strong><span>首次仿真完成后将在这里显示。</span></div>
<iframe id="report" title="完整 H3 仿真报告"></iframe>
</main></div>
<script>
'use strict';
const $ = id => document.getElementById(id);
const STORAGE = 'seqattn.h3.simulator.v1';
let fields = [], importedProfile = null, revision = 0, running = false;
let timer = null, queued = false, resultHTML = null, lastResponse = null;
const node = (tag, text) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; return n; };
function state(label, type = '', detail = '') {
  $('status').textContent = label; $('dot').className = 'dot ' + type; $('status-detail').textContent = detail;
}
function showError(message) { $('error').textContent = message; }
function markStale() {
  $('save-report').disabled = true;
  if (resultHTML) { $('report').classList.add('stale'); $('stale-note').style.display = 'block'; }
}
function formFields() {
  const groups = [...new Set(fields.map(f => f.group))];
  groups.forEach((group, index) => {
    const box = node('fieldset'), legend = node('legend');
    const number = node('span', String(index + 1).padStart(2, '0')); number.className = 'number';
    legend.append(number, document.createTextNode(group)); box.append(legend);
    const grid = node('div'); grid.className = 'grid';
    for (const f of fields.filter(f => f.group === group)) {
      const wrap = node('div'); wrap.className = 'field' + (f.kind === 'text' || f.kind === 'choice' ? ' wide' : '');
      const label = node('label', f.label); label.htmlFor = f.name;
      const input = node(f.kind === 'choice' ? 'select' : 'input'); input.id = f.name; input.name = f.name;
      if (f.kind === 'choice') for (const [value, title] of f.choices) {
        const option = node('option', title); option.value = value; input.append(option);
      } else if (f.kind === 'text') { input.type = 'text'; input.maxLength = 4096; }
      else {
        input.type = 'number'; input.step = f.step;
        if (f.minimum !== null) input.min = f.minimum;
        if (f.maximum !== null) input.max = f.maximum;
        input.required = !f.kind.startsWith('optional');
        if (f.kind.startsWith('optional')) input.placeholder = '自动 / 不限制';
      }
      input.value = f.default === null ? '' : f.default;
      wrap.append(label, input); if (f.hint) wrap.append(node('small', f.hint)); grid.append(wrap);
    }
    box.append(grid);
    if (index >= 4) { const details = node('details'); details.append(node('summary', group), box); $('fields').append(details); }
    else $('fields').append(box);
  });
}
function parameters() {
  const result = {};
  for (const f of fields) {
    const input = $(f.name);
    const value = input.disabled && input.value === '' && !f.kind.startsWith('optional') ? String(f.default) : input.value;
    result[f.name] = f.kind === 'choice' || f.kind === 'text' ? value : value === '' && f.kind.startsWith('optional') ? null : Number(value);
  }
  return result;
}
function apply(values) {
  const known = new Set(fields.map(f => f.name));
  if (!values || Array.isArray(values) || typeof values !== 'object') throw new Error('参数必须是 JSON 对象。');
  for (const key of Object.keys(values)) if (!known.has(key)) throw new Error('未知参数：' + key);
  for (const f of fields) {
    const value = Object.hasOwn(values, f.name) ? values[f.name] : f.default;
    if (value === null && f.kind.startsWith('optional')) continue;
    if (f.kind === 'text' || f.kind === 'choice') {
      if (typeof value !== 'string' || (f.kind === 'choice' && !f.choices.some(([key]) => key === value))) throw new Error(f.label + ' 的值无效');
    } else if (typeof value !== 'number' || !Number.isFinite(value) || (f.kind.includes('integer') && !Number.isInteger(value))) throw new Error(f.label + ' 必须是数字');
  }
  for (const f of fields) {
    const value = Object.hasOwn(values, f.name) ? values[f.name] : f.default;
    $(f.name).value = value === null ? '' : value;
  }
  dependentFields();
}
function dependentFields() {
  const segmented = $('segments').value.trim() !== '';
  $('tokens').disabled = segmented;
  if (segmented) {
    const parts = $('segments').value.trim().split(/[,，;；\s]+/);
    if (parts.every(p => /^[0-9]+$/.test(p))) $('tokens').value = parts.reduce((sum, p) => sum + Number(p), 0);
  }
  $('ffn_tile_tokens').disabled = $('ffn_candidates').value.trim() !== '';
  $('projection_tile_tokens').title = $('execution_mode').value === 'recompute' ? 'Recompute 不使用 materialized projection tile。' : '';
  for (const f of fields.filter(f => ['吞吐与带宽', '分算子速率'].includes(f.group))) $(f.name).disabled = importedProfile !== null;
  $('clear-profile').disabled = importedProfile === null;
  $('profile-label').textContent = importedProfile === null ? '未导入，使用表单数值' : '使用 Profile：' + (importedProfile.name || '未命名');
}
function settings() { return { version: 1, parameters: parameters(), profile: importedProfile }; }
function persist() { try { localStorage.setItem(STORAGE, JSON.stringify(settings())); } catch {} }
function changed() {
  revision++; clearTimeout(timer); queued = false; dependentFields(); markStale(); showError('');
  if (!$('form').checkValidity()) { state('请补全有效参数', 'error'); return; }
  state(running ? '参数已更新，等待当前计算完成…' : '等待更新…', running ? 'busy' : '');
  if ($('auto').checked) timer = setTimeout(calculate, 400);
}
function previousView() {
  try {
    const w = $('report').contentWindow, d = w.document;
    if (!d.getElementById('candidate')) return null;
    return { y: w.scrollY, values: Object.fromEntries(['range-start', 'range-end', 'backdrop', 'lifetime-pool', 'buffer-filter'].map(id => [id, d.getElementById(id)?.value])) };
  } catch { return null; }
}
function display(result, mine) {
  const view = previousView(); resultHTML = result.html; lastResponse = result;
  $('report').onload = () => {
    if (mine !== revision) return;
    try {
      const w = $('report').contentWindow, d = w.document;
      if (view) {
        for (const [id, value] of Object.entries(view.values)) {
          const element = d.getElementById(id);
          if (value !== undefined && element && (element.tagName !== 'SELECT' || [...element.options].some(o => o.value === value))) element.value = value;
        }
        w.redraw(); w.scrollTo(0, view.y);
      }
    } catch {}
    $('report').classList.remove('stale'); $('stale-note').style.display = 'none'; $('empty').style.display = 'none';
    $('save-report').disabled = false;
    state(result.selected_index === null ? '已更新 · 没有候选满足预算 / 性能目标' : '已更新', 'ready',
      `${result.candidate_count} 个候选 · ${result.actual_events.toLocaleString()} 个事件 · Python 计算 ${Math.round(result.elapsed_ms)} ms`);
  };
  $('report').srcdoc = result.html;
}
async function calculate() {
  clearTimeout(timer);
  if (!$('form').checkValidity()) { markStale(); state('请补全有效参数', 'error'); $('form').reportValidity(); return; }
  if (running) { queued = true; return; }
  const mine = revision;
  running = true; queued = false; markStale(); showError(''); state('正在计算 H3 完整路径…', 'busy');
  let retry = false;
  try {
    const response = await fetch('/api/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parameters: parameters(), profile: importedProfile }) });
    const result = await response.json();
    if (mine !== revision) return;
    if (response.status === 429) { retry = true; state('服务正在计算，稍后重试…', 'busy'); return; }
    if (!response.ok) throw new Error(result.error || '仿真请求失败');
    persist(); display(result, mine);
  } catch (error) {
    if (mine === revision) { showError(error.message); state('参数或服务需要检查', 'error'); }
  } finally {
    running = false;
    if (retry && mine === revision) timer = setTimeout(calculate, 600);
    else if (queued) calculate();
  }
}
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type })), a = node('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function readFile(input) {
  const file = input.files[0]; if (!file) return null;
  if (file.size > 1024 * 1024) throw new Error('导入文件不能超过 1 MiB。');
  return JSON.parse(await file.text());
}
$('form').addEventListener('input', changed);
$('form').addEventListener('submit', event => { event.preventDefault(); calculate(); });
$('run').addEventListener('click', calculate);
$('auto').addEventListener('change', () => { clearTimeout(timer); if ($('auto').checked) calculate(); });
$('defaults').addEventListener('click', () => { importedProfile = null; apply({}); persist(); changed(); });
$('save-settings').addEventListener('click', () => download('h3-parameters.json', JSON.stringify(settings(), null, 2), 'application/json'));
$('save-report').addEventListener('click', () => { if (resultHTML && !$('save-report').disabled) download('h3-simulation.html', resultHTML, 'text/html'); });
$('import').addEventListener('click', () => $('settings-file').click());
$('load-profile').addEventListener('click', () => $('profile-file').click());
$('clear-profile').addEventListener('click', () => { importedProfile = null; changed(); });
$('settings-file').addEventListener('change', async () => {
  try {
    const data = await readFile($('settings-file')); if (!data) return;
    if (data.version !== undefined && data.version !== 1) throw new Error('不支持的参数文件版本。');
    apply(data.parameters || data); importedProfile = data.profile || null; changed();
  } catch (error) { showError('导入失败：' + error.message); }
  $('settings-file').value = '';
});
$('profile-file').addEventListener('change', async () => {
  try {
    const data = await readFile($('profile-file')); if (!data) return;
    if (!data.operators || !data.device_pool || !data.host_pool) throw new Error('需要 H3DeviceProfile JSON。');
    importedProfile = data; changed();
  } catch (error) { showError('Profile 导入失败：' + error.message); }
  $('profile-file').value = '';
});
(async () => {
  try {
    const response = await fetch('/api/schema'); if (!response.ok) throw new Error('无法加载表单定义');
    fields = (await response.json()).fields; formFields();
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE));
      if (saved?.version === 1) { apply(saved.parameters); importedProfile = saved.profile || null; }
    } catch {}
    dependentFields(); calculate();
  } catch (error) { showError(error.message); state('加载失败', 'error'); }
})();
</script></body></html>"""
