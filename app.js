// ============================================================
// 1. 模型來源：預設 YOLO11n（model.onnx），或使用者上傳（存在瀏覽器 IndexedDB，熱更新刷新後仍在）
//    支援的輸出格式（自動判斷）：
//      [1, 4+類別數, N]  YOLOv8 / YOLO11（預設）
//      [1, N, 4+類別數]  轉置版
//      [1, N, 6]         已內建 NMS（x1,y1,x2,y2,score,class），N ≤ 1000
// ============================================================
const DEFAULT_NAMES = ['人','腳踏車','汽車','機車','飛機','公車','火車','卡車','船','紅綠燈','消防栓','停止標誌','停車計時器','長椅',
  '鳥','貓','狗','馬','羊','牛','大象','熊','斑馬','長頸鹿','背包','雨傘','手提包','領帶','行李箱','飛盤','滑雪板','滑雪單板',
  '球','風箏','棒球棒','棒球手套','滑板','衝浪板','網球拍','瓶子','酒杯','杯子','叉子','刀子','湯匙','碗','香蕉','蘋果',
  '三明治','柳橙','花椰菜','紅蘿蔔','熱狗','披薩','甜甜圈','蛋糕','椅子','沙發','盆栽','床','餐桌','馬桶','電視','筆電',
  '滑鼠','遙控器','鍵盤','手機','微波爐','烤箱','烤麵包機','水槽','冰箱','書','時鐘','花瓶','剪刀','泰迪熊','吹風機','牙刷'];

let NAMES = DEFAULT_NAMES;
let SIZE = 640;                                   // 模型輸入邊長，載入模型後自動偵測
let modelBytes = null, modelLabel = '', outDims = null;
const name = i => NAMES[i] ?? `類別 ${i}`;
const color = i => `hsl(${(i * 47) % 360} 85% 58%)`;

// ---- IndexedDB：記住上傳的模型 ----
const idb = () => new Promise((res, rej) => {
  const r = indexedDB.open('yolo-dev', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('kv');
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
});
const idbGet = async k => { const db = await idb(); return new Promise((res, rej) => { const q = db.transaction('kv').objectStore('kv').get(k); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }); };
const idbSet = async (k, v) => { const db = await idb(); return new Promise((res, rej) => { const t = db.transaction('kv', 'readwrite'); t.objectStore('kv').put(v, k); t.oncomplete = res; t.onerror = () => rej(t.error); }); };
const idbDel = async k => { const db = await idb(); return new Promise(res => { const t = db.transaction('kv', 'readwrite'); t.objectStore('kv').delete(k); t.oncomplete = res; }); };

// ---- 類別名稱解析 ----
// 1) .txt 每行一個；2) .json 陣列或 {"0":"person"}；3) data.yaml 的 names: 區塊（含 [a, b] 或 "0: a" 寫法）
function parseNames(text) {
  text = text.replace(/^﻿/, '').trim();
  try {
    const j = JSON.parse(text);
    if (Array.isArray(j)) return j.map(String);
    if (j && typeof j === 'object') return Object.keys(j).sort((a, b) => a - b).map(k => String(j[k]));
  } catch {}
  const inline = text.match(/names\s*:\s*\[([^\]]*)\]/);
  if (inline) return inline[1].split(',').map(s => s.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean);
  const body = /^names\s*:/m.test(text) ? text.split(/^names\s*:/m)[1] : text;
  const pairs = [...body.matchAll(/^\s*(\d+)\s*:\s*['"]?([^'"\r\n#]+?)['"]?\s*$/gm)];
  if (pairs.length) { const a = []; pairs.forEach(m => a[+m[1]] = m[2]); return a; }
  return text.split(/\r?\n/).map(s => s.trim()).filter(s => s && !s.startsWith('#'));
}
// Ultralytics 匯出的 ONNX 會把 names 以 "{0: 'person', 1: ...}" 存在 metadata，直接從檔尾抓
function namesFromOnnx(bytes) {
  const tail = new TextDecoder('utf-8', { fatal: false }).decode(bytes.subarray(Math.max(0, bytes.length - 60000)));
  const i = tail.lastIndexOf('names');
  if (i < 0) return null;
  const m = tail.slice(i).match(/\{\s*0\s*:[^}]*\}/);
  if (!m) return null;
  const out = []; for (const p of m[0].matchAll(/(\d+)\s*:\s*(?:'([^']*)'|"([^"]*)")/g)) out[+p[1]] = p[2] ?? p[3];
  return out.length ? out : null;
}

// ============================================================
// 2. ONNX Runtime Web：同一個模型可以跑在 Wasm 或 WebGPU 上
// ============================================================
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
ort.env.wasm.numThreads = 1;   // 沒有跨來源隔離，不能用多執行緒

let session = null, backend = navigator.gpu ? 'webgpu' : 'wasm', switching = false;
let prep, pctx, input;

function inputSizeOf(s) {
  try {
    const m = s.inputMetadata, meta = Array.isArray(m) ? m[0] : m[s.inputNames[0]];
    const h = meta.shape[2], w = meta.shape[3];
    if (Number.isInteger(h) && h > 0 && h === w) return h;
  } catch {}
  return 640;
}
function setSize(n) {
  SIZE = n;
  prep = document.createElement('canvas'); prep.width = prep.height = SIZE;
  pctx = prep.getContext('2d', { willReadFrequently: true });
  input = new Float32Array(3 * SIZE * SIZE);
  $('sizeV').textContent = `${SIZE} × ${SIZE}`;
}

// ---- fp16 模型：輸入要送 float16、輸出讀回來要轉成 float32 ----
const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
function f2h(v) {
  _f[0] = v; const x = _u[0], sign = (x >>> 16) & 0x8000, m = x & 0x7fffff;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  if (e <= 0) return e < -10 ? sign : sign | ((m | 0x800000) >> (14 - e));
  return e >= 31 ? sign | 0x7c00 : sign | (e << 10) | (m >> 13);
}
function h2f(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}
let input16 = null;
function makeInput(f32, size, f16) {
  const shape = [1, 3, size, size];
  if (!f16) return new ort.Tensor('float32', f32, shape);
  if (!input16 || input16.length !== f32.length) input16 = typeof Float16Array !== 'undefined' ? new Float16Array(f32.length) : new Uint16Array(f32.length);
  if (typeof Float16Array !== 'undefined') input16.set(f32);
  else for (let i = 0; i < f32.length; i++) input16[i] = f2h(f32[i]);
  return new ort.Tensor('float16', input16, shape);
}
function toF32(data) {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) { const o = new Float32Array(data.length); for (let i = 0; i < o.length; i++) o[i] = h2f(data[i]); return o; }
  return Float32Array.from(data);                // Float16Array
}
function isF16(s) {
  try { const m = s.inputMetadata; return (Array.isArray(m) ? m[0] : m[s.inputNames[0]]).type === 'float16'; } catch { return false; }
}

let inF16 = false;
async function createSession(b) {
  switching = true;
  setStatus(`建立 ${b === 'webgpu' ? 'WebGPU' : 'Wasm'} 推論環境中…`);
  try {
    const s = await ort.InferenceSession.create(modelBytes, { executionProviders: [b], graphOptimizationLevel: 'all' });
    const size = inputSizeOf(s), f16 = isF16(s);
    // 先跑一次暖機，第一次通常特別慢；順便取得輸出形狀
    const res = await s.run({ [s.inputNames[0]]: makeInput(new Float32Array(3 * size * size), size, f16) });
    outDims = res[s.outputNames[0]].dims;
    if (session) await session.release();
    session = s; backend = b; inF16 = f16; setSize(size);
    setStatus(`使用 ${b === 'webgpu' ? 'WebGPU（顯卡）' : 'Wasm（CPU）'} 執行中`);
    return true;
  } catch (e) {
    setStatus(`${b} 無法使用：${e.message}`, true);
    if (b === 'webgpu' && !session) return createSession('wasm');
    return false;
  } finally {
    switching = false; renderBackend();
  }
}

// 套用一個模型；失敗就還原成先前的
async function applyModel(bytes, names, label) {
  const prev = { modelBytes, NAMES, modelLabel };
  modelBytes = bytes; modelLabel = label;
  if (!(await createSession(backend))) { ({ modelBytes, NAMES, modelLabel } = prev); return false; }
  NAMES = names ?? namesFromOnnx(bytes) ?? [];
  dets = [];
  renderModel();
  return true;
}

async function loadDefault() {
  const bytes = new Uint8Array(await (await fetch('model.onnx')).arrayBuffer());
  return applyModel(bytes, DEFAULT_NAMES, 'YOLO11n（預設，80 類）');
}

// ============================================================
// 3. 前處理：鏡頭畫面 → 等比縮放貼到 SIZE×SIZE（不足補灰邊）→ Float32 CHW
// ============================================================
function preprocess(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(SIZE / vw, SIZE / vh);
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  const dx = (SIZE - w) / 2, dy = (SIZE - h) / 2;
  pctx.fillStyle = 'rgb(114,114,114)'; pctx.fillRect(0, 0, SIZE, SIZE);
  pctx.drawImage(video, dx, dy, w, h);
  const px = pctx.getImageData(0, 0, SIZE, SIZE).data;
  const area = SIZE * SIZE;
  for (let i = 0; i < area; i++) {               // RGBA 交錯排列 → R 平面、G 平面、B 平面
    input[i] = px[i * 4] / 255;
    input[i + area] = px[i * 4 + 1] / 255;
    input[i + area * 2] = px[i * 4 + 2] / 255;
  }
  return { scale, dx, dy };
}

// ============================================================
// 4. 後處理：依輸出形狀解碼 → 門檻篩選 → NMS 去掉重疊框
// ============================================================
function postprocess(out, conf, { scale, dx, dy }) {
  const d = outDims, back = (x1, y1, x2, y2, score, cls) =>
    ({ cls, score, x1: (x1 - dx) / scale, y1: (y1 - dy) / scale, x2: (x2 - dx) / scale, y2: (y2 - dy) / scale });
  const boxes = [];

  if (d.length === 3 && d[2] === 6 && d[1] <= 1000) {            // 內建 NMS：每列 x1,y1,x2,y2,score,cls
    for (let i = 0; i < d[1]; i++) {
      const r = i * 6, score = out[r + 4];
      if (score >= conf) boxes.push(back(out[r], out[r + 1], out[r + 2], out[r + 3], score, Math.round(out[r + 5])));
    }
    return boxes;
  }

  const chanFirst = d[1] < d[2];                                  // [1, 4+nc, N] 或 [1, N, 4+nc]
  const C = chanFirst ? d[1] : d[2], N = chanFirst ? d[2] : d[1], nc = C - 4;
  const at = chanFirst ? (i, k) => out[k * N + i] : (i, k) => out[i * C + k];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) { const s = at(i, 4 + c); if (s > best) { best = s; cls = c; } }
    if (best < conf) continue;
    const cx = at(i, 0), cy = at(i, 1), w = at(i, 2), h = at(i, 3);
    boxes.push(back(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, cls));
  }
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) {
    if (keep.every(k => k.cls !== b.cls || iou(k, b) < 0.45)) keep.push(b);
    if (keep.length >= 100) break;
  }
  return keep;
}
function iou(a, b) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = w * h;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// ============================================================
// 5. 鏡頭
// ============================================================
const video = document.createElement('video'); video.muted = true; video.playsInline = true;
let stream = null;

async function listCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const sel = $('camera'), cur = stream?.getVideoTracks()[0]?.getSettings().deviceId;
  sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${d.label || '鏡頭 ' + (i + 1)}</option>`).join('');
  if (cur) sel.value = cur;
}

async function startCamera(deviceId) {
  stream?.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                    : { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  view.width = video.videoWidth; view.height = video.videoHeight;
  $('placeholder').hidden = true;
  await listCameras();
}

// ============================================================
// 6. 主迴圈：畫面每幀都畫；推論做完一次才做下一次
// ============================================================
const view = $('view'), vctx = view.getContext('2d');
let dets = [], busy = false, frames = 0, lastFps = performance.now();

function draw() {
  if (video.readyState >= 2) {
    vctx.drawImage(video, 0, 0);
    const lw = Math.max(2, view.width / 400), fs = Math.max(14, view.width / 50);
    vctx.lineWidth = lw; vctx.font = `600 ${fs}px system-ui, sans-serif`; vctx.textBaseline = 'top';
    for (const d of dets) {
      const col = color(d.cls), label = `${name(d.cls)} ${(d.score * 100).toFixed(0)}%`;
      vctx.strokeStyle = col; vctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      const tw = vctx.measureText(label).width + 10, ty = d.y1 > fs + 6 ? d.y1 - fs - 6 : d.y1;
      vctx.fillStyle = col; vctx.fillRect(d.x1 - lw / 2, ty, tw, fs + 6);
      vctx.fillStyle = '#000'; vctx.fillText(label, d.x1 + 5, ty + 3);
    }
  }
  requestAnimationFrame(draw);
}

async function detectLoop() {
  if (session && !switching && !busy && video.readyState >= 2) {
    busy = true;
    const s = session, f16 = inF16;
    try {
      const t0 = performance.now();
      const meta = preprocess(video);
      const t1 = performance.now();
      const res = await s.run({ [s.inputNames[0]]: makeInput(input, SIZE, f16) });
      const out = toF32(res[s.outputNames[0]].data);
      const t2 = performance.now();
      dets = s === session ? postprocess(out, +$('conf').value, meta) : [];
      const t3 = performance.now();
      $('tPre').textContent = (t1 - t0).toFixed(1) + ' ms';
      $('inf').textContent = (t2 - t1).toFixed(0);
      $('tPost').textContent = (t3 - t2).toFixed(1) + ' ms';
      renderObjects();
      frames++;
    } catch (e) { setStatus('推論錯誤：' + e.message, true); }
    busy = false;
  }
  const now = performance.now();
  if (now - lastFps > 1000) { $('fps').textContent = (frames * 1000 / (now - lastFps)).toFixed(1); frames = 0; lastFps = now; }
  setTimeout(detectLoop, 0);
}

// ============================================================
// 7. 介面
// ============================================================
function $(id) { return document.getElementById(id); }
function setStatus(t, err) { $('status').textContent = t; $('status').classList.toggle('err', !!err); }
function renderBackend() { document.querySelectorAll('#backend button').forEach(b => b.classList.toggle('on', b.dataset.b === backend)); }
function renderModel() {
  const nc = outDims ? (outDims[2] === 6 && outDims[1] <= 1000 ? NAMES.length : Math.min(outDims[1], outDims[2]) - 4) : NAMES.length;
  const hasNames = NAMES.length >= nc;
  $('modelName').textContent = `${modelLabel}${inF16 ? '（fp16）' : ''}　${nc} 類${hasNames ? '' : '（缺類別名稱，顯示為「類別 N」）'}`;
  $('resetModel').hidden = modelLabel.startsWith('YOLO11n');
}
function renderObjects() {
  const count = {};
  for (const d of dets) count[d.cls] = (count[d.cls] || 0) + 1;
  const keys = Object.keys(count).sort((a, b) => count[b] - count[a]);
  $('objects').innerHTML = keys.length
    ? keys.map(k => `<div><span><i style="background:${color(k)}"></i>${name(k)}</span><b>${count[k]}</b></div>`).join('')
    : '<div style="color:var(--muted)">沒有偵測到物品</div>';
}

// ---- 上傳 ----
async function handleFiles(files) {
  let model = null, names = null;
  for (const f of files) {
    if (/\.onnx$/i.test(f.name)) model = f;
    else if (/\.(txt|json|ya?ml)$/i.test(f.name)) names = f;
    else if (/\.(pt|pth|engine|tflite)$/i.test(f.name)) { setStatus(`${f.name} 不能直接在瀏覽器跑，請先匯出成 ONNX：yolo export model=${f.name} format=onnx`, true); return; }
  }
  if (!model && !names) { setStatus('請選 .onnx 模型，或 .txt/.json/.yaml 類別檔', true); return; }

  const parsedNames = names ? parseNames(await names.text()) : null;
  if (model) {
    setStatus(`載入 ${model.name}…`);
    const bytes = new Uint8Array(await model.arrayBuffer());
    if (!(await applyModel(bytes, parsedNames, model.name))) return;
    await idbSet('model', { bytes: bytes.buffer, names: NAMES, label: model.name }).catch(() => setStatus('模型太大，無法記住（重新整理後需重傳）', true));
  } else {                                       // 只換類別名稱
    NAMES = parsedNames; renderModel();
    const rec = await idbGet('model').catch(() => null);
    if (rec) await idbSet('model', { ...rec, names: NAMES });
  }
}

$('pickModel').onclick = () => $('fileModel').click();
$('pickNames').onclick = () => $('fileNames').click();
$('fileModel').onchange = e => { handleFiles([...e.target.files]); e.target.value = ''; };
$('fileNames').onchange = e => { handleFiles([...e.target.files]); e.target.value = ''; };
$('resetModel').onclick = async () => { await idbDel('model'); await loadDefault(); };
addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('drag'); });
addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('drag'); });
addEventListener('drop', e => { e.preventDefault(); document.body.classList.remove('drag'); handleFiles([...e.dataTransfer.files]); });

$('conf').oninput = () => $('confV').textContent = (+$('conf').value * 100).toFixed(0) + '%';
$('conf').oninput();

$('start').onclick = async () => {
  try { await startCamera($('camera').value); $('start').textContent = '重新啟動鏡頭'; }
  catch (e) { setStatus('無法開啟鏡頭：' + e.message + '（請確認已允許攝影機權限，且沒有其他程式占用）', true); }
};
$('camera').onchange = () => { if (stream) startCamera($('camera').value).catch(e => setStatus(e.message, true)); };
$('backend').onclick = e => {
  const b = e.target.closest('button'); if (!b || b.dataset.b === backend || switching) return;
  if (b.dataset.b === 'webgpu' && !navigator.gpu) { setStatus('這個瀏覽器不支援 WebGPU', true); return; }
  createSession(b.dataset.b);
};

renderBackend();
(async () => {
  const saved = await idbGet('model').catch(() => null);          // 上次上傳的模型優先
  let ok = false;
  if (saved) ok = await applyModel(new Uint8Array(saved.bytes), saved.names?.length ? saved.names : null, saved.label);
  if (!ok) ok = await loadDefault();
  if (ok) { detectLoop(); draw(); }
})();
