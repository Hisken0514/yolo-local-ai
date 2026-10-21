import {
  ensureIsolation, detectEnv, renderEnv, renderWasmFiles, threads, threadChoices, setThreads,
  getPref, setPref, getSession, setSession, mb,
} from './wasm-env.js';

// ============================================================
// 1. 模型來源：預設 YOLO26n（model.onnx），或使用者上傳（存在瀏覽器 IndexedDB，熱更新刷新後仍在）
//    支援的輸出格式（自動判斷）：
//      [1, 4+類別數, N]  YOLOv8 / YOLO11 / YOLO26（預設）
//      [1, N, 4+類別數]  轉置版
//      [1, N, 6]         已內建 NMS（x1,y1,x2,y2,score,class），N ≤ 1000
// ============================================================
const DEFAULT_NAMES = ['人','腳踏車','汽車','機車','飛機','公車','火車','卡車','船','紅綠燈','消防栓','停止標誌','停車計時器','長椅',
  '鳥','貓','狗','馬','羊','牛','大象','熊','斑馬','長頸鹿','背包','雨傘','手提包','領帶','行李箱','飛盤','滑雪板','滑雪單板',
  '球','風箏','棒球棒','棒球手套','滑板','衝浪板','網球拍','瓶子','酒杯','杯子','叉子','刀子','湯匙','碗','香蕉','蘋果',
  '三明治','柳橙','花椰菜','紅蘿蔔','熱狗','披薩','甜甜圈','蛋糕','椅子','沙發','盆栽','床','餐桌','馬桶','電視','筆電',
  '滑鼠','遙控器','鍵盤','手機','微波爐','烤箱','烤麵包機','水槽','冰箱','書','時鐘','花瓶','剪刀','泰迪熊','吹風機','牙刷'];

const DEFAULT_LABEL = 'YOLO26n（預設，80 類）';
let NAMES = DEFAULT_NAMES;
let SIZE = 640;                                   // 模型輸入邊長，載入模型後自動偵測
let modelBytes = null, modelLabel = '', outDims = null;
const name = i => NAMES[i] ?? `類別 ${i}`;
const color = i => `hsl(${(i * 47) % 360} 85% 58%)`;
const esc = s => String(s).replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);

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
// 2. ONNX Runtime Web：C++ 推論引擎編譯成 WebAssembly；同一個模型可以跑在 Wasm（CPU）或 WebGPU（顯卡）
//    執行緒數必須在第一次建立 session 前設定，之後要改只能重新整理頁面
// ============================================================
const THREADS = threads();

// SIMD 開關：ONNX Runtime 1.19 起只提供 SIMD 版的 .wasm，env.wasm.simd = false 不會有作用。
// 所以關閉 SIMD 時改載入仍有「無 SIMD 版」的 1.18.0（預設關閉，iPhone 較不易崩潰）；開啟時用 1.24.3。
// 1.18.0 沒有 inputMetadata，所以無法自動偵測輸入尺寸（用 640）和 fp16 模型。
// 1.18.0 的 WebGPU 版程式只能搭配 SIMD 版 .wasm，所以無 SIMD 時載入純 Wasm 版（ort.min.js），不能用 WebGPU。
const SIMD = getPref('wasm-simd') === '1';
const ORT_VER = SIMD ? '1.24.3' : '1.18.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
await new Promise((res, rej) => {
  const s = document.createElement('script');
  s.src = ORT_BASE + (SIMD ? 'ort.webgpu.min.js' : 'ort.min.js'); s.crossOrigin = 'anonymous';
  s.onload = res; s.onerror = () => rej(new Error('ONNX Runtime 載入失敗'));
  document.head.append(s);
});
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = THREADS;
ort.env.wasm.simd = SIMD;

let ENV = null;                                   // detectEnv() 的結果
let session = null, switching = false;
let backend = getPref('yolo-backend') || (navigator.gpu ? 'webgpu' : 'wasm');
let prep, pctx, input;
const backendLabel = b => b === 'webgpu' ? 'WebGPU' : `WebAssembly · ${THREADS} 執行緒${SIMD ? '' : ' · 無 SIMD'}`;

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

const newSession = b => ort.InferenceSession.create(modelBytes, { executionProviders: [b], graphOptimizationLevel: 'all' });

let inF16 = false;
async function createSession(b) {
  switching = true;
  setStatus(`建立 ${b === 'webgpu' ? 'WebGPU' : 'WebAssembly'} 推論環境中…`, 'busy');
  try {
    const s = await newSession(b);
    const size = inputSizeOf(s), f16 = isF16(s);
    // 先跑一次暖機，第一次通常特別慢；順便取得輸出形狀
    const res = await s.run({ [s.inputNames[0]]: makeInput(new Float32Array(3 * size * size), size, f16) });
    outDims = res[s.outputNames[0]].dims;
    if (session) await session.release();
    session = s; backend = b; inF16 = f16; setSize(size);
    setPref('yolo-backend', b);
    resetTimes();
    setStatus(`使用 ${backendLabel(b)} 執行中`);
    return true;
  } catch (e) {
    setStatus(`${b} 無法使用：${e.message}`, true);
    if (b === 'webgpu' && !session) return createSession('wasm');
    return false;
  } finally {
    switching = false; renderBackend(); renderWasmFiles($('wasmFile'));
  }
}

// 套用一個模型；失敗就還原成先前的
async function applyModel(bytes, names, label) {
  const prev = { modelBytes, NAMES, modelLabel };
  modelBytes = bytes; modelLabel = label;
  if (!(await createSession(backend))) { ({ modelBytes, NAMES, modelLabel } = prev); return false; }
  NAMES = names ?? namesFromOnnx(bytes) ?? [];
  dets = [];
  renderModel(); renderBench();
  return true;
}

async function loadDefault() {
  const bytes = new Uint8Array(await (await fetch('model.onnx')).arrayBuffer());
  return applyModel(bytes, DEFAULT_NAMES, DEFAULT_LABEL);
}

// 物件偵測模式：上次上傳的模型優先，否則用預設模型
async function loadObjectsModel() {
  const saved = await idbGet('model').catch(() => null);
  let ok = false;
  if (saved) ok = await applyModel(new Uint8Array(saved.bytes), saved.names?.length ? saved.names : null, saved.label);
  if (!ok) ok = await loadDefault();
  return ok;
}

// ============================================================
// 2b. 年齡・性別模式（兩段式）：人臉偵測 → 每張臉裁成 224×224 → 年齡性別模型
//     兩個模型都在執行時從 Hugging Face 下載，存在瀏覽器的 Cache Storage，不放進這個網站
//     年齡性別模型輸出 [n, 2]：第 1 個值是歲數（四捨五入、限制 0~100），第 2 個值是女性機率（≥ 0.5 為女性）
// ============================================================
const FACE_URL = 'https://huggingface.co/AdamCodd/yolo11n-face-age/resolve/main/model.onnx';                // CC-BY-NC-4.0
const AGE_URL = 'https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main/onnx/model.onnx'; // Apache-2.0
const FACE_LABEL = '人臉偵測 yolo11n-face-age';
const FACE_NAMES = ['人臉', '人臉', '人臉'];   // 原本是 3 個年齡區間，這裡只拿來找臉
const AGE_SIZE = 224, AGE_MAX_FACES = 6;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let mode = getPref('yolo-mode') === 'age' ? 'age' : 'objects';
let ageSession = null, ageBytes = null, modeBusy = false;
const ageCanvas = document.createElement('canvas'); ageCanvas.width = ageCanvas.height = AGE_SIZE;
const actx = ageCanvas.getContext('2d', { willReadFrequently: true });

async function cachedFetch(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open('hf-models-v1');
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {}
  const res = await fetch(url);
  if (!res.ok) throw new Error(`下載失敗（HTTP ${res.status}）`);
  const total = +res.headers.get('content-length') || 0, reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress?.(got, total); }
  const bytes = new Uint8Array(got); let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  try { await cache?.put(url, new Response(bytes)); } catch {}
  return bytes;
}

async function createAgeSession(b) {
  const s = await ort.InferenceSession.create(ageBytes, { executionProviders: [b], graphOptimizationLevel: 'all' });
  await s.run({ pixel_values: new ort.Tensor('float32', new Float32Array(3 * AGE_SIZE * AGE_SIZE), [1, 3, AGE_SIZE, AGE_SIZE]) });
  if (ageSession) await ageSession.release();
  ageSession = s;
}

async function loadAgeMode() {
  const bar = what => (got, total) => {
    $('ageProgress').hidden = false; $('ageBar').style.width = total ? (got / total * 100) + '%' : '30%';
    setStatus(`下載${what} ${mb(got)}${total ? ' / ' + mb(total) : ''}`, 'busy');
  };
  try {
    const face = await cachedFetch(FACE_URL, bar('人臉模型'));
    ageBytes = await cachedFetch(AGE_URL, bar('年齡模型'));
    $('ageProgress').hidden = true;
    if (!(await applyModel(face, FACE_NAMES, FACE_LABEL))) return false;
    setStatus('建立年齡模型推論環境中…', 'busy');
    try { await createAgeSession(backend); }
    catch (e) { if (backend === 'wasm') throw e; await createAgeSession('wasm'); }
    setStatus(`年齡・性別模式：${backendLabel(backend)}`);
    return true;
  } catch (e) {
    $('ageProgress').hidden = true; ageBytes = null;
    setStatus('年齡模式載入失敗：' + e.message, true);
    return false;
  }
}

// 每張臉往外多取 20% 再縮放成 224×224，正規化後一次批次推論
async function runAge(faces) {
  const list = faces.slice(0, AGE_MAX_FACES).filter(d => d.x2 > d.x1 && d.y2 > d.y1);
  if (!list.length) return;
  const n = list.length, area = AGE_SIZE * AGE_SIZE, data = new Float32Array(n * 3 * area);
  const [vw, vh] = srcSize();
  list.forEach((d, k) => {
    const m = 0.2 * Math.max(d.x2 - d.x1, d.y2 - d.y1);
    const x1 = Math.max(0, d.x1 - m), y1 = Math.max(0, d.y1 - m), x2 = Math.min(vw, d.x2 + m), y2 = Math.min(vh, d.y2 + m);
    actx.drawImage(src.el, x1, y1, x2 - x1, y2 - y1, 0, 0, AGE_SIZE, AGE_SIZE);
    const px = actx.getImageData(0, 0, AGE_SIZE, AGE_SIZE).data, off = k * 3 * area;
    for (let i = 0; i < area; i++)
      for (let c = 0; c < 3; c++) data[off + c * area + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  });
  const out = (await ageSession.run({ pixel_values: new ort.Tensor('float32', data, [n, 3, AGE_SIZE, AGE_SIZE]) })).logits.data;
  list.forEach((d, k) => { d.age = Math.min(100, Math.max(0, Math.round(out[k * 2]))); d.female = out[k * 2 + 1]; });
}
const genderOf = d => d.female >= 0.5 ? '女' : '男';

// ============================================================
// 3. 輸入來源：鏡頭、影片檔（循環播放）、圖片（同一張圖反覆推論，方便量測）
// ============================================================
let src = null;                                   // { el, kind: 'camera' | 'video' | 'image' }
const camVideo = document.createElement('video'); camVideo.muted = true; camVideo.playsInline = true;
let stream = null, fileVideo = null, fileUrl = null;

const srcReady = () => !!src && (src.kind === 'image' || src.el.readyState >= 2);
const srcSize = () => src.kind === 'image' ? [src.el.naturalWidth, src.el.naturalHeight] : [src.el.videoWidth, src.el.videoHeight];

function useSource(el, kind) {
  src = { el, kind }; dets = [];
  const [w, h] = srcSize(); view.width = w; view.height = h;
  $('placeholder').hidden = true; $('hud').hidden = false;
}
function stopCamera() { stream?.getTracks().forEach(t => t.stop()); stream = null; }
function stopFile() {
  if (fileVideo) { fileVideo.pause(); fileVideo.removeAttribute('src'); fileVideo.load(); fileVideo = null; }
  if (fileUrl) { URL.revokeObjectURL(fileUrl); fileUrl = null; }
}

async function listCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const sel = $('camera'), cur = stream?.getVideoTracks()[0]?.getSettings().deviceId;
  sel.innerHTML = devs.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || '鏡頭 ' + (i + 1))}</option>`).join('');
  if (cur) sel.value = cur;
}

async function startCamera(deviceId) {
  stopCamera();
  stream = await navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                    : { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  stopFile();
  camVideo.srcObject = stream;
  await camVideo.play();
  useSource(camVideo, 'camera');
  $('startCam').textContent = '重新開啟鏡頭';
  await listCameras();
}

async function openMedia(file) {
  const url = URL.createObjectURL(file);
  try {
    if (file.type.startsWith('image/')) {
      // 用 onload 而不是 img.decode()：decode() 在部分瀏覽器環境會一直不回應
      const img = new Image();
      await new Promise((res, rej) => { img.onload = res; img.onerror = () => rej(new Error('圖片格式不支援')); img.src = url; });
      stopCamera(); stopFile(); fileUrl = url;
      useSource(img, 'image');
    } else {
      const v = document.createElement('video'); v.muted = true; v.loop = true; v.playsInline = true; v.src = url;
      await v.play();
      stopCamera(); stopFile(); fileUrl = url; fileVideo = v;
      useSource(v, 'video');
    }
    $('startCam').textContent = '開啟鏡頭';
    setStatus(`使用檔案 ${file.name}`);
  } catch (e) {
    URL.revokeObjectURL(url);
    setStatus(`無法開啟 ${file.name}：${e.message || '格式不支援'}`, true);
  }
}

// ============================================================
// 4. 前處理（JavaScript）：畫面 → 等比縮放貼到 SIZE×SIZE（不足補灰邊）→ Float32 CHW
// ============================================================
function preprocess(el) {
  const [vw, vh] = srcSize();
  const scale = Math.min(SIZE / vw, SIZE / vh);
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  const dx = (SIZE - w) / 2, dy = (SIZE - h) / 2;
  pctx.fillStyle = 'rgb(114,114,114)'; pctx.fillRect(0, 0, SIZE, SIZE);
  pctx.drawImage(el, dx, dy, w, h);
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
// 5. 後處理（JavaScript）：依輸出形狀解碼 → 門檻篩選 → NMS 去掉重疊框
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
// 6. 主迴圈：畫面每幀都畫；推論做完一次才做下一次。效能測試時暫停
// ============================================================
const view = $('view'), vctx = view.getContext('2d');
let dets = [], busy = false, paused = false, frames = 0, lastFps = performance.now(), fps = 0;

function draw() {
  if (srcReady()) {
    const [w, h] = srcSize();
    if (w && (view.width !== w || view.height !== h)) { view.width = w; view.height = h; }   // 手機轉向時鏡頭尺寸會變
    vctx.drawImage(src.el, 0, 0, view.width, view.height);
    const lw = Math.max(2, view.width / 400), fs = Math.max(14, view.width / 50);
    vctx.lineWidth = lw; vctx.font = `600 ${fs}px system-ui, sans-serif`; vctx.textBaseline = 'top';
    for (const d of dets) {
      const aged = d.age != null;
      const col = aged ? (d.female >= 0.5 ? 'hsl(330 85% 66%)' : 'hsl(200 85% 60%)') : color(d.cls);
      const label = aged
        ? `${d.age} 歲・${genderOf(d)} ${(Math.max(d.female, 1 - d.female) * 100).toFixed(0)}%`
        : `${name(d.cls)} ${(d.score * 100).toFixed(0)}%`;
      vctx.strokeStyle = col; vctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      const tw = vctx.measureText(label).width + 10, ty = d.y1 > fs + 6 ? d.y1 - fs - 6 : d.y1;
      vctx.fillStyle = col; vctx.fillRect(d.x1 - lw / 2, ty, tw, fs + 6);
      vctx.fillStyle = '#000'; vctx.fillText(label, d.x1 + 5, ty + 3);
    }
  }
  requestAnimationFrame(draw);
}

async function detectLoop() {
  if (session && !switching && !paused && !busy && srcReady()) {
    busy = true;
    const s = session, f16 = inF16;
    try {
      const t0 = performance.now();
      const meta = preprocess(src.el);
      const t1 = performance.now();
      const res = await s.run({ [s.inputNames[0]]: makeInput(input, SIZE, f16) });
      const out = toF32(res[s.outputNames[0]].data);
      const t2 = performance.now();
      const found = s === session ? postprocess(out, +$('conf').value, meta) : [];
      const t3 = performance.now();
      let tAge = 0;
      if (mode === 'age' && ageSession && found.length) {
        const ta = performance.now(); await runAge(found); tAge = performance.now() - ta;
        $('tAge').textContent = `${(tAge / Math.min(found.length, AGE_MAX_FACES)).toFixed(0)} ms × ${Math.min(found.length, AGE_MAX_FACES)}`;
      }
      dets = found;
      renderTimes(t1 - t0, t2 - t1 + tAge, t3 - t2);
      renderObjects();
      frames++;
    } catch (e) { setStatus('推論錯誤：' + e.message, true); }
    busy = false;
  }
  const now = performance.now();
  if (now - lastFps > 1000) {
    fps = frames * 1000 / (now - lastFps); frames = 0; lastFps = now;
    $('fps').textContent = paused || !srcReady() ? '-' : fps.toFixed(1);
    renderHud();
  }
  setTimeout(detectLoop, 0);
}

async function pauseLoop() { paused = true; while (busy) await new Promise(r => setTimeout(r, 10)); }
function resumeLoop() { paused = false; }

// ============================================================
// 7. 效能比較：同一個模型與畫面，換執行方式連續推論，取中位數
// ============================================================
const BENCH_KEY = 'yolo-bench', PLAN_KEY = 'yolo-thread-plan';
let benchHist = (() => { try { return JSON.parse(getPref(BENCH_KEY)) || []; } catch { return []; } })();
let benching = false;

function benchData(size) {
  if (srcReady() && size === SIZE) { preprocess(src.el); return input.slice(); }
  const d = new Float32Array(3 * size * size);                    // 沒有畫面時用隨機雜訊：推論時間和內容無關
  for (let i = 0; i < d.length; i++) d[i] = Math.random();
  return d;
}

// 至少 5 次、最多 20 次，或每種方式花滿約 6 秒為止
async function benchBackend(b, tag) {
  const s = await newSession(b);
  try {
    const size = inputSizeOf(s), f16 = isF16(s), data = benchData(size);
    const feed = () => ({ [s.inputNames[0]]: makeInput(data, size, f16) });
    for (let i = 0; i < 2; i++) await s.run(feed());            // 暖機
    const ts = [], start = performance.now();
    while (ts.length < 20 && (ts.length < 5 || performance.now() - start < 6000)) {
      const t = performance.now(); await s.run(feed()); ts.push(performance.now() - t);
      setStatus(`效能測試：${tag || backendLabel(b)}　${ts.length} 次`, 'busy');
    }
    ts.sort((a, c) => a - c);
    return { median: ts[ts.length >> 1], runs: ts.length };
  } finally { await s.release(); }
}

function record(b, r) {
  benchHist.push({ model: modelLabel, backend: b, threads: b === 'wasm' ? THREADS : 0, simd: b === 'wasm' ? SIMD : null, gpu: b === 'webgpu' ? ENV?.gpu || '' : '', median: r.median, runs: r.runs, at: Date.now() });
  benchHist = benchHist.slice(-40);
  setPref(BENCH_KEY, JSON.stringify(benchHist));
  renderBench();
}

async function runBenchmark() {
  if (benching || !modelBytes) return;
  benching = true; renderBenchButtons(); await pauseLoop();
  try {
    const list = ['wasm', ...(ENV?.webgpu && SIMD ? ['webgpu'] : [])], res = {};
    for (const b of list) { res[b] = await benchBackend(b); record(b, res[b]); }
    if (res.webgpu) {
      const fast = res.webgpu.median < res.wasm.median ? 'webgpu' : 'wasm', slow = fast === 'wasm' ? 'webgpu' : 'wasm';
      setStatus(`${backendLabel(fast)} 比 ${backendLabel(slow)} 快 ${(res[slow].median / res[fast].median).toFixed(1)} 倍`);
    } else setStatus(`${backendLabel('wasm')}：每次推論 ${res.wasm.median.toFixed(0)} ms（${SIMD ? '這個瀏覽器沒有 WebGPU 可比較' : '關閉 SIMD 時不能用 WebGPU'}）`);
  } catch (e) { setStatus('效能比較失敗：' + e.message, true); }
  benching = false; renderBenchButtons(); resumeLoop();
}

// 執行緒比較：執行緒數只能在初始化前設定，所以每一步都設定後重新整理頁面，結果暫存在 sessionStorage
function startThreadPlan() {
  const steps = [1, 2, 4, 8].filter(n => threadChoices().includes(n));
  if (steps.length < 2) return;
  setSession(PLAN_KEY, { orig: THREADS, steps, i: 0, results: [] });
  setThreads(steps[0]); location.reload();
}

async function continueThreadPlan() {
  const plan = getSession(PLAN_KEY);
  if (!plan) return;
  if (plan.done) { setSession(PLAN_KEY, null); showThreadSummary(plan); return; }
  const want = plan.steps[plan.i];
  if (THREADS !== want) { setSession(PLAN_KEY, null); setThreads(plan.orig); setStatus(`執行緒比較中斷：無法切換到 ${want} 個執行緒`, true); return; }

  benching = true; renderBenchButtons(); await pauseLoop();
  try {
    const r = await benchBackend('wasm', `WebAssembly · ${want} 執行緒（${plan.i + 1}/${plan.steps.length}）`);
    record('wasm', r); plan.results.push({ threads: want, median: r.median });
  } catch (e) {
    setSession(PLAN_KEY, null); setThreads(plan.orig);
    setStatus('執行緒比較失敗：' + e.message, true);
    benching = false; renderBenchButtons(); resumeLoop(); return;
  }
  plan.i++;
  if (plan.i < plan.steps.length) { setSession(PLAN_KEY, plan); setThreads(plan.steps[plan.i]); location.reload(); return; }
  plan.done = true; setThreads(plan.orig);
  if (plan.orig !== THREADS) { setSession(PLAN_KEY, plan); location.reload(); return; }
  setSession(PLAN_KEY, null); showThreadSummary(plan);
  benching = false; renderBenchButtons(); resumeLoop();
}

function showThreadSummary(plan) {
  const one = plan.results.find(r => r.threads === 1), best = plan.results.reduce((m, r) => r.median < m.median ? r : m);
  if (!one || best === one) { setStatus('執行緒比較完成：多執行緒在這台裝置上沒有變快'); return; }
  setStatus(`執行緒比較完成：1 個執行緒 ${one.median.toFixed(0)} ms → ${best.threads} 個執行緒 ${best.median.toFixed(0)} ms，快 ${(one.median / best.median).toFixed(1)} 倍`);
}

// ============================================================
// 8. 介面
// ============================================================
function $(id) { return document.getElementById(id); }
function setStatus(t, kind) {
  const s = $('status'); s.textContent = t;
  s.classList.toggle('err', kind === true); s.classList.toggle('busy', kind === 'busy');
}

function renderBackend() {
  document.querySelectorAll('#backend button').forEach(b => b.classList.toggle('on', b.dataset.b === backend));
  $('wasmSub').textContent = `CPU · ${THREADS} 執行緒`;
  if (ENV) {
    const gpuBtn = document.querySelector('#backend [data-b=webgpu]');
    gpuBtn.disabled = !ENV.webgpu || !SIMD;
    $('gpuSub').textContent = !SIMD ? '需先開啟 SIMD' : ENV.webgpu ? (ENV.gpu || '顯示卡') : '這個瀏覽器不支援';
  }
  const gpu = backend === 'webgpu';
  $('segInf').className = $('infSw').className = gpu ? 'gpu' : 'wasm';
  $('infSw').classList.add('sw');
  $('infName').textContent = `推論 · ${gpu ? 'WebGPU' : `WebAssembly ×${THREADS}`}`;
  $('hud').classList.toggle('gpu', gpu);
  renderHud();
}

function renderThreads() {
  const choices = threadChoices(), sel = $('threads');
  sel.innerHTML = choices.map(n => `<option value="${n}">${n} 個${n === 1 ? '（單執行緒）' : ''}</option>`).join('');
  sel.value = THREADS;
  sel.disabled = choices.length < 2 || benching;
  $('threadHint').textContent = !ENV?.isolated
    ? '這個頁面沒有跨來源隔離，瀏覽器不開放多執行緒，Wasm 只能用 1 個執行緒。'
    : choices.length < 2 ? '這台裝置只有 1 個 CPU 核心。'
    : `這台裝置有 ${ENV.cores} 個邏輯核心。變更後會重新載入頁面（ONNX Runtime 初始化後就不能改）。`;
}

// 平滑後的每幀時間
const ema = {};
const smooth = (k, v) => (ema[k] = ema[k] == null ? v : ema[k] * 0.8 + v * 0.2);
function resetTimes() { for (const k in ema) delete ema[k]; }
function renderTimes(pre, inf, post) {
  pre = smooth('pre', pre); inf = smooth('inf', inf); post = smooth('post', post);
  const total = pre + inf + post;
  $('segPre').style.width = (pre / total * 100) + '%';
  $('segInf').style.width = (inf / total * 100) + '%';
  $('segPost').style.width = (post / total * 100) + '%';
  $('tPre').textContent = pre.toFixed(1) + ' ms';
  $('tInf').textContent = inf.toFixed(1) + ' ms';
  $('tPost').textContent = post.toFixed(1) + ' ms';
  $('inf').textContent = inf.toFixed(0);
}
function renderHud() {
  const inf = ema.inf;
  $('hudText').textContent = `${backendLabel(backend)}${inf ? ` · ${inf.toFixed(0)} ms` : ''}${fps && !paused ? ` · ${fps.toFixed(1)} FPS` : ''}`;
}

function renderModel() {
  const nc = outDims ? (outDims[2] === 6 && outDims[1] <= 1000 ? NAMES.length : Math.min(outDims[1], outDims[2]) - 4) : NAMES.length;
  const hasNames = NAMES.length >= nc;
  $('modelName').textContent = `${modelLabel}${inF16 ? '（fp16）' : ''}　${nc} 類${hasNames ? '' : '（缺類別名稱，顯示為「類別 N」）'}`;
  $('resetModel').hidden = modelLabel === DEFAULT_LABEL || mode === 'age';
}

function renderMode() {
  document.querySelectorAll('#mode button').forEach(b => b.classList.toggle('on', b.dataset.m === mode));
  $('modeHint').textContent = mode === 'age'
    ? '先找出人臉，再估計歲數與性別（最多 6 張臉）。第一次會從 Hugging Face 下載約 355 MB，之後存在瀏覽器，建議用電腦。年齡平均誤差約 4.5 歲，12 歲以下較不準；人臉模型限非商業用途。拍攝他人前請先取得同意。'
    : '偵測 80 類常見物品，也可以在下方上傳自己的 .onnx 模型。';
  $('tAgeRow').hidden = $('tAge').hidden = mode !== 'age';
}

async function switchMode(m) {
  if (m === mode || modeBusy || switching || benching) return;
  modeBusy = true; await pauseLoop(); dets = [];
  mode = m; setPref('yolo-mode', mode); renderMode();
  if (mode === 'age') {
    if (!(await loadAgeMode())) { mode = 'objects'; setPref('yolo-mode', mode); renderMode(); await loadObjectsModel(); }
  } else {
    if (ageSession) { await ageSession.release(); ageSession = null; }
    ageBytes = null;
    await loadObjectsModel();
  }
  renderModel(); renderObjects();
  modeBusy = false; resumeLoop();
}

function renderObjects() {
  if (mode === 'age') {
    $('objects').innerHTML = dets.length
      ? dets.map((d, i) => `<div><span>臉 ${i + 1}</span><b>${d.age != null ? `約 ${d.age} 歲・${genderOf(d)}` : '—'}</b></div>`).join('')
      : '<div class="muted">沒有偵測到人臉</div>';
    return;
  }
  const count = {};
  for (const d of dets) count[d.cls] = (count[d.cls] || 0) + 1;
  const keys = Object.keys(count).sort((a, b) => count[b] - count[a]);
  $('objects').innerHTML = keys.length
    ? keys.map(k => `<div><span><i style="background:${color(k)}"></i>${esc(name(k))}</span><b>${count[k]}</b></div>`).join('')
    : '<div class="muted">沒有偵測到物品</div>';
}

// 每種設定只顯示最新一筆；長條越長越快，「×」= 比最慢的一種快幾倍
function renderBench() {
  const latest = new Map();
  for (const r of benchHist) if (r.model === modelLabel) latest.set(`${r.backend}|${r.threads}|${r.simd !== false}`, r);
  const rows = [...latest.values()].sort((a, b) => a.median - b.median);
  $('benchClear').hidden = !rows.length;
  if (!rows.length) { $('benchOut').innerHTML = ''; return; }
  const best = rows[0].median, worst = rows.at(-1).median;
  $('benchOut').innerHTML = rows.map(r => {
    const label = r.backend === 'webgpu' ? `WebGPU${r.gpu ? ' · ' + esc(r.gpu) : ''}` : `WebAssembly · ${r.threads} 執行緒${r.simd === false ? ' · 無 SIMD' : ''}`;
    const x = worst / r.median;
    return `<div class="bench-row ${r.backend}">
      <span class="name">${label}${rows.length > 1 && x >= 1.05 ? `<span class="x">×${x.toFixed(1)}</span>` : ''}</span>
      <span class="val">${(1000 / r.median).toFixed(1)} 次/秒 <small>${r.median.toFixed(0)} ms</small></span>
      <span class="track"><i style="width:${(best / r.median * 100).toFixed(1)}%"></i></span>
    </div>`;
  }).join('');
}
function renderBenchButtons() {
  $('benchRun').disabled = benching;
  $('benchThreads').disabled = benching || [1, 2, 4, 8].filter(n => threadChoices().includes(n)).length < 2;
  $('benchThreads').title = $('benchThreads').disabled && !benching ? '需要跨來源隔離和 2 個以上的 CPU 核心' : '';
  $('threads').disabled = benching || threadChoices().length < 2;
}

// ---- 上傳模型 ----
async function handleFiles(files) {
  let model = null, names = null;
  for (const f of files) {
    if (/\.onnx$/i.test(f.name)) model = f;
    else if (/\.(txt|json|ya?ml)$/i.test(f.name)) names = f;
    else if (/\.(pt|pth|engine|tflite)$/i.test(f.name)) { setStatus(`${f.name} 不能直接在瀏覽器跑，請先匯出成 ONNX：yolo export model=${f.name} format=onnx`, true); return; }
  }
  if (!model && !names) { setStatus('請選 .onnx 模型、.txt/.json/.yaml 類別檔，或圖片／影片', true); return; }

  const parsedNames = names ? parseNames(await names.text()) : null;
  if (model) {
    if (mode === 'age') {                        // 上傳自己的模型時回到物件偵測模式
      mode = 'objects'; setPref('yolo-mode', mode); renderMode();
      if (ageSession) { await ageSession.release(); ageSession = null; }
      ageBytes = null;
    }
    setStatus(`載入 ${model.name}…`, 'busy');
    const bytes = new Uint8Array(await model.arrayBuffer());
    if (!(await applyModel(bytes, parsedNames, model.name))) return;
    await idbSet('model', { bytes: bytes.buffer, names: NAMES, label: model.name }).catch(() => setStatus('模型太大，無法記住（重新整理後需重傳）', true));
  } else {                                       // 只換類別名稱
    NAMES = parsedNames; renderModel();
    const rec = await idbGet('model').catch(() => null);
    if (rec) await idbSet('model', { ...rec, names: NAMES });
  }
}

const isMedia = f => /^(image|video)\//.test(f.type);
function handleDrop(files) {
  const media = files.find(isMedia), rest = files.filter(f => !isMedia(f));
  if (media) openMedia(media);
  if (rest.length) handleFiles(rest);
}

$('pickModel').onclick = () => $('fileModel').click();
$('pickNames').onclick = () => $('fileNames').click();
$('pickMedia').onclick = () => $('fileMedia').click();
$('fileModel').onchange = e => { handleFiles([...e.target.files]); e.target.value = ''; };
$('fileNames').onchange = e => { handleFiles([...e.target.files]); e.target.value = ''; };
$('fileMedia').onchange = e => { if (e.target.files[0]) openMedia(e.target.files[0]); e.target.value = ''; };
$('resetModel').onclick = async () => { await idbDel('model'); await loadDefault(); };
addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('drag'); });
addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('drag'); });
addEventListener('drop', e => { e.preventDefault(); document.body.classList.remove('drag'); handleDrop([...e.dataTransfer.files]); });

$('conf').oninput = () => $('confV').textContent = (+$('conf').value * 100).toFixed(0) + '%';
$('conf').oninput();

$('startCam').onclick = async () => {
  try { await startCamera($('camera').value); }
  catch (e) { setStatus('無法開啟鏡頭：' + e.message + '（請確認已允許攝影機權限，且沒有其他程式占用）', true); }
};
$('camera').onchange = () => { if (stream) startCamera($('camera').value).catch(e => setStatus(e.message, true)); };
$('backend').onclick = e => {
  const b = e.target.closest('button'); if (!b || b.disabled || b.dataset.b === backend || switching || benching || modeBusy) return;
  createSession(b.dataset.b).then(async () => {
    if (mode !== 'age' || !ageBytes) return;
    await pauseLoop();
    try { await createAgeSession(backend); } catch (err) { setStatus('年齡模型切換失敗：' + err.message, true); }
    resumeLoop();
  });
};
$('mode').onclick = e => { const b = e.target.closest('button'); if (b) switchMode(b.dataset.m); };
$('threads').onchange = e => { setThreads(+e.target.value); location.reload(); };
$('simd').checked = SIMD;
$('simd').onchange = e => { setPref('wasm-simd', e.target.checked ? '1' : '0'); location.reload(); };
$('simdHint').textContent = SIMD
  ? `目前：ONNX Runtime ${ORT_VER}（SIMD 版）。`
  : `目前：ONNX Runtime ${ORT_VER} 無 SIMD 版。較慢，但對 iPhone 較保險；不能用 WebGPU，也不支援自動偵測輸入尺寸與 fp16 模型。`;
$('benchRun').onclick = runBenchmark;
$('benchThreads').onclick = startThreadPlan;
$('benchClear').onclick = () => { benchHist = benchHist.filter(r => r.model !== modelLabel); setPref(BENCH_KEY, JSON.stringify(benchHist)); renderBench(); };

// ============================================================
// 9. 啟動
// ============================================================
(async () => {
  if (await ensureIsolation()) return;            // GitHub Pages：註冊 service worker 後會重新整理一次
  ENV = await detectEnv();
  if (backend === 'webgpu' && (!ENV.webgpu || !SIMD)) backend = 'wasm';
  renderEnv($('env'), ENV, THREADS);
  renderThreads(); renderBackend(); renderBenchButtons();
  if (getSession(PLAN_KEY)) setStatus('比較執行緒數中，載入模型…', 'busy');

  renderMode();
  let ok = mode === 'age' ? await loadAgeMode() : false;
  if (!ok) {
    if (mode === 'age') { mode = 'objects'; setPref('yolo-mode', mode); renderMode(); }
    ok = await loadObjectsModel();
  }
  renderModel();
  if (!ok) return;
  draw(); detectLoop();
  await continueThreadPlan();
})();
