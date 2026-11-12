import { ensureIsolation, getPref, setPref, autoThreads } from './wasm-env.js';

// ============================================================
// VR 透視模式（WebXR immersive-ar），按扳機才辨識（像辨識照片一樣，不做即時偵測，省電）：
//   右扳機 → 物件偵測（YOLO26n）；左扳機 → 年齡・性別（人臉偵測 + 年齡模型）
//   按下時拍一張頭盔鏡頭畫面 → 推論 → 框畫在透明 canvas → 貼在「按下當時」眼前 D 公尺的平面上。
//   平面固定在空間中（不跟著頭），所以轉頭後框仍留在原物體附近。握把鍵清除結果。
// 鏡頭和眼睛的位置、視角不完全相同，框只會大致對準，可用搖桿調大小與高低。
// ============================================================
const NAMES = ['人','腳踏車','汽車','機車','飛機','公車','火車','卡車','船','紅綠燈','消防栓','停止標誌','停車計時器','長椅',
  '鳥','貓','狗','馬','羊','牛','大象','熊','斑馬','長頸鹿','背包','雨傘','手提包','領帶','行李箱','飛盤','滑雪板','滑雪單板',
  '球','風箏','棒球棒','棒球手套','滑板','衝浪板','網球拍','瓶子','酒杯','杯子','叉子','刀子','湯匙','碗','香蕉','蘋果',
  '三明治','柳橙','花椰菜','紅蘿蔔','熱狗','披薩','甜甜圈','蛋糕','椅子','沙發','盆栽','床','餐桌','馬桶','電視','筆電',
  '滑鼠','遙控器','鍵盤','手機','微波爐','烤箱','烤麵包機','水槽','冰箱','書','時鐘','花瓶','剪刀','泰迪熊','吹風機','牙刷'];
const FACE_URL = 'https://huggingface.co/AdamCodd/yolo11n-face-age/resolve/main/model.onnx';                // CC-BY-NC-4.0
const AGE_URL = 'https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main/onnx/model.onnx'; // Apache-2.0
const SIZE = 640, CONF = 0.4, FACE_CONF = 0.25, D = 1.5;  // 模型輸入邊長、信心門檻（遠處的臉分數偏低，門檻放寬）、平面距離（公尺）
const AGE_SIZE = 224, AGE_MAX_FACES = 6, MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const color = i => `hsl(${(i * 47) % 360} 85% 58%)`;
const $ = id => document.getElementById(id);
const mb = n => (n / 1048576).toFixed(0) + ' MB';
let note = '';                                        // 顯示在網頁與 VR 說明列的狀態
let vw = 0, vh = 0;                                   // 鏡頭畫面尺寸
// VR 平面寬度 = 2·D·tan(視角/2)。視角預設 80°（約略是頭盔鏡頭的水平視角，實際要用搖桿校正），會記住。
let fov = +getPref('xr-fov') || 80, yOff = +getPref('xr-yoff') || 0;
const setStatus = t => { note = t; $('status').textContent = t; drawOverlay(); };

if (await ensureIsolation()) throw 0;                 // GitHub Pages：註冊 service worker 後重新整理一次

// ---- 1. 環境檢查 ----
const xrOK = !!navigator.xr && await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
const checks = [
  [isSecureContext, '安全環境（HTTPS 或 localhost）', '要用 HTTPS 或 localhost 開啟'],
  [xrOK, 'WebXR 透視（immersive-ar）', '這個瀏覽器不支援；在電腦上只能用下方按鈕預覽'],
  [!!navigator.mediaDevices?.getUserMedia, '鏡頭存取', '不支援 getUserMedia'],
  [!!navigator.gpu, 'WebGPU', '沒有，改用 WebAssembly'],
  [self.crossOriginIsolated, `WebAssembly 多執行緒（${autoThreads} 個）`, '沒有跨來源隔離，只能單執行緒'],
];
$('checks').innerHTML = checks.map(([ok, a, b]) => `<div><b>${ok ? '✅' : '⚠️'}</b>${a}${ok ? '' : `：${b}`}</div>`).join('');

// ---- 2. ONNX Runtime 與模型（WebGPU 優先，失敗退回 Wasm） ----
const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
await new Promise((res, rej) => {
  const s = document.createElement('script');
  s.src = ORT_BASE + 'ort.webgpu.min.js'; s.crossOrigin = 'anonymous'; s.onload = res; s.onerror = rej;
  document.head.append(s);
});
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = self.crossOriginIsolated ? autoThreads : 1;
const want = new URLSearchParams(location.search).get('b');        // ?b=wasm 可強制用 CPU
const order = (want === 'wasm' || !navigator.gpu) ? ['wasm'] : ['webgpu', 'wasm'];
async function createSession(bytes) {
  for (const b of order) {
    try { const s = await ort.InferenceSession.create(bytes, { executionProviders: [b] }); s.backend = b; return s; }
    catch (e) { console.warn(b, e); }
  }
  throw new Error('無法建立推論環境');
}
const labelOf = s => s.backend === 'webgpu' ? 'WebGPU' : `Wasm ×${ort.env.wasm.numThreads}`;

async function cachedFetch(url, what) {               // 與主頁共用同一個 Cache Storage，主頁下載過就不用再下載
  let cache = null;
  try { cache = await caches.open('hf-models-v1'); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch {}
  const res = await fetch(url);
  if (!res.ok) throw new Error(`下載失敗（HTTP ${res.status}）`);
  const total = +res.headers.get('content-length') || 0, reader = res.body.getReader(), chunks = [];
  let got = 0, shown = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length;
    if (got - shown > 4 << 20) { shown = got; setStatus(`下載${what} ${mb(got)}${total ? ' / ' + mb(total) : ''}`); }
  }
  const bytes = new Uint8Array(got); let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  try { await cache?.put(url, new Response(bytes)); } catch {}
  return bytes;
}

setStatus('載入物件模型中…');
const objSession = await createSession(new Uint8Array(await (await fetch('model.onnx')).arrayBuffer()));
let faceSession = null, ageSession = null, ageLoading = null;
function loadAge() {                                  // 年齡模型 345 MB，第一次按左扳機才下載
  return ageLoading ??= (async () => {
    faceSession = await createSession(await cachedFetch(FACE_URL, '人臉模型'));
    const ageBytes = await cachedFetch(AGE_URL, '年齡模型');
    setStatus('建立年齡模型推論環境中…');
    ageSession = await createSession(ageBytes);
  })().catch(e => { ageLoading = null; throw e; });
}
setStatus(`物件模型就緒（${labelOf(objSession)}）。先開啟鏡頭。`);

// ---- 3. 前處理 / 後處理（YOLO [1, 4+類別數, 8400] 格式） ----
const pcv = new OffscreenCanvas(SIZE, SIZE), pctx = pcv.getContext('2d', { willReadFrequently: true });
const input = new Float32Array(3 * SIZE * SIZE);
function preprocess(el, vw, vh) {
  const scale = Math.min(SIZE / vw, SIZE / vh), w = Math.round(vw * scale), h = Math.round(vh * scale);
  const dx = (SIZE - w) / 2, dy = (SIZE - h) / 2;
  pctx.fillStyle = 'rgb(114,114,114)'; pctx.fillRect(0, 0, SIZE, SIZE);
  pctx.drawImage(el, dx, dy, w, h);
  const px = pctx.getImageData(0, 0, SIZE, SIZE).data, area = SIZE * SIZE;
  for (let i = 0; i < area; i++) {
    input[i] = px[i * 4] / 255; input[i + area] = px[i * 4 + 1] / 255; input[i + area * 2] = px[i * 4 + 2] / 255;
  }
  return { scale, dx, dy };
}
// agnostic：不分類別一起做 NMS（人臉模型的 3 個類別都當成「臉」）
function postprocess(out, dims, { scale, dx, dy }, agnostic) {
  const C = dims[1], N = dims[2], boxes = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = -1;
    for (let c = 4; c < C; c++) { const s = out[c * N + i]; if (s > best) { best = s; cls = c - 4; } }
    if (best < (agnostic ? FACE_CONF : CONF)) continue;
    const cx = out[i], cy = out[N + i], w = out[2 * N + i], h = out[3 * N + i];
    boxes.push({ cls: agnostic ? 0 : cls, score: best, x1: (cx - w / 2 - dx) / scale, y1: (cy - h / 2 - dy) / scale, x2: (cx + w / 2 - dx) / scale, y2: (cy + h / 2 - dy) / scale });
  }
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (keep.length < 50 && keep.every(k => k.cls !== b.cls || iou(k, b) < 0.45)) keep.push(b);
  return keep;
}
function iou(a, b) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)), h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return w * h / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - w * h);
}
async function yolo(session, el, vw, vh, agnostic) {
  const meta = preprocess(el, vw, vh);
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  const out = res[session.outputNames[0]];
  return postprocess(await out.getData(), out.dims, meta, agnostic);
}
// 每張臉往外多取 20% 再縮放成 224×224，正規化後一次批次推論（與主頁相同）
const acv = new OffscreenCanvas(AGE_SIZE, AGE_SIZE), actx = acv.getContext('2d', { willReadFrequently: true });
async function runAge(el, vw, vh, faces) {
  const list = faces.slice(0, AGE_MAX_FACES).filter(d => d.x2 > d.x1 && d.y2 > d.y1);
  if (!list.length) return [];
  const n = list.length, area = AGE_SIZE * AGE_SIZE, data = new Float32Array(n * 3 * area);
  list.forEach((d, k) => {
    const m = 0.2 * Math.max(d.x2 - d.x1, d.y2 - d.y1);
    const x1 = Math.max(0, d.x1 - m), y1 = Math.max(0, d.y1 - m), x2 = Math.min(vw, d.x2 + m), y2 = Math.min(vh, d.y2 + m);
    actx.drawImage(el, x1, y1, x2 - x1, y2 - y1, 0, 0, AGE_SIZE, AGE_SIZE);
    const px = actx.getImageData(0, 0, AGE_SIZE, AGE_SIZE).data, off = k * 3 * area;
    for (let i = 0; i < area; i++)
      for (let c = 0; c < 3; c++) data[off + c * area + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  });
  const out = (await ageSession.run({ pixel_values: new ort.Tensor('float32', data, [n, 3, AGE_SIZE, AGE_SIZE]) })).logits.data;
  list.forEach((d, k) => { d.age = Math.min(100, Math.max(0, Math.round(out[k * 2]))); d.female = out[k * 2 + 1]; });
  return list;
}

// ---- 4. 鏡頭 ----
// Quest 3 的瀏覽器把透視鏡頭當成一般鏡頭提供（需要新版 Horizon OS，第一次會詢問權限）。
const video = document.createElement('video');
video.muted = true; video.playsInline = true;
$('cam').onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } } });
    video.srcObject = stream; await video.play();
    vw = video.videoWidth; vh = video.videoHeight;
    setStatus(`鏡頭就緒（${vw}×${vh}）。按「辨識物件」或「辨識年齡」，VR 中按右／左扳機。`);
    $('cam').disabled = true; $('enter').disabled = !xrOK; $('detObj').disabled = $('detAge').disabled = false;
    drawPreview();
  } catch (e) { setStatus('無法開啟鏡頭：' + e.message); }
};

// ---- 5. 按一次辨識一次：拍下當下畫面 → 推論 → 更新結果 ----
const shot = document.createElement('canvas');        // 按下時的畫面，預覽用
let dets = [], mode = '', ms = 0, busy = false, showHelp = true;
let capturePose = null;                               // 按下當時頭的位置（空間座標），平面固定在這裡
async function detect(kind, pose = null) {
  if (busy || !vw) return;
  busy = true;
  try {
    shot.width = vw; shot.height = vh; shot.getContext('2d').drawImage(video, 0, 0);
    if (kind === 'age' && !ageSession) { setStatus('第一次使用，下載年齡模型（約 355 MB）…'); await loadAge(); }
    setStatus(kind === 'age' ? '辨識年齡中…' : '辨識物件中…');
    const t = performance.now();
    if (kind === 'age') dets = await runAge(shot, vw, vh, await yolo(faceSession, shot, vw, vh, true));
    else dets = await yolo(objSession, shot, vw, vh, false);
    ms = performance.now() - t; mode = kind; capturePose = pose;
    setStatus(`${kind === 'age' ? '年齡' : '物件'}：${dets.length} 個，${ms.toFixed(0)} ms（${labelOf(kind === 'age' ? ageSession : objSession)}）`);
    drawPreview();
  } catch (e) { setStatus('辨識失敗：' + e.message); }
  busy = false;
}
function clearDets() { dets = []; mode = ''; capturePose = null; setStatus('已清除。按扳機再辨識。'); drawPreview(); }
$('detObj').onclick = () => detect('objects');
$('detAge').onclick = () => detect('age');

const labelText = d => mode === 'age' ? `${d.female >= 0.5 ? '女' : '男'} ${d.age} 歲` : `${NAMES[d.cls] ?? d.cls} ${Math.round(d.score * 100)}%`;
const boxColor = d => mode === 'age' ? (d.female >= 0.5 ? '#ff6fae' : '#45b6ff') : color(d.cls);
function drawBoxes(ctx, W, sx, sy) {
  ctx.lineWidth = Math.max(3, W / 250);
  ctx.font = `bold ${Math.round(W / 32)}px system-ui, sans-serif`;
  for (const d of dets) {
    const x = d.x1 * sx, y = d.y1 * sy, w = (d.x2 - d.x1) * sx, h = (d.y2 - d.y1) * sy;
    ctx.strokeStyle = ctx.fillStyle = boxColor(d);
    ctx.strokeRect(x, y, w, h);
    const t = labelText(d), tw = ctx.measureText(t).width, th = W / 26;
    ctx.fillRect(x, y - th, tw + 12, th);
    ctx.fillStyle = '#000'; ctx.fillText(t, x + 6, y - th * 0.2);
  }
}
// VR 用：透明 canvas，只有框與說明列
const overlay = document.createElement('canvas'), octx = overlay.getContext('2d');
let overlayDirty = true;
function drawOverlay() {
  if (!vw) return;
  const W = 1024, H = Math.round(W * vh / vw);
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  octx.clearRect(0, 0, W, H);
  drawBoxes(octx, W, W / vw, H / vh);
  const lines = [note];
  if (showHelp) lines.push(`右扳機：物件　左扳機：年齡　握把：清除　右搖桿：大小 ${fov.toFixed(0)}°　左搖桿：高低　A：隱藏說明`);
  const lh = W / 30;
  octx.font = `${Math.round(W / 42)}px system-ui, sans-serif`;
  octx.fillStyle = 'rgba(0,0,0,.6)'; octx.fillRect(0, H - lh * lines.length - 8, W, lh * lines.length + 8);
  octx.fillStyle = '#fff';
  lines.forEach((l, i) => octx.fillText(l, 12, H - lh * (lines.length - i - 1) - 12));
  overlayDirty = true;
}
// 網頁預覽：按下時的畫面 + 框
const pv = $('preview'), pvctx = pv.getContext('2d');
function drawPreview() {
  if (!vw) return;
  if (pv.width !== vw) { pv.width = vw; pv.height = vh; }
  pvctx.drawImage(mode ? shot : video, 0, 0);
  drawBoxes(pvctx, vw, 1, 1);
}

// ---- 6. WebXR ----
$('enter').onclick = async () => {
  try {
    const xs = await navigator.xr.requestSession('immersive-ar', { optionalFeatures: ['local'] });
    runXR(xs);
  } catch (e) { setStatus('無法進入 VR：' + e.message); }
};

const mul = (a, b) => {                               // 4×4 矩陣相乘（column-major）
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s;
  }
  return o;
};
const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

async function runXR(xs) {
  const glc = document.createElement('canvas');
  const gl = glc.getContext('webgl2', { xrCompatible: true, alpha: true });
  xs.updateRenderState({ baseLayer: new XRWebGLLayer(xs, gl) });
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, `#version 300 es
    in vec2 p; uniform mat4 mvp; uniform vec3 size; out vec2 uv;
    void main(){ uv = vec2(p.x*.5+.5, .5-p.y*.5); gl_Position = mvp * vec4(p.x*size.x, p.y*size.y + size.z, -${D.toFixed(2)}, 1.); }`));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, `#version 300 es
    precision mediump float; in vec2 uv; uniform sampler2D tex; out vec4 o;
    void main(){ o = texture(tex, uv); }`));
  gl.linkProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p');
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);

  // 空間座標：有 local 就把結果固定在空間中；沒有就退回跟著頭
  const space = await xs.requestReferenceSpace('local').catch(() => xs.requestReferenceSpace('viewer'));
  xs.addEventListener('select', e => {                 // 扳機：右 = 物件、左 = 年齡
    const pose = e.frame.getViewerPose(space);
    detect(e.inputSource.handedness === 'left' ? 'age' : 'objects', pose && Float32Array.from(pose.transform.matrix));
  });
  xs.addEventListener('squeeze', clearDets);           // 握把：清除
  xs.addEventListener('end', () => { setPref('xr-fov', fov); setPref('xr-yoff', yOff); setStatus('已離開 VR。'); });
  setStatus('右扳機辨識物件、左扳機辨識年齡。');

  let last = 0, aWas = false;
  xs.requestAnimationFrame(function frame(t, f) {
    xs.requestAnimationFrame(frame);
    const dt = last ? Math.min(0.1, (t - last) / 1000) : 0; last = t;
    let aNow = false;
    for (const src of xs.inputSources) {               // 搖桿：右手大小、左手高低；右手 A 鍵切換說明
      const gp = src.gamepad; if (!gp) continue;
      const y = gp.axes.length >= 4 && Math.abs(gp.axes[3]) > 0.2 ? gp.axes[3] : 0;
      if (src.handedness === 'right') { fov = Math.min(150, Math.max(30, fov - y * 40 * dt)); aNow ||= !!gp.buttons[4]?.pressed; }
      if (src.handedness === 'left') yOff = Math.min(0.8, Math.max(-0.8, yOff - y * 0.4 * dt));
      if (y) drawOverlay();
    }
    if (aNow && !aWas) { showHelp = !showHelp; drawOverlay(); }
    aWas = aNow;

    const pose = f.getViewerPose(space); if (!pose) return;
    const layer = xs.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!overlay.width) return;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (overlayDirty) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, overlay); overlayDirty = false; }
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const halfW = D * Math.tan(fov * Math.PI / 360), halfH = halfW * overlay.height / overlay.width;
    gl.uniform3f(gl.getUniformLocation(prog, 'size'), halfW, halfH, yOff);
    for (const view of pose.views) {
      const vp = layer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      // 有結果：平面放在按下當時頭的位置（固定在空間）；沒有結果：只顯示說明列，跟著頭
      const model = capturePose ? mul(view.transform.inverse.matrix, capturePose) : IDENTITY;
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'mvp'), false, mul(view.projectionMatrix, model));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  });
}
