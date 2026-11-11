import { ensureIsolation, getPref, setPref, autoThreads } from './wasm-env.js';

// ============================================================
// VR 透視模式（WebXR immersive-ar）：
//   頭盔鏡頭（getUserMedia）→ YOLO 推論 → 把框畫在透明 canvas →
//   當成貼圖貼在「跟著頭」的平面上（距離眼睛 D 公尺），疊在透視畫面上。
// 鏡頭和眼睛位置、視角不完全相同，所以框只會大致對準，可用搖桿調大小與高低。
// ============================================================
const NAMES = ['人','腳踏車','汽車','機車','飛機','公車','火車','卡車','船','紅綠燈','消防栓','停止標誌','停車計時器','長椅',
  '鳥','貓','狗','馬','羊','牛','大象','熊','斑馬','長頸鹿','背包','雨傘','手提包','領帶','行李箱','飛盤','滑雪板','滑雪單板',
  '球','風箏','棒球棒','棒球手套','滑板','衝浪板','網球拍','瓶子','酒杯','杯子','叉子','刀子','湯匙','碗','香蕉','蘋果',
  '三明治','柳橙','花椰菜','紅蘿蔔','熱狗','披薩','甜甜圈','蛋糕','椅子','沙發','盆栽','床','餐桌','馬桶','電視','筆電',
  '滑鼠','遙控器','鍵盤','手機','微波爐','烤箱','烤麵包機','水槽','冰箱','書','時鐘','花瓶','剪刀','泰迪熊','吹風機','牙刷'];
const SIZE = 640, CONF = 0.4, D = 1.5;                // 模型輸入邊長、信心門檻、平面距離（公尺）
const color = i => `hsl(${(i * 47) % 360} 85% 58%)`;
const $ = id => document.getElementById(id);
const setStatus = t => { $('status').textContent = t; };

if (await ensureIsolation()) throw 0;                 // GitHub Pages：註冊 service worker 後重新整理一次

// ---- 1. 環境檢查 ----
const xrOK = !!navigator.xr && await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
const checks = [
  [isSecureContext, '安全環境（HTTPS 或 localhost）', '要用 HTTPS 或 localhost 開啟'],
  [xrOK, 'WebXR 透視（immersive-ar）', '這個瀏覽器不支援；在電腦上只能預覽辨識'],
  [!!navigator.mediaDevices?.getUserMedia, '鏡頭存取', '不支援 getUserMedia'],
  [!!navigator.gpu, 'WebGPU', '沒有，改用 WebAssembly'],
  [self.crossOriginIsolated, `WebAssembly 多執行緒（${autoThreads} 個）`, '沒有跨來源隔離，只能單執行緒'],
];
$('checks').innerHTML = checks.map(([ok, a, b]) => `<div><b>${ok ? '✅' : '⚠️'}</b>${a}${ok ? '' : `：${b}`}</div>`).join('');

// ---- 2. 載入模型（WebGPU 優先，失敗退回 Wasm） ----
const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
await new Promise((res, rej) => {
  const s = document.createElement('script');
  s.src = ORT_BASE + 'ort.webgpu.min.js'; s.crossOrigin = 'anonymous'; s.onload = res; s.onerror = rej;
  document.head.append(s);
});
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = self.crossOriginIsolated ? autoThreads : 1;

setStatus('載入模型中…');
const modelBytes = new Uint8Array(await (await fetch('model.onnx')).arrayBuffer());
const want = new URLSearchParams(location.search).get('b');        // ?b=wasm 可強制用 CPU
let session, backend;
for (const b of (want === 'wasm' || !navigator.gpu) ? ['wasm'] : ['webgpu', 'wasm']) {
  try { session = await ort.InferenceSession.create(modelBytes, { executionProviders: [b] }); backend = b; break; }
  catch (e) { console.warn(b, e); }
}
const backendLabel = backend === 'webgpu' ? 'WebGPU' : `Wasm ×${ort.env.wasm.numThreads}`;
setStatus(`模型就緒（${backendLabel}）。先開啟鏡頭。`);

// ---- 3. 前處理 / 後處理（與主頁相同的 YOLO26 [1, 84, 8400] 格式） ----
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
function postprocess(out, dims, { scale, dx, dy }) {
  const C = dims[1], N = dims[2], boxes = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = -1;
    for (let c = 4; c < C; c++) { const s = out[c * N + i]; if (s > best) { best = s; cls = c - 4; } }
    if (best < CONF) continue;
    const cx = out[i], cy = out[N + i], w = out[2 * N + i], h = out[3 * N + i];
    boxes.push({ cls, score: best, x1: (cx - w / 2 - dx) / scale, y1: (cy - h / 2 - dy) / scale, x2: (cx + w / 2 - dx) / scale, y2: (cy + h / 2 - dy) / scale });
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

// ---- 4. 鏡頭 ----
// Quest 3 的瀏覽器把兩顆透視鏡頭當成一般鏡頭提供（需要新版 Horizon OS，第一次會詢問權限）。
const video = document.createElement('video');
video.muted = true; video.playsInline = true;
let vw = 0, vh = 0;
$('cam').onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } } });
    video.srcObject = stream; await video.play();
    vw = video.videoWidth; vh = video.videoHeight;
    const label = stream.getVideoTracks()[0].label;
    setStatus(`鏡頭：${label || '未命名'}（${vw}×${vh}）· ${backendLabel}`);
    $('cam').disabled = true; $('enter').disabled = !xrOK;
    detectLoop();
  } catch (e) { setStatus('無法開啟鏡頭：' + e.message); }
};

// ---- 5. 偵測迴圈：推論完一次才做下一次；結果畫到透明的 overlay canvas ----
const overlay = document.createElement('canvas');
const octx = overlay.getContext('2d');
let dets = [], infMs = 0, fps = 0, overlayDirty = true, showHelp = true;
async function detectLoop() {
  let frames = 0, t0 = performance.now();
  for (;;) {
    if (video.readyState < 2) { await new Promise(r => setTimeout(r, 50)); continue; }
    const meta = preprocess(video, vw, vh);
    const t = performance.now();
    const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
    infMs = performance.now() - t;
    const out = res[session.outputNames[0]];
    dets = postprocess(await out.getData(), out.dims, meta);
    frames++;
    if (performance.now() - t0 > 1000) { fps = frames * 1000 / (performance.now() - t0); frames = 0; t0 = performance.now(); }
    drawOverlay(); drawPreview();
    await new Promise(r => setTimeout(r, 0));
  }
}
function drawBoxes(ctx, W, H, sx, sy) {
  ctx.lineWidth = Math.max(3, W / 250);
  ctx.font = `bold ${Math.round(W / 32)}px system-ui, sans-serif`;
  for (const d of dets) {
    const x = d.x1 * sx, y = d.y1 * sy, w = (d.x2 - d.x1) * sx, h = (d.y2 - d.y1) * sy;
    ctx.strokeStyle = ctx.fillStyle = color(d.cls);
    ctx.strokeRect(x, y, w, h);
    const t = `${NAMES[d.cls] ?? d.cls} ${Math.round(d.score * 100)}%`, tw = ctx.measureText(t).width, th = W / 26;
    ctx.fillRect(x, y - th, tw + 12, th);
    ctx.fillStyle = '#000'; ctx.fillText(t, x + 6, y - th * 0.2);
  }
}
function drawOverlay() {
  const W = 1024, H = Math.round(W * vh / vw);
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  octx.clearRect(0, 0, W, H);
  drawBoxes(octx, W, H, W / vw, H / vh);
  octx.font = `${Math.round(W / 40)}px system-ui, sans-serif`;
  octx.fillStyle = 'rgba(0,0,0,.55)'; octx.fillRect(0, H - W / 22, W, W / 22);
  octx.fillStyle = '#fff';
  const help = showHelp ? `　右搖桿：大小 ${fov.toFixed(0)}°　左搖桿：高低　扳機：隱藏說明` : '';
  octx.fillText(`${backendLabel}　推論 ${infMs.toFixed(0)} ms　${fps.toFixed(1)} FPS　${dets.length} 個${help}`, 12, H - W / 80);
  overlayDirty = true;
}
const pv = $('preview'), pvctx = pv.getContext('2d');
function drawPreview() {
  if (pv.width !== vw) { pv.width = vw; pv.height = vh; }
  pvctx.drawImage(video, 0, 0);
  drawBoxes(pvctx, vw, vh, 1, 1);
}

// ---- 6. WebXR：跟著頭的平面 ----
// 平面寬度 = 2·D·tan(視角/2)。視角預設 80°（約略是頭盔鏡頭的水平視角，實際要用搖桿校正），會記住。
let fov = +getPref('xr-fov') || 80, yOff = +getPref('xr-yoff') || 0;
$('enter').onclick = async () => {
  try {
    const xs = await navigator.xr.requestSession('immersive-ar', { optionalFeatures: ['local'] });
    runXR(xs);
  } catch (e) { setStatus('無法進入 VR：' + e.message); }
};

function runXR(xs) {
  const glc = document.createElement('canvas');
  const gl = glc.getContext('webgl2', { xrCompatible: true, alpha: true });
  xs.updateRenderState({ baseLayer: new XRWebGLLayer(xs, gl) });
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, `#version 300 es
    in vec2 p; uniform mat4 proj; uniform vec3 size; out vec2 uv;
    void main(){ uv = vec2(p.x*.5+.5, .5-p.y*.5); gl_Position = proj * vec4(p.x*size.x, p.y*size.y + size.z, -${D.toFixed(2)}, 1.); }`));
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
  overlayDirty = true;

  xs.addEventListener('select', () => { showHelp = !showHelp; drawOverlay(); });
  xs.addEventListener('end', () => { setPref('xr-fov', fov); setPref('xr-yoff', yOff); setStatus('已離開 VR。'); });

  let last = 0;
  xs.requestReferenceSpace('viewer').then(space => xs.requestAnimationFrame(function frame(t, f) {
    xs.requestAnimationFrame(frame);
    const dt = last ? Math.min(0.1, (t - last) / 1000) : 0; last = t;
    for (const src of xs.inputSources) {                    // 搖桿調整：右手大小、左手高低
      const ax = src.gamepad?.axes; if (!ax || ax.length < 4) continue;
      const y = Math.abs(ax[3]) > 0.2 ? ax[3] : 0;
      if (src.handedness === 'right') fov = Math.min(150, Math.max(30, fov - y * 40 * dt));
      if (src.handedness === 'left') yOff = Math.min(0.8, Math.max(-0.8, yOff - y * 0.4 * dt));
      if (y && showHelp) drawOverlay();
    }
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
    for (const view of pose.views) {                        // 只用投影矩陣、不乘視角矩陣 → 平面固定在眼前
      const vp = layer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'proj'), false, view.projectionMatrix);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }));
}
