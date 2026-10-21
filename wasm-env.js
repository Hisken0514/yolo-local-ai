// ============================================================
// 兩個頁面共用：WebAssembly 執行環境偵測、跨來源隔離、執行緒設定
// ============================================================
export const cores = navigator.hardwareConcurrency || 1;

export function getPref(k) { try { return localStorage.getItem(k); } catch { return null; } }
export function setPref(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, String(v)); } catch {} }
export function getSession(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } }
export function setSession(k, v) { try { v == null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, JSON.stringify(v)); } catch {} }

// ---- 跨來源隔離 ----
// 有隔離標頭（本機伺服器）就直接是隔離狀態；沒有（GitHub Pages）就註冊 coi-sw.js 再重新整理一次。
// 回傳 true 代表頁面即將重新整理，呼叫端應停止後續初始化。
export async function ensureIsolation() {
  if (self.crossOriginIsolated || !isSecureContext || !('serviceWorker' in navigator)) return false;
  if (getSession('coi-reload')) return false;      // 重新整理過仍未隔離：放棄，改用單執行緒
  try {
    await navigator.serviceWorker.register(new URL('coi-sw.js', import.meta.url));
    await navigator.serviceWorker.ready;
    setSession('coi-reload', 1);
    location.reload();
    return true;
  } catch { return false; }
}

// ---- 執行緒數 ----
// 執行緒數只能在 ONNX Runtime 第一次初始化前設定，所以變更後要重新整理頁面。
const THREADS_KEY = 'wasm-threads';
export const autoThreads = Math.min(4, Math.max(1, Math.ceil(cores / 2)));   // ONNX Runtime 的預設算法
export function threadChoices() {
  if (!self.crossOriginIsolated) return [1];
  return [...new Set([1, 2, 4, 8, 16, autoThreads].filter(n => n <= cores))].sort((a, b) => a - b);
}
export function threads() {
  const n = +getPref(THREADS_KEY);
  return threadChoices().includes(n) ? n : self.crossOriginIsolated ? autoThreads : 1;
}
export function setThreads(n) { setPref(THREADS_KEY, n); }

// ---- 功能偵測 ----
// SIMD 偵測用的最小 wasm 模組（含一個 v128 指令），出自 wasm-feature-detect
const SIMD_PROBE = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]);

export async function detectEnv() {
  const wasm = typeof WebAssembly === 'object';
  const info = {
    wasm,
    simd: wasm && WebAssembly.validate(SIMD_PROBE),
    isolated: !!self.crossOriginIsolated,
    cores,
    webgpu: false,
    gpu: '',
  };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) {
        info.webgpu = true;
        const i = a.info || {};
        info.gpu = [i.vendor, i.architecture].filter(Boolean).join(' ') || i.description || '';
      }
    } catch {}
  }
  return info;
}

// 目前頁面實際下載的 .wasm 檔（由 Resource Timing 取得大小與下載時間）
export function wasmFiles() {
  return performance.getEntriesByType('resource')
    .filter(e => /\.wasm(\?|$)/.test(e.name))
    .map(e => ({
      name: decodeURIComponent(e.name.split('?')[0].split('/').pop()),
      bytes: e.encodedBodySize || e.transferSize,
      raw: e.decodedBodySize,
      ms: e.duration,
    }));
}

export const mb = b => (b / 1048576).toFixed(b >= 10485760 ? 0 : 1) + ' MB';

// ---- 環境面板 ----
export function renderEnv(el, info, n) {
  const chip = (ok, text, tip) => `<span class="chip ${ok ? 'ok' : 'no'}" title="${tip}"><i></i>${text}</span>`;
  const threadText = info.isolated ? `多執行緒 · ${n} / ${info.cores} 核` : '多執行緒（未開啟）';
  const threadTip = info.isolated
    ? `頁面已跨來源隔離，可用 SharedArrayBuffer，Wasm 把運算分給 ${n} 個執行緒`
    : '頁面沒有跨來源隔離，瀏覽器不開放 SharedArrayBuffer，Wasm 只能用 1 個執行緒';
  el.querySelector('.chips').innerHTML = [
    chip(info.wasm, 'WebAssembly', '瀏覽器能執行 .wasm：ONNX Runtime 的 C++ 程式編譯成 Wasm 後在這裡跑'),
    chip(info.simd, 'SIMD', '128 位元向量指令：一個指令同時算 4 個浮點數'),
    chip(info.isolated && n > 1, threadText, threadTip),
    chip(info.webgpu, info.webgpu ? `WebGPU${info.gpu ? ' · ' + info.gpu : ''}` : 'WebGPU（不支援）', '用顯示卡運算；不支援時只能用 Wasm'),
  ].join('');
}

export function renderWasmFiles(el) {
  const f = wasmFiles().at(-1);
  el.textContent = f
    ? `${f.name}　下載 ${f.bytes ? mb(f.bytes) : '—'}${f.raw ? `（解壓後 ${mb(f.raw)}）` : ''}・${f.ms.toFixed(0)} ms`
    : '尚未載入 .wasm';
}
