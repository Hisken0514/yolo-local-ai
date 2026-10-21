// 跨來源隔離 service worker
// GitHub Pages 不能自訂 HTTP 標頭，這裡替同網域的回應補上 COOP/COEP，
// 頁面就會變成 crossOriginIsolated，WebAssembly 才能用 SharedArrayBuffer 開多執行緒。
// 跨網域的請求（CDN、Hugging Face）不經過這裡，由瀏覽器照 CORS 規則處理。
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', e => {
  const req = e.request;
  const url = new URL(req.url);
  if (url.origin !== location.origin || url.pathname.endsWith('/__reload')) return;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;

  e.respondWith(fetch(req).then(res => {
    if (res.status === 0) return res;              // opaque / 轉址：原樣交回
    const headers = new Headers(res.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Resource-Policy', 'same-origin');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
  }));
});
