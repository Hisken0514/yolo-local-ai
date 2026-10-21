# yolo-local-ai

在瀏覽器本機執行的 AI 實驗，全部運算都在你的裝置上，不會上傳影像或對話。

- `index.html`：YOLO 物件偵測。ONNX Runtime 編譯成 WebAssembly，可切換 WebGPU；顯示執行環境（SIMD、多執行緒、WebGPU）、每一幀的時間拆解，並能比較 Wasm／WebGPU 與不同執行緒數的速度。支援鏡頭、圖片、影片，以及拖入自己的 `.onnx` 模型與類別檔。
- `chat.html`：本地語言模型（Transformers.js），模型從 Hugging Face 下載。

## 多執行緒

WebAssembly 多執行緒需要頁面跨來源隔離（COOP/COEP 標頭）。GitHub Pages 不能自訂標頭，所以由 `coi-sw.js`（service worker）補上，第一次開啟頁面會自動重新整理一次。

需要 HTTPS 才能使用鏡頭，GitHub Pages 符合這個條件。
