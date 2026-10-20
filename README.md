# yolo-local-ai

在瀏覽器本機執行的 AI 實驗，全部運算都在你的裝置上，不會上傳影像或對話。

- `index.html`：YOLO 鏡頭辨識（ONNX Runtime Web，WebGPU / Wasm），可拖入自己的 `.onnx` 模型與類別檔。
- `chat.html`：本地語言模型（Transformers.js），模型從 Hugging Face 下載。

需要 HTTPS 才能使用鏡頭，GitHub Pages 符合這個條件。
