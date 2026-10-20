import { pipeline, TextStreamer, InterruptableStoppingCriteria } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

const $ = id => document.getElementById(id);
const setStatus = (t, err) => { $('status').textContent = t; $('status').classList.toggle('err', !!err); };

let gen = null, loading = false, busy = false;
let stopper = null;
const history = [{ role: 'system', content: '你是一個樂於助人的助理，請用繁體中文簡潔回答。' }];

// 預設模型：選了就填入下面的 ID 與量化；手機預選最小的（大模型會因記憶體不足被系統關掉分頁）
const applyPreset = () => { const [id, dt] = $('preset').value.split('|'); $('model').value = id; $('dtype').value = dt; };
const MOBILE = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
$('preset').selectedIndex = MOBILE ? 0 : 2;
$('preset').onchange = applyPreset; applyPreset();

// 手機的 WebGPU 容易在推論中途失效（device lost），手機預設改用 Wasm（CPU）
if (!navigator.gpu || MOBILE) { $('device').value = 'wasm'; $('dtype').value = 'q8'; }
if (!navigator.gpu) $('device').querySelector('[value=webgpu]').disabled = true;
$('device').onchange = () => { $('dtype').value = $('device').value === 'wasm' ? 'q8' : $('preset').value.split('|')[1]; };
$('maxTok').oninput = () => $('maxTokV').textContent = $('maxTok').value;
$('maxTok').oninput();

// ---- 載入模型 ----
$('load').onclick = async () => {
  if (loading) return;
  loading = true; gen = null;
  $('load').disabled = true; $('input').disabled = $('send').disabled = true;
  $('progress').hidden = false; $('bar').style.width = '0';
  const files = {};
  const t0 = performance.now();
  try {
    gen = await pipeline('text-generation', $('model').value.trim(), {
      device: $('device').value,
      dtype: $('dtype').value,
      progress_callback: p => {
        if (p.status === 'progress' && p.total) {
          files[p.file] = { loaded: p.loaded, total: p.total };
          const all = Object.values(files), l = all.reduce((a, f) => a + f.loaded, 0), t = all.reduce((a, f) => a + f.total, 0);
          $('bar').style.width = (l / t * 100).toFixed(1) + '%';
          setStatus(`下載中 ${(l / 1048576).toFixed(0)} / ${(t / 1048576).toFixed(0)} MB（${p.file}）`);
        } else if (p.status === 'ready') setStatus('初始化中…');
      },
    });
    setStatus(`已載入（${((performance.now() - t0) / 1000).toFixed(1)} 秒），${$('device').value} / ${$('dtype').value}`);
    $('input').disabled = $('send').disabled = false; $('input').focus();
  } catch (e) {
    setStatus('載入失敗：' + (e.message || e), true);
    console.error(e);
  }
  $('progress').hidden = true; $('load').disabled = false; loading = false;
};

// ---- 對話 ----
function addMsg(role, text) {
  const d = document.createElement('div'); d.className = 'msg ' + role; d.textContent = text;
  $('log').append(d); $('log').scrollTop = $('log').scrollHeight; return d;
}

async function send() {
  const text = $('input').value.trim();
  if (!text || !gen || busy) return;
  busy = true; $('input').value = ''; $('send').disabled = true; $('stop').hidden = false;
  addMsg('user', text); history.push({ role: 'user', content: text });
  const bubble = addMsg('bot', '…');

  let out = '', n = 0, tFirst = 0; const t0 = performance.now();
  stopper = new InterruptableStoppingCriteria();
  const streamer = new TextStreamer(gen.tokenizer, {
    skip_prompt: true, skip_special_tokens: true,
    callback_function: t => { out += t; bubble.textContent = out; $('log').scrollTop = $('log').scrollHeight; },
    token_callback_function: () => {
      if (!n++) { tFirst = performance.now(); $('ttft').textContent = (tFirst - t0).toFixed(0); }
      else $('tps').textContent = (n / ((performance.now() - tFirst) / 1000)).toFixed(1);
    },
  });
  try {
    await gen(history, { max_new_tokens: +$('maxTok').value, do_sample: false, streamer, stopping_criteria: stopper });
    history.push({ role: 'assistant', content: out });
  } catch (e) {
    const lost = /WebGPU|Instance reference|device/i.test(String(e.message || e));
    bubble.textContent = '錯誤：' + (e.message || e);
    setStatus('推論錯誤：' + (e.message || e) + (lost ? '\n→ WebGPU 已失效，請把「執行方式」改成 Wasm（CPU），重新整理頁面後再載入模型。' : ''), true);
    history.pop(); if (lost) gen = null;
  }
  busy = false; stopper = null; $('send').disabled = false; $('stop').hidden = true; $('input').focus();
}

$('form').onsubmit = e => { e.preventDefault(); send(); };
$('input').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } };
$('stop').onclick = () => stopper?.interrupt();
