const fileInput = document.getElementById("file");
const qrWrap = document.getElementById("qrWrap");
const canvas = document.getElementById("qr");
const statusEl = document.getElementById("status");

let frames = [];
let fps = 3;
let current = 0;
let timer = null;

function frameToModules(text) {
  const qr = qrcode(0, "L");
  qr.addData(text, "Byte");
  qr.make();
  const count = qr.getModuleCount();
  const modules = new Array(count);
  for (let r = 0; r < count; r++) {
    const row = new Array(count);
    for (let c = 0; c < count; c++) row[c] = qr.isDark(r, c) ? 1 : 0;
    modules[r] = row;
  }
  return { count, modules };
}

function drawFrame(i) {
  const { count, modules } = frames[i];
  const scale = Math.max(3, Math.floor(720 / count));
  const margin = 4 * scale;
  const size = count * scale + margin * 2;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = "#000";
  for (let r = 0; r < count; r++) {
    for (let c = 0; c < count; c++) {
      if (modules[r][c]) ctx.fillRect(margin + c * scale, margin + r * scale, scale, scale);
    }
  }
  statusEl.textContent = "Frame " + (i + 1) + " / " + frames.length;
}

function startLoop() {
  clearInterval(timer);
  current = 0;
  drawFrame(current);
  timer = setInterval(() => {
    current = (current + 1) % frames.length;
    drawFrame(current);
  }, 1000 / fps);
}

async function handleFile(file) {
  statusEl.textContent = "Preparing...";
  qrWrap.style.display = "flex";
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/encode", { method: "POST", body: form });
  if (!res.ok) {
    statusEl.textContent = "Encode failed: " + res.status;
    return;
  }
  const data = await res.json();
  fps = data.fps || 1;
  frames = data.frames.map((b64) => frameToModules(b64));
  startLoop();
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});
