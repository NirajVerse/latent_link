const startBtn = document.getElementById("start");
const errorEl = document.getElementById("error");
const snapBtn = document.getElementById("snap");
const progressWrap = document.getElementById("progressWrap");
const fill = document.getElementById("fill");
const progressEl = document.getElementById("progress");
const debugEl = document.getElementById("debug");
const result = document.getElementById("result");
const outImg = document.getElementById("outImg");
const againBtn = document.getElementById("again");

const MAGIC = [0x4c, 0x4b, 0x51, 0x31];
const received = new Map();
let totalFrames = null;
let scanning = false;
let ticks = 0;
let hits = 0;
let stored = 0;
let rejected = 0;

const scanner = new Html5Qrcode("reader");

const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? (0xedb88320 ^ (c >>> 1)) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = crcTable[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function parseFrame(bytes) {
  if (bytes.length < 17) return null;
  for (let i = 0; i < 4; i++) if (bytes[i] !== MAGIC[i]) return null;
  const total = (bytes[5] << 8) | bytes[6];
  const index = (bytes[7] << 8) | bytes[8];
  const length = ((bytes[9] << 24) | (bytes[10] << 16) | (bytes[11] << 8) | bytes[12]) >>> 0;
  const crc = ((bytes[13] << 24) | (bytes[14] << 16) | (bytes[15] << 8) | bytes[16]) >>> 0;
  if (length > bytes.length - 17) return null;
  const payload = bytes.subarray(17, 17 + length);
  if (crc32(payload) !== crc) return null;
  return { total, index, payload };
}

function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function bytesToBase64(bytes) {
  const parts = [];
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    parts.push(String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK)));
  }
  return btoa(parts.join(""));
}

function updateProgress() {
  const got = received.size;
  fill.style.width = ((got / totalFrames) * 100) + "%";
  progressEl.textContent = "Received " + got + " / " + totalFrames + " frames";
}

function handleFrame(bytes) {
  const parsed = parseFrame(bytes);
  if (!parsed) return false;
  if (totalFrames === null || parsed.total !== totalFrames) {
    received.clear();
    totalFrames = parsed.total;
  }
  if (received.has(parsed.index)) return false;
  received.set(parsed.index, parsed.payload);
  updateProgress();
  if (received.size === totalFrames) finish();
  return true;
}

function onDecode(text) {
  ticks++;
  hits++;
  let bytes;
  try {
    bytes = base64ToBytes(text);
  } catch (e) {
    rejected++;
    return;
  }
  if (handleFrame(bytes)) stored++;
  else rejected++;
  if (ticks % 5 === 0) {
    debugEl.textContent =
      "qr hits " + hits + " | stored " + stored + " | rejected " + rejected;
  }
}

async function finish() {
  scanning = false;
  await scanner.stop();
  progressEl.textContent = "Reconstructing image...";
  const parts = [];
  for (let i = 0; i < totalFrames; i++) parts.push(received.get(i));
  const totalLen = parts.reduce((a, p) => a + p.length, 0);
  const blob = new Uint8Array(totalLen);
  let off = 0;
  for (const p of parts) { blob.set(p, off); off += p.length; }

  try {
    const res = await fetch("/decode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: bytesToBase64(blob) }),
    });
    const data = await res.json();
    outImg.src = "data:image/png;base64," + data.image;
    result.style.display = "flex";
    progressWrap.style.display = "none";
  } catch (e) {
    progressEl.textContent = "Decode failed, try again.";
    result.style.display = "none";
    progressWrap.style.display = "flex";
  }
}

function reset() {
  received.clear();
  totalFrames = null;
  ticks = 0;
  hits = 0;
  stored = 0;
  rejected = 0;
  result.style.display = "none";
  progressWrap.style.display = "flex";
  debugEl.textContent = "";
  updateProgress();
}

async function startCamera() {
  errorEl.style.display = "none";
  reset();
  try {
    await scanner.start(
      { facingMode: "environment" },
      { fps: 10, qrbox: { width: 260, height: 260 }, disableFlip: true },
      onDecode,
      () => {}
    );
    progressWrap.style.display = "flex";
    startBtn.disabled = true;
    scanning = true;
  } catch (e) {
    errorEl.textContent = "Camera error: " + e.message;
    errorEl.style.display = "block";
  }
}

snapBtn.addEventListener("click", () => {
  const v = document.querySelector("#reader video");
  if (!v || !v.videoWidth) return;
  const c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  const scale = Math.min(1, 640 / c.width);
  const dc = document.createElement("canvas");
  dc.width = Math.round(c.width * scale);
  dc.height = Math.round(c.height * scale);
  dc.getContext("2d").drawImage(c, 0, 0, dc.width, dc.height);
  const dataUrl = dc.toDataURL("image/jpeg", 0.85);
  fetch("/debug/snapshot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  });
});

startBtn.addEventListener("click", startCamera);
againBtn.addEventListener("click", () => {
  reset();
  startCamera();
});
