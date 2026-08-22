# LatentLink

Transfer images between two phones using nothing but **QR codes and a VQ-VAE** — no internet, no Bluetooth, no NFC. An image is compressed into discrete latent codes by a vector-quantized autoencoder, the codes are streamed as a sequence of QR codes on one phone, and a second phone scans them and reconstructs the original image on-device via a shared backend.

## Why

QR codes max out at ~3 KB of payload, while a single 512x512 RGB image is 768 KB of raw pixels. Raw pixels will never fit. LatentLink closes the gap by transmitting the *essence* of the image instead:

| Stage | Size |
|---|---|
| 512x512 RGB image | 768 KB |
| VQ-VAE latent grid (128x128 codes x 13 bits) | 26.6 KB |
| After zlib compression of the code stream | ~4 KB |
| Split into QR frames | ~27 frames |

The result is roughly **190x compression** end-to-end, with reconstructions measuring **~27.4 dB PSNR** — faithful, if slightly soft compared to the original.

## How it works

```
Phone A (sender)                Laptop (backend)               Phone B (receiver)
-----------------               ----------------               ------------------
pick image  ────────────────►   VQ-VAE encode
                                fit to 512 px, keep aspect
                                (e.g. 512x384 ─► 128x96 grid)
                                bit-pack 13-bit codes + zlib
                                frame + CRC32 + base64
QR sequence ◄────────────────   frame list + fps
(cycles MISSING frames ◄───┐    )
                           │    POST /progress ────────────  html5-qrcode scan
                           └───────────────────────────────  decode header+CRC,
                                                             dedupe frames
                                                            when complete:
                             VQ-VAE decode ◄───────────────  POST /decode
                             pixels ─► PNG                 show/save image
```

1. **Encode** — the backend runs the image through a pre-trained VQ-VAE (`CompVis/ldm-super-resolution-4x-openimages`, `vqvae` subfolder), producing a grid of indices into an 8192-entry codebook (e.g. a landscape photo becomes a 128x96 grid). The image is scaled so its longest side is 512 px with the aspect ratio preserved, and the original dimensions travel inside the payload.
2. **Pack** — indices are written as 13-bit values with grid dimensions and original width/height in a small header, zlib-compressed, and split into ~1000-byte chunks.
3. **Frame** — each chunk gets a binary header: magic `LKQ1`, version, total-frame count, frame index, payload length, CRC32 checksum.
4. **Transmit** — each framed chunk is base64-encoded into a QR code and displayed fullscreen at ~3 fps.
5. **Receive** — the receiver scans continuously, validates each frame's CRC32, keeps unique frame indices, and relays progress to the backend.
6. **Feedback loop** — the sender polls `/progress` every 700 ms and *only cycles frames the receiver hasn't confirmed yet*, so stragglers get retried automatically instead of restarting the whole sequence.
7. **Reconstruct** — once all frames arrive, the receiver reassembles the payload, sends it to `/decode`, and displays the reconstructed image — resized back to its original dimensions — with a save button.

## Repository layout

```
server.py            FastAPI app: /encode, /decode, /progress endpoints
transport.py         Bit-packing, zlib, framing (LKQ1 header + CRC32)
encoder.py           Image -> latent code indices
decoder.py           Latent code indices -> image
model_loader.py      Loads the pre-trained VQModel from local HF cache
config.py            Model ID, image size, QR fps, chunk size
serve.sh             HTTPS server launcher (auto-generates self-signed cert)
static/
  sender.html/js     Sender UI: renders QR frames, polls progress
  receiver.html/js   Receiver UI: camera scanner, progress bar, result view
  vendor/            Vendored qrcode.js + html5-qrcode.min.js (ZXing-based)
requirement.txt      Python dependencies
```

## Getting started

Requirements: Python 3.10+, a Linux laptop and two phones on the same Wi-Fi network.

```bash
python -m venv .venv
.venv/bin/pip install -r requirement.txt

# One-time: fetch the VQ-VAE weights into the local Hugging Face cache
# (the server loads with local_files_only=True)
.venv/bin/python -c "from diffusers import VQModel; \
  VQModel.from_pretrained('CompVis/ldm-super-resolution-4x-openimages', subfolder='vqvae')"

./serve.sh
```

`serve.sh` detects the machine's LAN IP, generates a self-signed TLS certificate on first run (`ssl/`, gitignored), and prints the URLs.

> Camera access requires a secure context, which is why the server runs over HTTPS. On first visit each phone will show a certificate warning — accept it to proceed.

## Usage

**Phone A — sender:** open `/static/sender.html`, pick an image. QR codes start cycling immediately and skip any frame already received.

**Phone B — receiver:** open `/static/receiver.html`, tap *Start Camera*, point at the sender. Move back until the entire QR fits inside the guide box. The progress bar fills as frames land; when complete, the reconstructed image appears with a *Save Image* option. Tap *Scan Another* to receive again.

A typical transfer takes ~10–15 seconds including retries.

## Configuration

Tune in `config.py`:

- `QR_FPS` — sender frame rate (higher = faster transfer, harder to scan)
- `QR_CHUNK_SIZE` — bytes per frame (larger = fewer frames, denser QRs)
- `IMAGE_SIZE` — longest-side resolution; the latent grid is that size divided by 4, preserving aspect ratio

## Limitations & future ideas

- Reconstruction is faithful but soft (~27 dB PSNR); a higher-capacity VAE would sharpen it at the cost of more codes.
- Requires line-of-sight and a steady hand; scanning degrades under screen moiré at extreme distances.
- No encryption — anyone photographing the QR stream can reconstruct the image. An interesting extension would be encrypting the code stream so the key lives only on the two phones.
