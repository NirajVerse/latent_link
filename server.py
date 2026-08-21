"""
FastAPI backend for the LatentLink QR image-transfer demo.

Sender phone:   uploads an image to /encode, receives QR frames to display.
Receiver phone: scans QR frames, reassembles them, POSTs to /decode and gets
                the reconstructed image back.

The VQ-VAE encoder/decoder run here on the laptop; the phones only handle
QR display and camera scanning.
"""
import base64
import io
import os
import tempfile

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torchvision.transforms import ToPILImage

import config
import transport
from decoder import decode_from_integers
from encoder import encode_to_integers, prepare_image
from model_loader import get_vq_model

app = FastAPI(title="LatentLink")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = get_vq_model()
    return _model


class DecodeRequest(BaseModel):
    data: str  # base64 of the compressed, bit-packed indices


def tensor_to_png_base64(tensor):
    """Denormalize a VQ-VAE output tensor and return it as base64 PNG."""
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    image = ToPILImage()(tensor.squeeze(0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/encode")
def encode(file: UploadFile = File(...)):
    """Encode an uploaded image into QR-ready frames."""
    contents = file.file.read()
    suffix = os.path.splitext(file.filename or "img")[1] or ".webp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        model = get_model()
        img_tensor = prepare_image(tmp_path)
        indices = encode_to_integers(model, img_tensor)
        packed = transport.pack_indices(indices)
        frames = transport.split_frames(packed, config.QR_CHUNK_SIZE)
        return {
            "width": config.IMAGE_SIZE,
            "height": config.IMAGE_SIZE,
            "num_codes": int(indices.numel()),
            "total_frames": len(frames),
            "fps": config.QR_FPS,
            "frames": [base64.b64encode(f).decode("ascii") for f in frames],
        }
    finally:
        os.unlink(tmp_path)


@app.post("/decode")
def decode(req: DecodeRequest):
    """Reconstruct an image from compressed, bit-packed indices."""
    try:
        packed = base64.b64decode(req.data)
        indices = transport.unpack_indices(packed)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid data: {e}")

    model = get_model()
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    tensor = decode_from_integers(model, idx_tensor)
    return {"image": tensor_to_png_base64(tensor)}


@app.get("/")
def root():
    return RedirectResponse(url="/static/sender.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
