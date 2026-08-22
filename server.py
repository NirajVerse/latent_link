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
from PIL import Image
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

_current_transfer = {"total": 0, "received": set()}


def get_model():
    global _model
    if _model is None:
        _model = get_vq_model()
    return _model


class DecodeRequest(BaseModel):
    data: str  # base64 of the compressed, bit-packed indices


class SnapshotRequest(BaseModel):
    image: str  # data URL (base64) of the captured camera frame


class ProgressRequest(BaseModel):
    index: int


def tensor_to_png_base64(tensor, size=None):
    """Denormalize a VQ-VAE output tensor and return it as base64 PNG.
    Optionally resizes back to the original (width, height)."""
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    image = ToPILImage()(tensor.squeeze(0))
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
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
        img_tensor, (enc_w, enc_h), orig_size = prepare_image(tmp_path)
        indices = encode_to_integers(model, img_tensor)
        packed = transport.pack_latents(
            indices, enc_w // 4, enc_h // 4, original_size=orig_size
        )
        frames = transport.split_frames(packed, config.QR_CHUNK_SIZE)
        _current_transfer["total"] = len(frames)
        _current_transfer["received"] = set()
        return {
            "width": enc_w,
            "height": enc_h,
            "original_width": orig_size[0],
            "original_height": orig_size[1],
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
        meta = transport.unpack_latents(packed)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid data: {e}")

    model = get_model()
    idx_tensor = torch.tensor(meta["indices"], dtype=torch.long)
    grid_w, grid_h = meta["grid"]
    tensor = decode_from_integers(model, idx_tensor, grid_w, grid_h)
    return {"image": tensor_to_png_base64(tensor, size=meta["original"])}


@app.get("/")
def root():
    return RedirectResponse(url="/static/sender.html")


@app.post("/progress")
def report_progress(req: ProgressRequest):
    """Receiver reports a frame index it has successfully captured."""
    if 0 <= req.index < _current_transfer["total"]:
        _current_transfer["received"].add(req.index)
    return {
        "total": _current_transfer["total"],
        "received": sorted(_current_transfer["received"]),
    }


@app.get("/progress")
def get_progress():
    """Sender polls which frame indices still need to be shown."""
    return {
        "total": _current_transfer["total"],
        "received": sorted(_current_transfer["received"]),
    }


@app.post("/debug/snapshot")
def snapshot(req: SnapshotRequest):
    """Save a camera frame from the receiver for debugging."""
    import pathlib

    payload = req.image.split(",", 1)[-1]
    data = base64.b64decode(payload)
    path = pathlib.Path("/tmp/opencode/snapshot.jpg")
    path.write_bytes(data)
    return {"saved": str(path), "bytes": len(data)}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
