"""
Serializes VQ-VAE latent codes into a compact byte payload and splits that
payload into self-describing QR frames (with headers and CRC32 checksums).

The idea: an image is encoded into a list of integer codebook indices
(0..8191). Each index fits in 13 bits. We bit-pack those indices into bytes,
zlib-compress them, then split the compressed blob into chunks small enough
to fit inside a single QR code. Each chunk gets a header so the receiver can
reassemble the original blob even when frames arrive out of order.
"""
import struct
import zlib

MAGIC = b"LKQ1"
VERSION = 1
HEADER_LEN = 17  # magic(4) + version(1) + total(2) + index(2) + len(4) + crc(4)

BITS_PER_CODE = 13  # codebook has 8192 entries -> log2(8192) = 13

# Max payload bytes per QR frame. QR v40 at EC level L holds ~2953 bytes of
# binary data; we stay comfortably under that so scans stay reliable.
DEFAULT_CHUNK_SIZE = 2800


def _pack_bits(values, bits):
    """Pack a list of integers into bytes using `bits` bits each (MSB first)."""
    out = bytearray()
    acc = 0
    acc_bits = 0
    mask = (1 << bits) - 1
    for v in values:
        acc = (acc << bits) | (int(v) & mask)
        acc_bits += bits
        while acc_bits >= 8:
            acc_bits -= 8
            out.append((acc >> acc_bits) & 0xFF)
        acc &= (1 << acc_bits) - 1 if acc_bits > 0 else 0
    if acc_bits > 0:
        out.append((acc << (8 - acc_bits)) & 0xFF)
    return bytes(out)


def _unpack_bits(data, bits, count):
    """Unpack `count` integers of `bits` bits each from a byte string."""
    values = []
    acc = 0
    acc_bits = 0
    mask = (1 << bits) - 1
    for byte in data:
        acc = (acc << 8) | byte
        acc_bits += 8
        while acc_bits >= bits:
            acc_bits -= bits
            values.append((acc >> acc_bits) & mask)
            acc &= (1 << acc_bits) - 1 if acc_bits > 0 else 0
            if len(values) == count:
                return values
    return values


def pack_indices(indices):
    """Compress a list/tensor of codebook indices into a byte string.

    Layout before zlib: [bits_per_code u8][grid_dim u16][bit-packed indices]
    """
    if hasattr(indices, "tolist"):
        values = indices.tolist()
    else:
        values = list(indices)

    n = len(values)
    grid = int(n ** 0.5)
    if grid * grid != n:
        raise ValueError("index count must be a perfect square")

    body = bytes([BITS_PER_CODE]) + struct.pack(">H", grid) + _pack_bits(values, BITS_PER_CODE)
    return zlib.compress(body, 9)


def unpack_indices(data):
    """Decompress bytes produced by pack_indices back into a list of indices."""
    body = zlib.decompress(data)
    bits = body[0]
    grid = struct.unpack(">H", body[1:3])[0]
    count = grid * grid
    return _unpack_bits(body[3:], bits, count)


def build_frame(payload, frame_index, total_frames):
    """Wrap a payload chunk into a self-describing frame with a CRC32."""
    header = (
        MAGIC
        + bytes([VERSION])
        + struct.pack(">HHI", total_frames, frame_index, len(payload))
    )
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return header + struct.pack(">I", crc) + payload


def parse_frame(frame):
    """Parse a frame into a dict; raises ValueError on corruption."""
    if len(frame) < HEADER_LEN:
        raise ValueError("frame too short")
    if frame[0:4] != MAGIC:
        raise ValueError("bad magic bytes")
    version = frame[4]
    total, index, length = struct.unpack(">HHI", frame[5:13])
    crc = struct.unpack(">I", frame[13:17])[0]
    payload = frame[17:17 + length]
    if zlib.crc32(payload) & 0xFFFFFFFF != crc:
        raise ValueError(f"crc mismatch on frame {index}")
    return {"version": version, "total": total, "index": index, "payload": payload}


def split_frames(data, chunk_size=DEFAULT_CHUNK_SIZE):
    """Split a byte string into a list of QR-ready frames."""
    total = max(1, (len(data) + chunk_size - 1) // chunk_size)
    frames = []
    for i in range(total):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        frames.append(build_frame(chunk, i, total))
    return frames


def join_frames(frame_bytes_list):
    """Reassemble the original byte string from a list of frame byte strings."""
    parsed = [parse_frame(f) for f in frame_bytes_list]
    parsed.sort(key=lambda p: p["index"])
    return b"".join(p["payload"] for p in parsed)


if __name__ == "__main__":
    import random

    # Round-trip the index packing.
    grid = 16
    indices = [random.randrange(1 << BITS_PER_CODE) for _ in range(grid * grid)]
    packed = pack_indices(indices)
    unpacked = unpack_indices(packed)
    assert unpacked == indices, "index round-trip failed"
    print(f"packed {len(indices)} indices -> {len(packed)} bytes")

    # Round-trip frame split/join, including out-of-order frames.
    frames = split_frames(packed)
    print(f"split into {len(frames)} frames")
    shuffled = frames[1:] + frames[:1]
    rejoined = join_frames(shuffled)
    assert rejoined == packed, "frame round-trip failed"
    assert unpack_indices(rejoined) == indices, "rejoined round-trip failed"
    print("transport self-test passed")
