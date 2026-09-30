"""ffmpeg/ffprobe helpers: read a clip into a [T,H,W,3] uint8 array, write one back."""
import json
import subprocess
from pathlib import Path

import numpy as np


def _encoder_args() -> list[str]:
    """NVIDIA NVENC only (no CPU fallback) - several look-videos encode in parallel, and this
    project assumes an NVIDIA GPU is present. -cq is NVENC's constant-quality mode."""
    return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "16", "-b:v", "0"]


def extract_audio(path) -> bytes | None:
    """The source's audio track as 16-bit PCM WAV bytes, or None if it has no audio stream."""
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    if not has_audio:
        return None
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-acodec", "pcm_s16le",
                        "-f", "wav", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-600:])
    return p.stdout


def probe(path) -> dict:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)])
    s = json.loads(raw)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": num / den}


def read_frames(path, start: int = 0, count: int = 0, width: int = 0) -> tuple[np.ndarray, float]:
    """Decode to RGB [T,H,W,3]. ``width`` (0 = native) rescales, keeping aspect (even height)."""
    info = probe(path)
    w = width or info["width"]
    h = round(info["height"] * w / info["width"] / 2) * 2 if width else info["height"]
    vf = [rf"select=gte(n\,{start})"] if start else []
    if width and width != info["width"]:
        vf.append(f"scale={w}:{h}:flags=area")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw, np.uint8)[: n * w * h * 3].reshape(n, h, w, 3), info["fps"]


def write_video(frames: np.ndarray, out, fps: float, audio_from=None) -> None:
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    cmd += _encoder_args() + ["-pix_fmt", "yuv420p", str(out)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames, np.uint8).tobytes(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-600:])


def iter_frames(path, chunk: int = 64, start: int = 0, count: int = 0):
    """Stream full-resolution RGB frames from ONE ffmpeg process, ``chunk`` frames at a time."""
    info = probe(path)
    w, h = info["width"], info["height"]
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if start:
        cmd += ["-vf", rf"select=gte(n\,{start})"]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size * chunk)
            n = len(buf) // size
            if n:
                yield np.frombuffer(buf, np.uint8)[: n * size].reshape(n, h, w, 3)
            if n < chunk:
                break
    finally:
        proc.stdout.close()
        proc.wait()


class VideoWriter:
    """Incremental H.264 writer (frames may arrive in chunks); optional audio passthrough."""

    def __init__(self, out, width: int, height: int, fps: float, audio_from=None):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio_from:
            cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
        cmd += _encoder_args() + ["-pix_fmt", "yuv420p", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, frames: np.ndarray) -> None:
        # One write() per FRAME, not one for the whole chunk: a single large buffer (a 64-frame
        # chunk of a wide --stack video can be ~1 GB) can make Windows' pipe write fail with
        # OSError: [Errno 22] Invalid argument, even well under any documented 2 GB/4 GB limit -
        # hit for real on a 720x1280 4-panel stack. Per-frame writes stay small (a few MB even at
        # 4K) regardless of resolution or how many panels are stacked, so this needs no tuning.
        data = np.ascontiguousarray(frames, np.uint8)
        for frame in data:
            self.proc.stdin.write(frame.tobytes())
        self.frames += len(frames)

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace")
        if self.proc.wait():
            raise RuntimeError(err[-600:])
