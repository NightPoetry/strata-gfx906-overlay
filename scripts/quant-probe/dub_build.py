#!/usr/bin/env python3
# dub_build.py — F5-TTS 克隆音色 → 按场景起点铺 narration 轨 → 与原 BGM 混合
# 用法: python dub_build.py <v11|v4>   （需 .venv-f5tts；HF_ENDPOINT 走 hf-mirror）
import json, os, subprocess, sys, wave
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REF_WAV = "/tmp/dubwork/ref.wav"
REF_TEXT = "为什么呢？因为 V 一百降得实在太狠了，没办法，那只能退而求其次。我们用千问三点六三十五 B A 三 B。"
CFG = json.load(open(os.path.join(BASE, "scenes.json"), encoding="utf-8"))
which = sys.argv[1] if len(sys.argv) > 1 else "v4"
cfg = CFG[which]
SR = 24000
OUT = os.path.join(BASE, f"dub-{which}")
os.makedirs(OUT, exist_ok=True)

def read_wav(path):
    w = wave.open(path, "rb")
    assert w.getframerate() == SR and w.getnchannels() == 1, path
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767

def main():
    from f5_tts.api import F5TTS
    tts = F5TTS()
    track = np.zeros(int(cfg["dur"] * SR), dtype=np.float32)
    for sc in cfg["scenes"]:
        out = os.path.join(OUT, f"scene_{int(sc['t']*10):03d}.wav")
        if not os.path.exists(out):
            print("TTS", sc["t"], sc["text"][:18], "…")
            wav, sr, _ = tts.infer(REF_WAV, REF_TEXT, sc["text"],
                                   speed=1.05, cross_fade_duration=0.15, seed=42,
                                   file_wave=out)
        seg = read_wav(out)
        at = int(sc["t"] * SR)
        end = min(len(track), at + len(seg))
        # 6ms 淡入淡出防爆音
        fade = int(0.006 * SR)
        if end - at > 2*fade:
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[end-at-fade:end-at] *= np.linspace(1, 0, fade)
        track[at:end] += seg[:end-at]
    # 轻限幅
    peak = np.max(np.abs(track)) or 1.0
    if peak > 0.97: track *= 0.97 / peak
    pcm = (np.clip(track, -1, 1) * 32767).astype(np.int16)
    narr = os.path.join(OUT, f"narration_{which}.wav")
    with wave.open(narr, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print("NARR", narr)

    video = os.path.join(BASE, cfg["video"])
    dubbed = os.path.join(BASE, cfg["video"].replace(".mp4", "-dubbed.mp4"))
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", narr,
           "-filter_complex",
           f"[0:a]volume=0.32[bg];[bg][1:a]amix=inputs=2:duration=first:normalize=0[a]",
           "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", dubbed]
    subprocess.run(cmd, check=True)
    print("DUBBED", dubbed)

if __name__ == "__main__":
    main()
