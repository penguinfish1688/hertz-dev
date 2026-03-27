#!/usr/bin/env python3
"""Local streaming-style batch inference for hertz-dev.

This script mirrors the server's chunk processing logic without starting any
websocket server/client. It recursively finds input.wav under root-dir and
writes output.wav in each sample folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as T
import torchaudio

from model import get_hertz_dev_config

TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local chunked streaming inference for rootdir/*/input.wav -> output.wav"
    )
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory to scan recursively")
    parser.add_argument("--input-name", type=str, default="input.wav", help="Input wav filename")
    parser.add_argument("--output-name", type=str, default="output.wav", help="Output wav filename")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Samples per processing chunk")
    parser.add_argument("--token-temp", type=float, default=0.8, help="LM token temperature")
    parser.add_argument("--categorical-temp", type=float, default=0.4, help="VAE categorical temperature")
    parser.add_argument("--gaussian-temp", type=float, default=0.1, help="VAE gaussian temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    return parser.parse_args()


def resolve_device(device_str: str) -> str:
    if device_str.startswith("cuda") and not T.cuda.is_available():
        print("[WARN] CUDA is unavailable, falling back to CPU")
        return "cpu"
    return device_str


def load_input_mono(audio_path: Path) -> np.ndarray:
    wav, sr = torchaudio.load(str(audio_path))
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav.squeeze(0).cpu().numpy().astype(np.float32)


def run_streaming_inference(
    model,
    samples: np.ndarray,
    chunk_size: int,
    device: str,
    temps: tuple[float, tuple[float, float]],
) -> np.ndarray:
    original_len = len(samples)
    pad = (-original_len) % chunk_size
    if pad:
        samples = np.pad(samples, (0, pad), mode="constant")

    next_model_audio = T.zeros((1, 1, chunk_size), dtype=T.float32, device=device)
    out_chunks: list[np.ndarray] = []

    with T.inference_mode():
        for start in range(0, len(samples), chunk_size):
            in_chunk = samples[start : start + chunk_size]
            in_tensor = T.from_numpy(in_chunk).to(device).reshape(1, 1, -1)

            # Match inference_server: concatenate user input with previous model output.
            model_input = T.cat([in_tensor, next_model_audio], dim=1)

            with T.autocast(device_type="cuda", dtype=T.bfloat16, enabled=device.startswith("cuda")):
                curr_model_audio = model.next_audio_from_audio(model_input, temps=temps)

            next_model_audio = curr_model_audio
            out_chunks.append(curr_model_audio.squeeze().float().cpu().numpy())

    output = np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros((0,), dtype=np.float32)
    output = output[:original_len]

    max_abs = np.max(np.abs(output)) if output.size > 0 else 0.0
    if max_abs > 1.0:
        output = output / max_abs

    return output.astype(np.float32)


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"[ERROR] Invalid root dir: {root_dir}")

    if args.chunk_size <= 0:
        raise SystemExit("[ERROR] --chunk-size must be positive")

    device = resolve_device(args.device)
    temps = (args.token_temp, (args.categorical_temp, args.gaussian_temp))

    print(f"[INIT] root_dir={root_dir}")
    print(f"[INIT] device={device}")
    print(f"[INIT] chunk_size={args.chunk_size}")

    model_config = get_hertz_dev_config(is_split=True)
    model = model_config().eval().to(device)  # type: ignore[operator]
    if device.startswith("cuda"):
        model = model.bfloat16()

    input_paths = sorted(root_dir.rglob(args.input_name))
    if not input_paths:
        print(f"[WARN] No {args.input_name} found under {root_dir}")
        return

    success = 0
    failed = 0

    for input_path in input_paths:
        rel = input_path.relative_to(root_dir)
        output_path = input_path.parent / args.output_name
        try:
            samples = load_input_mono(input_path)
            out = run_streaming_inference(
                model=model,
                samples=samples,
                chunk_size=args.chunk_size,
                device=device,
                temps=temps,
            )
            out_tensor = T.from_numpy(out).unsqueeze(0)
            torchaudio.save(str(output_path), out_tensor, TARGET_SR)
            print(f"[OK] {rel} -> {output_path.name} ({len(samples) / TARGET_SR:.2f}s)")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {rel}: {exc}")
            failed += 1

    print("\n[SUMMARY]")
    print(f"success={success} failed={failed} total={len(input_paths)}")


if __name__ == "__main__":
    main()
