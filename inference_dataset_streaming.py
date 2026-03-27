#!/usr/bin/env python3
"""Batch streaming inference via hertz websocket server.

This script recursively finds input.wav under a root directory, streams each
file to a running hertz server (/audio websocket), and saves output.wav in the
same folder.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

import numpy as np
import torch as T
import torchaudio
import websockets

TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream rootdir/*/input.wav to hertz server and save output.wav"
    )
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory to scan recursively")
    parser.add_argument("--input-name", type=str, default="input.wav", help="Input wav filename")
    parser.add_argument("--output-name", type=str, default="output.wav", help="Output wav filename")
    parser.add_argument("--server-url", type=str, default="ws://localhost:8000/audio", help="Websocket endpoint")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Samples per chunk (hertz default: 2000)")
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="Recv timeout per chunk")
    return parser.parse_args()


def load_and_preprocess_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sr = torchaudio.load(str(audio_path))

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio = resampler(audio)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    mono = audio.squeeze(0).cpu().numpy().astype(np.float32)
    return mono, mono.shape[0]


async def stream_one_file(
    server_url: str,
    samples: np.ndarray,
    chunk_size: int,
    timeout_sec: float,
) -> np.ndarray:
    pad = (-len(samples)) % chunk_size
    if pad:
        samples = np.pad(samples, (0, pad), mode="constant")

    outputs: list[np.ndarray] = []

    async with websockets.connect(server_url, max_size=None) as ws:
        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            chunk_i16 = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)

            payload = base64.b64encode(chunk_i16.tobytes()).decode("utf-8")
            await ws.send(f"data:audio/raw;base64,{payload}")

            response = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
            if isinstance(response, bytes):
                response = response.decode("utf-8")
            if "," not in response:
                raise RuntimeError("Unexpected server response format")

            b64 = response.split(",", 1)[1]
            out_i16 = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
            out_f32 = (out_i16.astype(np.float32) / 32767.0).reshape(-1)
            outputs.append(out_f32)

    if not outputs:
        return np.zeros((0,), dtype=np.float32)

    return np.concatenate(outputs, axis=0)


async def run_dataset(args: argparse.Namespace) -> int:
    root_dir = args.root_dir.expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"[ERROR] Invalid root dir: {root_dir}")

    input_paths = sorted(root_dir.rglob(args.input_name))
    if not input_paths:
        print(f"[WARN] No {args.input_name} found under {root_dir}")
        return 0

    print(f"[INIT] root_dir={root_dir}")
    print(f"[INIT] server_url={args.server_url}")
    print(f"[INIT] files={len(input_paths)}")

    success = 0
    failed = 0

    for input_path in input_paths:
        rel = input_path.relative_to(root_dir)
        output_path = input_path.parent / args.output_name

        try:
            samples, original_len = load_and_preprocess_audio(input_path)
            streamed = await stream_one_file(
                server_url=args.server_url,
                samples=samples,
                chunk_size=args.chunk_size,
                timeout_sec=args.timeout_sec,
            )
            streamed = streamed[:original_len]
            output = T.from_numpy(np.expand_dims(streamed, axis=0))
            torchaudio.save(str(output_path), output, TARGET_SR)
            print(f"[OK] {rel} -> {output_path.name} ({original_len / TARGET_SR:.2f}s)")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {rel}: {exc}")
            failed += 1

    print("\n[SUMMARY]")
    print(f"success={success} failed={failed} total={len(input_paths)}")
    return 0 if failed == 0 else 1


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run_dataset(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
