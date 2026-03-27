#!/usr/bin/env python3
"""Local streaming-style batch inference for hertz-dev.

This script mirrors the server's chunk processing logic without starting any
websocket server/client. It recursively finds input.wav under root-dir and
writes output.wav in each sample folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch as T
import torchaudio

from model import get_hertz_dev_config

TARGET_SR = 16000
REPLAY_SECONDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local chunked streaming inference for rootdir/*/input.wav -> output.wav"
    )
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory to scan recursively")
    parser.add_argument("--input-name", type=str, default="input.wav", help="Input wav filename")
    parser.add_argument("--output-name", type=str, default="output.wav", help="Output wav filename")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Samples per processing chunk")
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("./prompts/bob_mono.wav"),
        help="Prompt wav used to initialize streaming state",
    )
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


def to_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def from_i16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32767.0


class LocalAudioProcessor:
    def __init__(self, model, prompt_path: Path, chunk_size: int, device: str, temps: tuple[float, tuple[float, float]]):
        self.model = model
        self.prompt_path = prompt_path
        self.chunk_size = chunk_size
        self.device = device
        self.temps = temps
        self.replay_seconds = REPLAY_SECONDS
        self.loaded_audio: T.Tensor | None = None
        self.recorded_audio: T.Tensor | None = None
        self.next_model_audio: T.Tensor | None = None
        self.prompt_buffer: list[np.ndarray] = []
        self.chunks_until_live = 0
        self.initialize_state()

    def initialize_state(self) -> None:
        loaded_audio, sr = torchaudio.load(str(self.prompt_path))
        if sr != TARGET_SR:
            loaded_audio = torchaudio.transforms.Resample(sr, TARGET_SR)(loaded_audio)

        if loaded_audio.shape[0] == 1:
            loaded_audio = loaded_audio.repeat(2, 1)
        elif loaded_audio.shape[0] > 2:
            loaded_audio = loaded_audio[:2, :]

        num_chunks = loaded_audio.shape[-1] // self.chunk_size
        if num_chunks == 0:
            pad = self.chunk_size - loaded_audio.shape[-1]
            loaded_audio = T.nn.functional.pad(loaded_audio, (0, pad))
            num_chunks = 1
        else:
            loaded_audio = loaded_audio[..., : num_chunks * self.chunk_size]

        self.loaded_audio = loaded_audio.to(self.device)
        self.recorded_audio = self.loaded_audio.clone()

        cache_dtype = T.bfloat16 if self.device.startswith("cuda") else T.float32
        with T.autocast(device_type="cuda", dtype=T.bfloat16, enabled=self.device.startswith("cuda")), T.inference_mode():
            self.model.init_cache(bsize=1, device=self.device, dtype=cache_dtype, length=1024)
            self.next_model_audio = self.model.next_audio_from_audio(self.loaded_audio.unsqueeze(0), temps=self.temps)

        prompt_audio = self.loaded_audio.reshape(1, 2, -1)
        prompt_audio = prompt_audio[:, :, -(TARGET_SR * self.replay_seconds):].cpu().numpy()
        prompt_audio_mono = prompt_audio.mean(axis=1)
        self.prompt_buffer = np.array_split(prompt_audio_mono[0], int(self.replay_seconds * 8))
        self.chunks_until_live = int(self.replay_seconds * 8)

    def process_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        if self.chunks_until_live > 0:
            chunk = self.prompt_buffer[int(self.replay_seconds * 8) - self.chunks_until_live]
            self.chunks_until_live -= 1
            # Mirror server pacing during replay stage.
            time.sleep(0.05)
            return chunk.astype(np.float32)

        assert self.next_model_audio is not None
        assert self.recorded_audio is not None

        audio_tensor = T.from_numpy(audio_data).to(self.device).reshape(1, 1, -1)
        audio_tensor = T.cat([audio_tensor, self.next_model_audio], dim=1)

        with T.autocast(device_type="cuda", dtype=T.bfloat16, enabled=self.device.startswith("cuda")), T.inference_mode():
            curr_model_audio = self.model.next_audio_from_audio(audio_tensor, temps=self.temps)

        self.recorded_audio = T.cat([self.recorded_audio.cpu(), audio_tensor.squeeze(0).cpu()], dim=-1)
        self.next_model_audio = curr_model_audio
        return curr_model_audio.float().cpu().numpy().reshape(-1)

    def cleanup(self) -> None:
        self.model.deinit_cache()
        self.initialize_state()


def run_streaming_inference(
    processor: LocalAudioProcessor,
    samples: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    original_len = len(samples)
    pad = (-original_len) % chunk_size
    if pad:
        samples = np.pad(samples, (0, pad), mode="constant")

    out_chunks: list[np.ndarray] = []

    try:
        for start in range(0, len(samples), chunk_size):
            in_chunk = samples[start : start + chunk_size]
            in_chunk_i16 = to_i16(in_chunk)
            in_chunk_f32 = from_i16(in_chunk_i16)
            out_chunk_f32 = processor.process_chunk(in_chunk_f32)
            out_chunk_i16 = to_i16(out_chunk_f32)
            out_chunk_f32_roundtrip = from_i16(out_chunk_i16)
            out_chunks.append(out_chunk_f32_roundtrip.reshape(-1))
    finally:
        processor.cleanup()

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

    prompt_path = args.prompt_path.expanduser().resolve()
    if not prompt_path.exists() or not prompt_path.is_file():
        raise SystemExit(f"[ERROR] Invalid prompt path: {prompt_path}")

    device = resolve_device(args.device)
    temps = (args.token_temp, (args.categorical_temp, args.gaussian_temp))

    print(f"[INIT] root_dir={root_dir}")
    print(f"[INIT] device={device}")
    print(f"[INIT] chunk_size={args.chunk_size}")
    print(f"[INIT] prompt_path={prompt_path}")

    model_config = get_hertz_dev_config(is_split=True)
    model = model_config().eval().to(device)  # type: ignore[operator]
    if device.startswith("cuda"):
        model = model.bfloat16()

    processor = LocalAudioProcessor(
        model=model,
        prompt_path=prompt_path,
        chunk_size=args.chunk_size,
        device=device,
        temps=temps,
    )

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
                processor=processor,
                samples=samples,
                chunk_size=args.chunk_size,
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
