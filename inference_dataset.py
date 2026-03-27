#!/usr/bin/env python3
"""Batch offline inference for hertz-dev datasets.

Given a root directory, this script recursively finds files named input.wav,
runs hertz-dev completion on each one, and writes output.wav in the same folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch as T
import torchaudio

from model import get_hertz_dev_config
from tokenizer import make_tokenizer

TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for rootdir/*/input.wav -> output.wav")
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory to scan recursively")
    parser.add_argument("--input-name", type=str, default="input.wav", help="Input wav filename")
    parser.add_argument("--output-name", type=str, default="output.wav", help="Output wav filename")
    parser.add_argument("--two-speaker", action="store_true", help="Use two-speaker split model")
    parser.add_argument("--use-pure-audio-ablation", action="store_true", help="Use pure-audio ablation checkpoint")
    parser.add_argument("--prompt-seconds", type=float, default=3.0, help="Prompt duration used for conditioning")
    parser.add_argument("--gen-seconds", type=float, default=20.0, help="Generated continuation duration")
    parser.add_argument("--token-temp", type=float, default=0.8, help="LM token temperature")
    parser.add_argument("--categorical-temp", type=float, default=0.5, help="Resynthesizer categorical temperature")
    parser.add_argument("--gaussian-temp", type=float, default=0.1, help="Resynthesizer gaussian temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on, e.g. cuda or cpu")
    return parser.parse_args()


def resolve_device(device_str: str) -> str:
    if device_str.startswith("cuda") and not T.cuda.is_available():
        print("[WARN] CUDA is unavailable, falling back to CPU")
        return "cpu"
    return device_str


def load_and_preprocess_audio(audio_path: Path, two_speaker: bool) -> T.Tensor:
    audio, sr = torchaudio.load(str(audio_path))

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio = resampler(audio)

    if two_speaker:
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
    else:
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.shape[0] > 2:
            audio = audio[:1, :]

    # Shape: [batch, channels, samples]
    return audio.unsqueeze(0)


def encode_prompt(audio: T.Tensor, audio_tokenizer, two_speaker: bool, device: str) -> T.Tensor:
    with T.autocast(device_type="cuda", dtype=T.bfloat16, enabled=device.startswith("cuda")):
        if two_speaker:
            ch1 = audio_tokenizer.latent_from_data(audio[:, 0:1].to(device))
            ch2 = audio_tokenizer.latent_from_data(audio[:, 1:2].to(device))
            return T.cat([ch1, ch2], dim=-1)
        return audio_tokenizer.latent_from_data(audio.to(device))


def generate_completion(
    encoded_prompt: T.Tensor,
    prompt_chunks: int,
    gen_chunks: int,
    generator,
    audio_tokenizer,
    two_speaker: bool,
    device: str,
    temps: tuple[float, tuple[float, float]],
) -> T.Tensor:
    prompt_chunks = max(1, min(prompt_chunks, encoded_prompt.shape[1]))
    prompt_slice = encoded_prompt[:, :prompt_chunks]

    with T.autocast(device_type="cuda", dtype=T.bfloat16, enabled=device.startswith("cuda")):
        completed = generator.completion(prompt_slice, temps=temps, use_cache=True, gen_len=gen_chunks)
        if two_speaker:
            decoded_ch1 = audio_tokenizer.data_from_latent(completed[:, :, :32].bfloat16())
            decoded_ch2 = audio_tokenizer.data_from_latent(completed[:, :, 32:].bfloat16())
            decoded = T.cat([decoded_ch1, decoded_ch2], dim=0)
        else:
            decoded = audio_tokenizer.data_from_latent(completed.bfloat16())

    wav = decoded.cpu().float()
    if wav.ndim == 3:
        wav = wav.squeeze(0)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    max_abs = wav.abs().max()
    if max_abs > 1:
        wav = wav / max_abs

    # Keep continuation-focused segment, same strategy as the notebook.
    start = max(prompt_chunks * 2000 - TARGET_SR, 0)
    return wav[:, start:]


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"[ERROR] Invalid root dir: {root_dir}")

    device = resolve_device(args.device)
    print(f"[INIT] root_dir={root_dir}")
    print(f"[INIT] device={device}")

    if args.two_speaker and args.use_pure_audio_ablation:
        raise SystemExit("[ERROR] --two-speaker and --use-pure-audio-ablation cannot be enabled together")

    model_config = get_hertz_dev_config(
        is_split=args.two_speaker,
        use_pure_audio_ablation=args.use_pure_audio_ablation,
    )
    generator = model_config().eval().to(device)  # type: ignore[operator]
    if device.startswith("cuda"):
        generator = generator.to(T.bfloat16)

    audio_tokenizer = make_tokenizer(device=device if device.startswith("cuda") else "cpu")

    input_paths = sorted(root_dir.rglob(args.input_name))
    if not input_paths:
        print(f"[WARN] No {args.input_name} found under {root_dir}")
        return

    prompt_chunks = int(args.prompt_seconds * 8)
    gen_chunks = int(args.gen_seconds * 8)
    temps = (args.token_temp, (args.categorical_temp, args.gaussian_temp))

    success = 0
    failed = 0

    for input_path in input_paths:
        output_path = input_path.parent / args.output_name
        rel = input_path.relative_to(root_dir)

        try:
            audio = load_and_preprocess_audio(input_path, two_speaker=args.two_speaker)
            encoded = encode_prompt(audio, audio_tokenizer, args.two_speaker, device)
            output_wav = generate_completion(
                encoded_prompt=encoded,
                prompt_chunks=prompt_chunks,
                gen_chunks=gen_chunks,
                generator=generator,
                audio_tokenizer=audio_tokenizer,
                two_speaker=args.two_speaker,
                device=device,
                temps=temps,
            )
            torchaudio.save(str(output_path), output_wav, TARGET_SR)
            print(f"[OK] {rel} -> {output_path.name}")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {rel}: {exc}")
            failed += 1

    print("\n[SUMMARY]")
    print(f"success={success} failed={failed} total={len(input_paths)}")


if __name__ == "__main__":
    main()
