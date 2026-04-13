#!/usr/bin/env python3
"""
Apply a simple mixing + mastering chain with selectable modes (EQ, compression, limiting, loudness).
Reads WAV/FLAC/etc. via soundfile; writes WAV.

Requires: soundfile, numpy, pedalboard, pyloudnorm.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    PeakFilter,
)

Mode = Literal["clean", "warm", "bright", "club", "radio"]


@dataclass(frozen=True)
class ModeProfile:
    """Targets and effect settings for one mastering mode."""

    target_lufs: float
    highpass_hz: float
    lowpass_hz: float | None
    peak_db_at_hz: tuple[float, float, float]  # (freq_hz, gain_db, q)
    compressor_threshold_db: float
    compressor_ratio: float
    makeup_db: float


PROFILES: dict[Mode, ModeProfile] = {
    "clean": ModeProfile(
        target_lufs=-14.0,
        highpass_hz=30.0,
        lowpass_hz=18000.0,
        peak_db_at_hz=(0.0, 0.0, 1.0),
        compressor_threshold_db=-18.0,
        compressor_ratio=2.5,
        makeup_db=0.0,
    ),
    "warm": ModeProfile(
        target_lufs=-13.5,
        highpass_hz=35.0,
        lowpass_hz=16500.0,
        peak_db_at_hz=(220.0, 1.8, 0.7),
        compressor_threshold_db=-20.0,
        compressor_ratio=2.2,
        makeup_db=0.5,
    ),
    "bright": ModeProfile(
        target_lufs=-13.0,
        highpass_hz=40.0,
        lowpass_hz=19000.0,
        peak_db_at_hz=(6500.0, 1.2, 0.8),
        compressor_threshold_db=-17.0,
        compressor_ratio=3.0,
        makeup_db=0.0,
    ),
    "club": ModeProfile(
        target_lufs=-11.0,
        highpass_hz=25.0,
        lowpass_hz=17000.0,
        peak_db_at_hz=(90.0, 2.0, 0.6),
        compressor_threshold_db=-16.0,
        compressor_ratio=3.5,
        makeup_db=1.5,
    ),
    "radio": ModeProfile(
        target_lufs=-9.0,
        highpass_hz=40.0,
        lowpass_hz=15500.0,
        peak_db_at_hz=(3000.0, 0.8, 0.9),
        compressor_threshold_db=-14.0,
        compressor_ratio=4.0,
        makeup_db=2.0,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply mixing/mastering chain with a chosen mode.",
    )
    p.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Input audio file (e.g. assets/track.wav)",
    )
    p.add_argument(
        "--mode",
        choices=list(PROFILES.keys()),
        default="clean",
        help="Processing character / loudness profile (default: clean)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path. Default: <stem>_mastered_<mode>.wav beside input",
    )
    p.add_argument(
        "--peak",
        type=float,
        default=-1.0,
        help="True peak ceiling in dBFS after limiting (default: -1.0)",
    )
    return p.parse_args()


def ensure_float_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    return audio


def apply_lufs_match(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
    except ValueError:
        # Too few samples for BS.1770 metering; skip loudness match
        return audio
    if not np.isfinite(loudness):
        return audio
    gain_lin = 10.0 ** ((target_lufs - loudness) / 20.0)
    return np.clip(audio * gain_lin, -1.0, 1.0)


def soft_limit(audio: np.ndarray, ceiling: float) -> np.ndarray:
    """Gentle tanh limiter to approximate peak ceiling in float space."""
    # Scale so that peaks near ±1 map below ceiling
    c = max(0.05, float(ceiling))
    x = audio / c
    return np.tanh(x) * c


def build_board(profile: ModeProfile, sr: int) -> Pedalboard:
    f_hz, g_db, q = profile.peak_db_at_hz
    chain: list = [
        HighpassFilter(cutoff_frequency_hz=profile.highpass_hz),
    ]
    if profile.lowpass_hz is not None:
        chain.append(LowpassFilter(cutoff_frequency_hz=profile.lowpass_hz))
    if g_db != 0.0 and f_hz > 0:
        chain.append(PeakFilter(cutoff_frequency_hz=f_hz, gain_db=g_db, q=q))
    chain.extend(
        [
            Compressor(
                threshold_db=profile.compressor_threshold_db,
                ratio=profile.compressor_ratio,
                attack_ms=10.0,
                release_ms=120.0,
            ),
            Gain(gain_db=profile.makeup_db),
        ]
    )
    return Pedalboard(chain)


def main() -> int:
    args = parse_args()
    src: Path = args.file.expanduser().resolve()
    if not src.is_file():
        print(f"Input file not found: {src}", file=sys.stderr)
        return 1

    mode: Mode = args.mode
    profile = PROFILES[mode]

    default_out = src.with_name(f"{src.stem}_mastered_{mode}.wav")
    out_path: Path = (args.output or default_out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(src, always_2d=True, dtype="float32")
    audio = ensure_float_stereo(audio)

    board = build_board(profile, sr)
    processed = board(audio, sr)

    processed = apply_lufs_match(processed, sr, profile.target_lufs)
    processed = soft_limit(processed, 10 ** (args.peak / 20.0))

    sf.write(out_path, processed, sr, subtype="PCM_24")
    print(f"Mastered ({mode}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
