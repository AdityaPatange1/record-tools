#!/usr/bin/env python3
"""
Strip vocals using Demucs two-stem separation (vocals vs no_vocals / instrumental).
Outputs the beat-only WAV next to the input by default, or under --output.

Requires: ffmpeg on PATH; Demucs models download on first run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract instrumental (beat) from a mixed song using Demucs.",
    )
    p.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to input audio (e.g. assets/track.wav or .mp3)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path. Default: <stem>_beat.wav beside the input file",
    )
    p.add_argument(
        "--model",
        default="htdemucs_ft",
        help="Demucs model name (default: htdemucs_ft)",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device (default: auto)",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete Demucs output folder under the temp directory",
    )
    return p.parse_args()


def find_no_vocals(sep_root: Path, stem_name: str) -> Path:
    """Demucs writes separated/<model>/<track_stem>/no_vocals.wav."""
    matches = list(sep_root.rglob("no_vocals.wav"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find no_vocals.wav under {sep_root}. "
            f"Expected Demucs layout under {sep_root!s}."
        )
    # Prefer exact stem folder if present
    for m in matches:
        if m.parent.name == stem_name:
            return m
    return matches[0]


def main() -> int:
    args = parse_args()
    src: Path = args.file.expanduser().resolve()
    if not src.is_file():
        print(f"Input file not found: {src}", file=sys.stderr)
        return 1

    stem_name = src.stem
    default_out = src.with_name(f"{stem_name}_beat.wav")
    out_path: Path = (args.output or default_out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        args.model,
    ]
    if args.device != "auto":
        cmd.extend(["-d", args.device])

    tmp_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
    sep_out = tmp_dir / "separated"
    try:
        cmd.extend(["-o", str(sep_out), str(src)])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        nv = find_no_vocals(sep_out, stem_name)
        shutil.copy2(nv, out_path)
        print(f"Beat (instrumental) written to: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"Demucs failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode or 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif args.keep_temp:
            print(f"Kept temp directory: {tmp_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
