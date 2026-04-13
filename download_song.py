#!/usr/bin/env python3
"""
Download audio from a YouTube URL into ./assets (or --output-dir) as WAV by default.
Requires: ffmpeg on PATH (used by yt-dlp for extraction).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yt_dlp import YoutubeDL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download audio from a YouTube URL.")
    p.add_argument(
        "--url",
        required=True,
        help="YouTube video URL",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets"),
        help="Directory for downloaded files (default: assets)",
    )
    p.add_argument(
        "--format",
        choices=("wav", "mp3", "m4a", "flac"),
        default="wav",
        help="Audio container/codec after extraction (default: wav)",
    )
    p.add_argument(
        "--filename",
        default="%(title)s",
        help="yt-dlp output template basename without extension (default: %(title)s)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(out_dir / f"{args.filename}.%(ext)s")

    codec_map = {
        "wav": "wav",
        "mp3": "mp3",
        "m4a": "m4a",
        "flac": "flac",
    }
    preferredcodec = codec_map[args.format]

    ydl_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": preferredcodec,
                "preferredquality": "192",
            }
        ],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.url])
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    print(f"Saved under: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
