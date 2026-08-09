#!/usr/bin/env python3

import subprocess
import os
import argparse
import json
import re

class ChapterFrameExtractor:
    def __init__(self, url, output_dir="frames", interval=10):
        self.url = url
        self.output_dir = os.path.abspath(output_dir)
        self.video_path = os.path.abspath("yt_temp_video.mp4")
        self.interval = interval
        self.metadata = {}

    def _sanitize_path(self, text):
        """Removes illegal characters from filenames."""
        return re.sub(r'[\\/*?:"<>|]', "", text).replace(" ", "_")

    def fetch_metadata(self):
        """Retrieves video metadata using yt-dlp."""
        print("Fetching video metadata...")
        cmd = ["yt-dlp", "--dump-json", "--no-playlist", self.url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.metadata = json.loads(result.stdout)

    def download_video(self):
        """Downloads video using a fixed path to avoid extension confusion."""
        if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > 0:
            print(f"Using existing video: {self.video_path}")
            return

        print(f"Downloading: {self.metadata.get('title', 'Video')}")
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", self.video_path,
            self.url
        ]
        subprocess.run(cmd, check=True)

    def extract_by_chapters(self):
        """Iterates through chapters or defaults to full video extraction."""
        chapters = self.metadata.get("chapters", [])

        # Ensure the base output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if not chapters:
            print("No chapters found. Extracting from full duration.")
            self._extract_range(0, self.metadata.get("duration"), self.output_dir)
            return

        for i, chap in enumerate(chapters, 1):
            title = self._sanitize_path(chap.get("title", f"chapter_{i}"))
            start, end = chap.get("start_time"), chap.get("end_time")
            chap_dir = os.path.join(self.output_dir, f"{i:02d}_{title}")
            os.makedirs(chap_dir, exist_ok=True)

            print(f"Processing Chapter {i}: {title} ({start}s - {end}s)")
            self._extract_range(start, end, chap_dir)

    def _extract_range(self, start, end, target_dir):
        """Executes FFmpeg to pull frames at the specified interval."""
        duration = end - start
        output_pattern = os.path.join(target_dir, "frame_%04d.png")

        # Ensure target_dir exists (critical for the 'No Chapters' fallback)
        os.makedirs(target_dir, exist_ok=True)

        # Build command: -ss before -i for fast seeking
        base_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start)]

        if duration < self.interval:
            cmd = base_cmd + ["-i", self.video_path, "-frames:v", "1", output_pattern]
        else:
            fps_val = f"1/{self.interval}"
            cmd = base_cmd + [
                "-t", str(duration),
                "-i", self.video_path,
                "-vf", f"fps={fps_val}",
                "-fps_mode", "vfr",
                output_pattern
            ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  FFmpeg failed at {start}s. Check if {self.video_path} is valid.")

    def run(self):
        try:
            self.fetch_metadata()
            self.download_video()
            self.extract_by_chapters()
            print("\nExtraction complete.")
        except Exception as e:
            print(f"Error during execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    extractor = ChapterFrameExtractor(args.url, interval=args.interval)
    extractor.run()
