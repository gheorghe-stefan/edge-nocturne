"""Converts legacy MPEG-4 clips in storage/events to web-standard H.264 (avc1) with faststart."""

import glob
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_DIR = os.path.join(BASE_DIR, "storage", "events")


def convert_all():
    clips = glob.glob(os.path.join(EVENTS_DIR, "event_*.mp4"))
    print(f"[INFO] Checking {len(clips)} event clips in {EVENTS_DIR}...")

    converted = 0
    for clip in clips:
        if clip.endswith(".raw.mp4") or clip.endswith(".tmp.mp4"):
            continue

        # Check if already h264
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", clip]
        res = subprocess.run(probe_cmd, capture_output=True, text=True)
        codec = res.stdout.strip()
        if codec == "h264":
            continue

        print(f"[INFO] Converting {os.path.basename(clip)} ({codec} -> h264)...")
        tmp_path = clip + ".h264.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", clip,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", tmp_path,
        ]
        conv_res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if conv_res.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, clip)
            converted += 1
            print(f"[SUCCESS] Converted {os.path.basename(clip)} to H.264!")
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print(f"[DONE] Finished converting {converted} clips.")


if __name__ == "__main__":
    convert_all()
