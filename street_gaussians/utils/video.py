"""Video encoding utilities."""

import subprocess
from pathlib import Path

from street_gaussians.utils.logger import get_logger

log = get_logger(__name__)


def frames_to_video(frames_dir: Path, output_path: Path, fps: int = 10) -> bool:
    """Encode a directory of numbered PNGs into an MP4 video.

    Args:
        frames_dir: Directory containing %04d.png frame files.
        output_path: Output .mp4 path.
        fps: Frames per second.

    Returns:
        True if encoding succeeded, False if ffmpeg unavailable or failed.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", str(frames_dir / "%04d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        log.info("Video saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log.warning("ffmpeg failed: %s — frames kept at %s/", e, frames_dir)
        return False
