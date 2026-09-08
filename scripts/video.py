import argparse
import glob
import os
from pathlib import Path
from typing import List

from PIL import Image
import cv2
import numpy as np

def render_video(frames: List[str], output: Path, fps: int) -> None:

    video = cv2.VideoWriter(output, cv2.VideoWriter.fourcc(*'mp4v'), fps, (1000, 1000), True) # type: ignore
    frames.append(frames[-1])
    # Appending images to video
    for idx, image in enumerate(frames):
        cv_img = cv2.resize(cv2.imread(image), (1000, 1000), interpolation=cv2.INTER_NEAREST) # type: ignore
        video.write(cv_img)
        print(f"Renderd {idx} ({image})")

    # Release the video file
    video.release()
    cv2.destroyAllWindows() # type: ignore
    return

def main():
    parser = argparse.ArgumentParser(
        description="Stitch frames into video"
    )
    parser.add_argument("frames", type=str, help="Path to the target images")
    parser.add_argument("output", type=str, help="Path to the output video")
    parser.add_argument("--fps", type=int, default=60, help="FPS for video")
    args = parser.parse_args()
    pattern = os.path.join(args.frames, "*.png")
    frame_files = sorted(glob.glob(pattern), key=lambda x: int(Path(x).name.split(".")[0]))
    render_video(frame_files, args.output, args.fps)
if __name__ == "__main__":
    main()