import os.path

import cv2
import numpy as np

from ClipProcessing.all_clips_decor import all_clips
from ClipProcessing.helping_functions import capture_video_clip_frames, get_clip_file_name

THRESHOLD = 32


def get_event(diff):
    if diff < -THRESHOLD:
        return np.array((0xFF, 0, 0))
    elif diff < THRESHOLD:
        return np.array((0, 0, 0))
    else:
        return np.array((0, 0xFF, 00))


event_mat = np.vectorize(get_event)


def event_video(output_dir: str, input_clip: str):
    clip_file_name = get_clip_file_name(input_clip)
    greyscale_frames = capture_video_clip_frames(input_clip)
    if greyscale_frames:
        event_frames = [np.zeros(greyscale_frames[0].shape)]
        for prev, cur in zip(greyscale_frames[:-1], greyscale_frames[1:]):
            diff = np.array(cur, dtype=int) - np.array(prev, dtype=int)
            event_frame = event_mat(diff)
            event_frames.append(event_frame)
        out = cv2.VideoWriter(os.path.join(output_dir, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, event_frames[0].shape)
        for frame in event_frames:
            out.write(frame)
        out.release()


if __name__ == '__main__':
    all_clips(event_video)
