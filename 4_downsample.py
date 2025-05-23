#! /usr/bin/env python3

import os
import multiprocessing as mp
import pandas as pd
import ffmpeg
import cv2
import numpy as np
from datetime import timedelta

FFMPEG_BIN = "ffmpeg"
import subprocess as sp
import os

fps = str(13)

CSV_DIR = './Data/ScoreInfo/Sabre/'
VID_DIR = './Data/YtDownloads/Sabre/'
L_CLIP_DIR = './Data/Clips/Sabre/Left/'
R_CLIP_DIR = './Data/Clips/Sabre/Right/'
L_DOWNSAMPLED_DIR = './Data/DownsampledClips/Sabre/Left/'
R_DOWNSAMPLED_DIR = './Data/DownsampledClips/Sabre/Right/'

# how long the 'full fps' part of the vid should be, the rest is at half-fps
FULL_FPS_LEN = 24

def downsample_clip(in_file):
    if "_Left.mp4" in in_file:
        out_file = L_DOWNSAMPLED_DIR + os.path.splitext(in_file)[0].split('/')[-1] + '.mp4'
    else:
        out_file = R_DOWNSAMPLED_DIR + os.path.splitext(in_file)[0].split('/')[-1] + '.mp4'

    cap = cv2.VideoCapture(in_file)
    downsample_until = cap.get(cv2.CAP_PROP_FRAME_COUNT) - FULL_FPS_LEN
    downsample_by = 2

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    downsampled_fps = original_fps // 2
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{frame_size[0]}*{frame_size[1]}',
        '-pix_fmt', 'bgr24',
        '-r', f'{original_fps}',
        '-i', '-',
        '-an',
        '-vcodec', 'h264',
        '-b:v', '5000k',
        '-v', 'quiet',
        out_file
    ]

    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

    # frames = []

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_no <= downsample_until and frame_no % downsample_by == 0:
                # frames.append(frame)
                proc.stdin.write(frame.tobytes())
            elif frame_no > downsample_until:
                # frames.append(frame)
                proc.stdin.write(frame.tobytes())
        else:
            break
        frame_no += 1

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # writer = cv2.VideoWriter(out_file, fourcc, float(downsampled_fps), frame_size)
    # for frame in frames:
    #     writer.write(frame)

    # writer.release()

    proc.stdin.close()
    proc.stderr.close()
    # proc.kill()

    cap.release()

    print(f'{in_file} converted to {out_file}')

if __name__ == '__main__':
    files = [L_CLIP_DIR + f for f in os.listdir(L_CLIP_DIR)] + \
            [R_CLIP_DIR + f for f in os.listdir(R_CLIP_DIR)]

    total = len(files)

    tasks = [(f,) for f in files]

    with mp.Pool(processes=6) as pool:
        pool.starmap(downsample_clip, tasks)

    # [downsample_clip(f) for f in files[:100]]
