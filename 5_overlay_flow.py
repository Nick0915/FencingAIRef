#! /usr/bin/env python3

import os
import multiprocessing as mp
import subprocess as sp
import sys
import pandas as pd
import ffmpeg
import cv2
import numpy as np
from datetime import timedelta

CSV_DIR = './Data/ScoreInfo/Sabre/'
VID_DIR = './Data/YtDownloads/Sabre/'
L_CLIP_DIR = './Data/Clips/Sabre/Left/'
R_CLIP_DIR = './Data/Clips/Sabre/Right/'
L_DOWNSAMPLED_DIR = './Data/DownsampledClips/Sabre/Left/'
R_DOWNSAMPLED_DIR = './Data/DownsampledClips/Sabre/Right/'
L_OVERLAID_DIR = './Data/OverlaidClips/Sabre/Left/'
R_OVERLAID_DIR = './Data/OverlaidClips/Sabre/Right/'

def overlay_clip(in_file):
    out_file = L_OVERLAID_DIR + os.path.splitext(in_file)[0].split('/')[-1] + '_overlaid.mp4'

    # skip files that were already processed
    if os.path.exists(out_file):
        print(f'skipping already processed: {in_file}')
        return

    cap = cv2.VideoCapture(in_file)

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    downsampled_fps = original_fps * 2
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{frame_size[0]}*{frame_size[1]}',
        '-pix_fmt', 'bgr24',
        '-r', f'{downsampled_fps}',
        '-i', '-',
        '-an',
        '-vcodec', 'h264',
        '-crf', '10',
        '-b:v', '5000k',
        '-v', 'quiet',
        out_file
    ]

    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT)

    flags, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, next_frame = cap.read()
        if ret:
            hsv = np.zeros((*prev_frame.shape, 3), dtype='uint8')
            hsv[:, :, 1] = 255

            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            # actual optical flow calculation
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                0.5, # pyramid level scaling factor (each level is this multiple of the higher level)
                3, # number of pyramid levels
                15, # window size, higher is more robust to noise but is blurrier
                1, # n iterations done at each pyramid level
                5, # polynomial pixel neighborhood size
                1.2, # sigma for gaussian (poly_n = 5 -> 1.1, poly_n = 7 -> 1.5
                0, # flags
            )


            # center the flow values at the mean to reduce effect of camera panning
            flow[:, :, 0] -= np.mean(flow[:, :, 0])
            flow[:, :, 1] -= np.mean(flow[:, :, 1])

            # flow[:, :, 0] = cv2.blur(flow[:, :, 0], (5, 5))
            # flow[:, :, 1] = cv2.blur(flow[:, :, 1], (5, 5))


            magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
            hsv[:, :, 0] = angle * (180 / np.pi / 2)
            # hsv[:, :, 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # hsv[:, :, 2] = cv2.blur(hsv[:, :, 2], (11, 11))

            # hsv[:, :, 1][hsv[:, :, 1] < 112] = 0
            hsv[:, :, 2][hsv[:, :, 2] < 50] = 0
            flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # display = np.vstack((cv2.cvtColor(next_frame, cv2.COLOR_GRAY2BGR), flow_frame))
            overlaid = np.zeros((*next_frame.shape, 3), dtype='uint8')
            overlaid[:, :, 0] = next_frame
            overlaid[:, :, 1] = next_frame
            overlaid[:, :, 2] = next_frame
            # display[:, :, 1] = next_frame
            # display[:, :, 0] = flow_frame[:, :, 0]
            # display[:, :, 2] = flow_frame[:, :, 1]
            overlay_opacity = 0.5
            cv2.addWeighted(flow_frame, overlay_opacity, overlaid, 1 - overlay_opacity,
                0, overlaid)


            proc.stdin.write(overlaid.tobytes())
            # proc.wait()
            # print('frame processed')

            prev_frame = next_frame
        else:
            break

    proc.stdin.close()
    # proc.stdout.close()
    # proc.stderr.close()
    cap.release()
    print(f'{in_file} converted to {out_file}')


if __name__ == '__main__':
    files = [L_DOWNSAMPLED_DIR + f for f in os.listdir(L_DOWNSAMPLED_DIR)] + \
            [R_DOWNSAMPLED_DIR + f for f in os.listdir(R_DOWNSAMPLED_DIR)]

    total = len(files)

    tasks = [(f,) for f in files[:1000]]

    with mp.Pool(processes=12) as pool:
        pool.starmap(overlay_clip, tasks)
    # [overlay_clip(f) for f in files[:36]]
