#! /usr/bin/env python3

import os
import multiprocessing as mp
import pandas as pd
import ffmpeg
import numpy as np
from datetime import timedelta

CSV_DIR = './Data/ScoreInfo/Sabre/'
VID_DIR = './Data/YtDownloads/Sabre/'
CLIP_DIR = './Data/Clips/Sabre/'

def ms_to_hhmmssms(ms):
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    ms, seconds, minutes, hours = [int(t) for t in [ms, seconds, minutes, hours]]
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def trim(in_file, out_file, start_ms, end_ms, flipped=False):
    formatted_start = ms_to_hhmmssms(start_ms)
    formatted_end = ms_to_hhmmssms(end_ms)

    probe_result = ffmpeg.probe(in_file)

    stream = ffmpeg.input(in_file, ss=formatted_start, to=formatted_end)
    if flipped:
        stream = ffmpeg.hflip(stream)
        pass

    # stream = stream.trim(start=formatted_start, end=formatted_end).setpts('PTS-STARTPTS')
    output = ffmpeg.output(
        stream,
        out_file,
        vcodec='copy',
        # vcodec='h264',
        loglevel='quiet'
    ).run()

def clip_video(csv_file):
    video_file = VID_DIR + os.path.splitext(csv_file)[0] + '.mp4'

    info = pd.read_csv(CSV_DIR + csv_file, header=0)
    for i, row in info.iterrows():
        if i == 0 or not info.iloc[i]['nominal'] or not info.iloc[i - 1]['nominal']:
            # skip the first event and all abnormal events
            continue

        prev_lscore, prev_rscore = info.iloc[i - 1]['lscore'], info.iloc[i - 1]['rscore']
        lscore, rscore = info.iloc[i]['lscore'], info.iloc[i]['rscore']

        winner = 'Right'
        loser = 'Left'
        if lscore > prev_lscore:
            winner = 'Left'
            loser = 'Right'

        orig_clip_file = CLIP_DIR + f'{winner}/' + os.path.splitext(csv_file)[0] + f'_clip{i:02d}_o_{winner}' + '.mp4'
        # flipped_clip_file = CLIP_DIR + f'{loser}/' + os.path.splitext(csv_file)[0] + f'_clip{i:02d}_f_{loser}' + '.mp4'

        trim(video_file, orig_clip_file, row['clip_start_ms'], row['clip_end_ms'], flipped=False)
        # for now, no flipped vids
        # trim(video_file, flipped_clip_file, row['clip_start_ms'], row['clip_end_ms'], flipped=True)
        # print(f'{video_file}: {(i + 1) / info.shape[0] * 100:.2f}%')

    print(f'finished cutting up {video_file}')


if __name__ == '__main__':
    files = os.listdir(CSV_DIR)
    tasks = [(f,) for f in files]

    with mp.Pool(processes=64) as pool:
        pool.starmap(clip_video, tasks)
        print(f'finished processing all files in {CSV_DIR}')
    # [clip_video(f) for f in files[:1]]
