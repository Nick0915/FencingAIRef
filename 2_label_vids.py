#! python3

import time
import cv2
import pandas as pd
import numpy as np

score_info = pd.DataFrame(columns=['ms', 'lscore', 'rscore'], index=[])

def get_score_from_frame(index, cap, score_info):
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if index >= num_frames:
        raise Exception(f'index {index} out of bounds for {num_frames} frames in video')

    if index not in score_info.index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        flag, frame = cap.read()
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.imshow(f'Frame{index}', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cv2.waitKey()
        score = [int(s) for s in input('enter the scores separated by a space: ').split()]
        cv2.destroyAllWindows()

        score_info.loc[index, 'ms'] = score[0]
        score_info.loc[index, 'lscore'] = score[0]
        score_info.loc[index, 'rscore'] = score[1]

    return score_info.loc[index, 'lscore'], score_info.loc[index, 'rscore']

cap = cv2.VideoCapture('./test/test_vid.mp4')
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# will contain index of all frames where the score changes
score_change_frames = [0]
l = 0 # first frame
r = num_frames - 1 # last frame

# outer loop
while get_score_from_frame(l, cap, score_info) != get_score_from_frame(r, cap, score_info):
    t = get_score_from_frame(l, cap, score_info)
    while l != r - 1:
        m = (l + r) // 2
        lscore = get_score_from_frame(l, cap, score_info)
        mscore = get_score_from_frame(m, cap, score_info)
        if lscore == mscore:
            l = m
        else:
            r = m
    score_change_frames.append(r)
    l = r
    r = num_frames - 1

