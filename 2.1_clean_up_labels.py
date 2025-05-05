#! python3

import os
import pandas as pd
import cv2
import numpy as np
import multiprocessing as mp

CLIP_LENGTH_MS = 3000

# through manual counting, each light lasts at least 50 frames,
# so jumping backward by 20 frames at a time guarantees we won't miss a light
BACKWARD_JUMP_FRAMES = 5
# if we have to jump back more than 7.5 seconds from a score increase, we should probably discard that clip
MAX_BACKWARD_JUMP_MS = 7_500

# bounding boxes for score lights
PATCH_HEIGHT = 5
PATCH_WIDTH = 100

PATCH_TOP = 330
PATCH_BOTTOM = PATCH_TOP + PATCH_HEIGHT
L_PATCH_LEFT = 90
L_PATCH_RIGHT = L_PATCH_LEFT + PATCH_WIDTH
R_PATCH_LEFT = 450
R_PATCH_RIGHT = R_PATCH_LEFT + PATCH_WIDTH

COLORED_LIGHT_THRESH = 0.25

CSV_DIR = './Data/ScoreInfo/Sabre/'
VID_DIR = './Data/YtDownloads/Sabre/'
CLIP_DIR = './Data/Clips/Sabre/'

csvs = os.listdir(CSV_DIR)

def clean_up_file(csv_file):
    exceptional, nominal, time_saved = 0, 0, 0.

    info = pd.read_csv(CSV_DIR + csv_file, header=0)
    info = info.drop(columns=[
        'Unnamed: 0', 'Unnamed: 0.1', 'nominal',
        'clip_start_ms', 'clip_end_ms', 'left_lit', 'right_lit'
    ], errors='ignore')
    info.insert(6, 'nominal', True)
    info.insert(7, 'clip_start_ms', -1.0)
    info.insert(8, 'clip_end_ms', -1.0)
    info.insert(9, 'left_lit', False)
    info.insert(10, 'right_lit', False)
    info['frame_no'] = info['frame_no'].astype(int)

    prev_lscore, prev_rscore = 0, 0

    video_file = os.path.splitext(csv_file)[0] + '.mp4'
    cap = cv2.VideoCapture(VID_DIR + video_file)
    for i, row in info.iterrows():
        #region clean up based on scores
        lscore, rscore = row['lscore'], row['rscore']
        if prev_lscore > lscore or prev_rscore > rscore:
            # weed out scores that get LOWER some how (not possible in nomral fencing)
            exceptional += 1
            info.iloc[i, 6] = False
        elif lscore > prev_lscore + 1 or rscore > prev_rscore + 1:
            # weed out scores that jump by more than ONE (not possible without a red card, we aren't training on this scenario)
            exceptional += 1
            info.iloc[i, 6] = False
        elif lscore > prev_lscore and rscore > prev_rscore:
            # weed out situations where BOTH fencers score a point (not possible without a red card, we aren't training on this scenario)
            exceptional += 1
            info.iloc[i, 6] = False
        elif row['lconf'] < 0.5 or row['rconf'] < 0.5:
            # weed out low confidence scores so they don't put bad data into the model
            exceptional += 1
            info.iloc[i, 6] = False
        else:
            nominal += 1

        prev_lscore, prev_rscore = lscore, rscore
        #endregion

        #region clean up based on lights
        if i == 0 or not info.iloc[i, 6]:
            # don't bother with abnormal score events
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_no'])

        earliest_lit_ms = row['ms']
        left_lit, right_lit = False, False
        while True:
            flags, frame = cap.read()

            # these are encoded as BGR channels, so blue is 0, green is 1, red is 2
            l_light = frame[PATCH_TOP:PATCH_BOTTOM, L_PATCH_LEFT:L_PATCH_RIGHT, :]
            r_light = frame[PATCH_TOP:PATCH_BOTTOM, R_PATCH_LEFT:R_PATCH_RIGHT, :]

            # left's light will always be red and right's light will always be green

            # we want a high activation in the desired channel AND a low activation in the opposite channel
            # to be robust against random things in the background

            # left's light has a high red channel activation AND a low green channel activation
            left_colored_on = (l_light[:, :, 2].sum() / l_light[:, :].size > COLORED_LIGHT_THRESH * 255) and \
                l_light[:, :, 1].sum() / l_light[:, :].size < COLORED_LIGHT_THRESH * 255
            # right's light has a high green channel activation AND a low red channel activation
            right_colored_on = (r_light[:, :, 1].sum() / r_light[:, :].size > COLORED_LIGHT_THRESH * 255) and \
                r_light[:, :, 2].sum() / r_light[:, :].size < COLORED_LIGHT_THRESH * 255

            # cv2.imshow('frame', frame)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # searching too far back yet haven't found a lit up frame... probably abnormal score event
            if pos_ms < row['ms'] - MAX_BACKWARD_JUMP_MS:
                info.iloc[i, 6] = False
                exceptional += 1
                nominal -= 1
                break

            # info.insert(6, 'nominal', True)
            # info.insert(7, 'clip_start_ms', -1.0)
            # info.insert(8, 'clip_end_ms', -1.0)
            # info.insert(9, 'left_lit', False)
            # info.insert(10, 'right_lit', False)

            # if it's a lit frame
            if left_colored_on or right_colored_on:
                earliest_lit_ms = pos_ms
                left_lit = left_lit or left_colored_on
                right_lit = right_lit or right_colored_on

            # if (left_lit or right_lit) and (not left_colored_on) and (not right_colored_on):
            if (left_lit or right_lit) and (int(left_lit) + int(right_lit) > int(left_colored_on) + int(right_colored_on)):
                # if we've seen a lit frame ahead of us but number of lights went down,
                # we've searched too far backward, the last searched frame is the earliest lit frame
                info.iloc[i, 8] = earliest_lit_ms
                info.iloc[i, 7] = np.max((earliest_lit_ms - CLIP_LENGTH_MS, 0.)) # can't have negative time

                info.iloc[i, 9] = left_lit
                info.iloc[i, 10] = right_lit
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frames - BACKWARD_JUMP_FRAMES)

        if info.iloc[i, 6]:
            time_saved += row['ms'] - info.iloc[i, 8]
        #endregion

        progress = i / info.shape[0]
        print(f'{csv_file}: {progress * 100 :.2f}%')

    csv_text = info.to_csv(index=False)
    with open(CSV_DIR + csv_file, 'w') as f:
        f.write(csv_text)

    print(f'finished processing {csv_file}')
    return exceptional, nominal, time_saved


if __name__ == '__main__':
    files = os.listdir(CSV_DIR)
    tasks = [(f,) for f in files]

    nominal, exceptional, time_saved = 0, 0, 0

    # mp.set_start_method('spawn')
    with mp.Pool(processes=20) as pool:
        for exceptional_, nominal_, time_saved_ in pool.starmap(clean_up_file, tasks):
            nominal += nominal_
            exceptional += exceptional_
            time_saved += time_saved_
    # [clean_up_file(f) for f in files[:1]]

    print(f'{exceptional = }, {nominal = }, %exceptional = {exceptional / (exceptional + nominal) * 100 :.2f}%, %nominal = {nominal / (exceptional + nominal) * 100 :.2f}%')
    print(f'{time_saved = :.2f}ms, {time_saved / nominal :.2f}ms per nominal event')
