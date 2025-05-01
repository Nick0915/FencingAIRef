#! python3

import os
import time
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch

from transformers import AutoImageProcessor, SiglipForImageClassification


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('using the GPU!!! 😎')
else:
    device = torch.device('cpu')
    print('using the CPU!!! 😢')

# ! Make sure to run `1_download_data.py` first!!!

VID_DIR = './Data/YtDownloads/Sabre/'
CSV_DIR = './Data/ScoreInfo/Sabre/'

# Model to read score from frames
model_name = "prithivMLmods/Mnist-Digits-SigLIP2"
mnist_model = SiglipForImageClassification.from_pretrained(model_name).to(device)
input_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

# info on where in the frame the score can be found
PATCH_HEIGHT = 14
PATCH_WIDTH = 19
PATCH_TOP = 310
PATCH_BOTTOM = PATCH_TOP + PATCH_HEIGHT
L_PATCH_LEFT = 268
L_PATCH_RIGHT = L_PATCH_LEFT + PATCH_WIDTH
L_PATCH_MID = (L_PATCH_LEFT + L_PATCH_RIGHT) // 2 - 2
R_PATCH_LEFT = 358
R_PATCH_RIGHT = R_PATCH_LEFT + PATCH_WIDTH
R_PATCH_MID = (R_PATCH_LEFT + R_PATCH_RIGHT) // 2 - 2
ADJ = -32

def horiz_pad(patch):
    hpad = np.zeros((patch.shape[0], 28 - patch.shape[1]))
    return np.hstack((patch, hpad))

def predict_score_from_frame(frame, view_patches=False):
    # the whole patch of FotL's score
    # this will be most accurate if score is 1 digit
    lwhole_patch = frame[PATCH_TOP:PATCH_BOTTOM, L_PATCH_LEFT:L_PATCH_RIGHT]
    _, lwhole_patch = cv2.threshold(lwhole_patch, 255 + ADJ, 255, cv2.THRESH_BINARY)
    l_double = np.any(lwhole_patch[:, 0] > 100) # if any pixels on the left border are lit, it's double digits (1X where 0 <= X <= 5)
    lwhole_patch = horiz_pad(lwhole_patch)

    if l_double:
        # the masked RHS half-patch of FotL's score
        lhalf_patch2 = lwhole_patch.copy()
        lhalf_patch2[:, :(L_PATCH_MID-L_PATCH_LEFT)] = 0
        _, lhalf_patch2 = cv2.threshold(lhalf_patch2, 255 + ADJ, 255, cv2.THRESH_BINARY)
        lhalf_patch2 = horiz_pad(lhalf_patch2)
    else:
        lhalf_patch2 = None

    if view_patches:
        if not l_double:
            cv2.imshow('left patch whole', lwhole_patch)
        else:
            cv2.imshow('left patch half2', lhalf_patch2)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # the whole patch of FotR's score
    rwhole_patch = frame[PATCH_TOP:PATCH_BOTTOM, R_PATCH_LEFT:R_PATCH_RIGHT]
    _, rwhole_patch = cv2.threshold(rwhole_patch, 255 + ADJ, 255, cv2.THRESH_BINARY)
    r_double = np.any(rwhole_patch[:, 0] > 100) # if any pixels on the left border are lit, it's double digits (1X where 0 <= X <= 5)
    rwhole_patch = horiz_pad(rwhole_patch)

    if r_double:
        # the masked RHS half-patch of FotR's score
        rhalf_patch2 = rwhole_patch.copy()
        rhalf_patch2[:, :(L_PATCH_MID-L_PATCH_LEFT)] = 0
        _, rhalf_patch2 = cv2.threshold(rhalf_patch2, 255 + ADJ, 255, cv2.THRESH_BINARY)
        rhalf_patch2 = horiz_pad(rhalf_patch2)
    else:
        rhalf_patch2 = None

    if view_patches:
        if not r_double:
            cv2.imshow('right patch whole', rwhole_patch)
        else:
            cv2.imshow('left patch half2', rhalf_patch2)
        cv2.waitKey()
        cv2.destroyAllWindows()

    lwhole_input = None
    lhalf2_input = None
    rwhole_input = None
    rhalf2_input = None

    if not l_double:
        lwhole_input = input_processor(images=Image.fromarray(lwhole_patch), return_tensors='pt')
        lwhole_input['pixel_values'] = lwhole_input['pixel_values'].to(device)
    else:
        lhalf2_input = input_processor(images=Image.fromarray(lhalf_patch2), return_tensors='pt')
        lhalf2_input['pixel_values'] = lhalf2_input['pixel_values'].to(device)

    if not r_double:
        rwhole_input = input_processor(images=Image.fromarray(rwhole_patch), return_tensors='pt')
        rwhole_input['pixel_values'] = rwhole_input['pixel_values'].to(device)
    else:
        rhalf2_input = input_processor(images=Image.fromarray(rhalf_patch2), return_tensors='pt')
        rhalf2_input['pixel_values'] = rhalf2_input['pixel_values'].to(device)

    with torch.no_grad():
        if not l_double:
            lwhole_outputs = mnist_model(**lwhole_input)
            lwhole_logits = lwhole_outputs.logits
            lwhole_probs = torch.nn.functional.softmax(lwhole_logits, dim=1).squeeze()
        else:
            lwhole_outputs = None
            lwhole_logits = None
            lwhole_probs = None

        if l_double:
            lhalf2_outputs = mnist_model(**lhalf2_input)
            lhalf2_logits = lhalf2_outputs.logits
            lhalf2_probs = torch.nn.functional.softmax(lhalf2_logits, dim=1).squeeze()
        else:
            lhalf2_outputs = None
            lhalf2_logits = None
            lhalf2_probs = None

        if not r_double:
            rwhole_outputs = mnist_model(**rwhole_input)
            rwhole_logits = rwhole_outputs.logits
            rwhole_probs = torch.nn.functional.softmax(rwhole_logits, dim=1).squeeze()
        else:
            rwhole_outputs = None
            rwhole_logits = None
            rwhole_probs = None

        if r_double:
            rhalf2_outputs = mnist_model(**rhalf2_input)
            rhalf2_logits = rhalf2_outputs.logits
            rhalf2_probs = torch.nn.functional.softmax(rhalf2_logits, dim=1).squeeze()
        else:
            rhalf2_outputs = None
            rhalf2_logits = None
            rhalf2_probs = None

    def most_likely(whole_probs, half2_probs, double_digit):
        if double_digit:
            num = 10 + half2_probs.argmax().item()
            prob = half2_probs.max().item()

            # 17 not a possible answer, but 1's get classified as 7's too often so this is most likely
            if num == 17:
                num = 11
        else:
            num = whole_probs.argmax().item()
            prob = whole_probs.max().item()

        return num, prob

    return most_likely(lwhole_probs, lhalf2_probs, l_double),\
            most_likely(rwhole_probs, rhalf2_probs, r_double)

DATA_COLUMNS = ['frame_no', 'ms', 'lscore', 'rscore', 'lconf', 'rconf']

def get_score_from_frame(index, cap, score_info):
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if index >= num_frames:
        raise Exception(f'index {index} out of bounds for {num_frames} frames in video')

    if index not in score_info.index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        (lscore, lconf), (rscore, rconf) = predict_score_from_frame(frame)

        score_info.loc[index, 'frame_no'] = frame_no
        score_info.loc[index, 'ms'] = time

        score_info.loc[index, 'lscore'] = lscore
        score_info.loc[index, 'rscore'] = rscore

        score_info.loc[index, 'lconf'] = lconf
        score_info.loc[index, 'rconf'] = rconf

    return score_info.loc[index, 'lscore'], score_info.loc[index, 'rscore']

def main():
    files = os.listdir(VID_DIR)
    for file in files[:5]:
        cap = cv2.VideoCapture(VID_DIR + file)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # num_frames = 10_000
        score_info = pd.DataFrame(columns=DATA_COLUMNS, index=[])

        # will contain index of all frames where the score changes
        score_change_frames = [0]
        # note: the following l and r (and lscore and rscore) do NOT refer to the fencers' scores
        #     they refer to the total score (expressed as X-Y) at the left and right frame markers
        l = 0 # first frame
        r = num_frames - 1 # last frame

        # outer loop
        while l != num_frames - 1:
            while l != r - 1:
                m = (l + r) // 2
                lscore = get_score_from_frame(l, cap, score_info)
                mscore = get_score_from_frame(m, cap, score_info)
                if lscore == mscore:
                    l = m
                else:
                    r = m
            score_change_frames.append(r)
            # print(f'Score changed to {score_info.loc[r, 'lscore']}-{score_info.loc[r, 'rscore']} at frame#{r}, ({score_info.loc[r, 'ms']}ms)')
            print(f'{file} progress: {100 * r / num_frames:.2f}%')
            l = r
            r = num_frames - 1


        # score_info = pd.DataFrame(columns=DATA_COLUMNS, index=[])
        score_info.set_index('frame_no')
        score_info = score_info.sort_values('ms', ascending=True)

        rows_to_drop = score_info.index.difference(score_change_frames)
        score_info = score_info.drop(index=rows_to_drop)

        score_info.reset_index()

        csv = score_info.to_csv()
        with open(CSV_DIR + f'{os.path.splitext(file)[0]}.csv', 'w') as f:
            f.write(csv)

if __name__ == '__main__':
    main()
