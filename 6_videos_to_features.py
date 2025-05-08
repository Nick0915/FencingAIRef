#! /usr/bin/env python3

import os
import numpy as np
from PIL import Image
from pprint import pprint
import cv2
import multiprocessing as mp

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch

CSV_DIR = './Data/ScoreInfo/Sabre/'
VID_DIR = './Data/YtDownloads/Sabre/'
L_CLIP_DIR = './Data/Clips/Sabre/Left/'
R_CLIP_DIR = './Data/Clips/Sabre/Right/'
L_DOWNSAMPLED_DIR = './Data/DownsampledClips/Sabre/Left/'
R_DOWNSAMPLED_DIR = './Data/DownsampledClips/Sabre/Right/'
L_OVERLAID_DIR = './Data/OverlaidClips/Sabre/Left/'
R_OVERLAID_DIR = './Data/OverlaidClips/Sabre/Right/'
L_VECTOR_DIR = './Data/Vectors/Sabre/Left/'
R_VECTOR_DIR = './Data/Vectors/Sabre/Right/'

def video2vector(in_file, model, transform):
    if 'Left' in in_file:
        out_file = L_VECTOR_DIR + os.path.splitext(in_file)[0].split('/')[-1] + '.pt'
    else:
        out_file = R_VECTOR_DIR + os.path.splitext(in_file)[0].split('/')[-1] + '.pt'

    if os.path.exists(out_file):
        print(f'skipping (already converted): {out_file}')
        return

    print(f'converting frames to 4D tensor: {in_file}')
    cap = cv2.VideoCapture(in_file)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    tensors = []

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < n_frames:
        flags, frame = cap.read()
        if not flags:
            raise Exception('No frame read!!!')

        frame = Image.fromarray(frame)
        tensor = transform(frame)
        tensors.append(tensor)

    if len(tensors) == 0:
        print(f'got 0-tensor from {in_file}, skipping...')
        return

    tensors = torch.tensor(np.array(tensors)).to(device)

    print(f'running tensors ({tensors.shape}) through model: {in_file}')
    with torch.no_grad():
        out = model(tensors)
        torch.save(out.cpu(), out_file)
        print(f'save {out_file}')

if __name__ == '__main__':
    mp.set_start_method('spawn')

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    model = timm.create_model('inception_v4', pretrained=True).to(device)
    model.share_memory()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)


    files = [L_OVERLAID_DIR + f for f in os.listdir(L_OVERLAID_DIR)] + \
            [R_OVERLAID_DIR + f for f in os.listdir(R_OVERLAID_DIR)]

    # tasks = [(f, model, transform) for f in files[:100]]

    # with mp.Pool(processes=32) as pool:
    #     pool.starmap(video2vector, tasks)

    # sequentially, takes 1m for 100 files on 1g.10gb GPU, 3:30:00+ for 21200 files
    [video2vector(f, model, transform) for f in files]
