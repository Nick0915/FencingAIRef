
# ! #THIS FILE SHOULD NOT BE RUN!!! IT IS A DEPENDENCY FOR ANOTHER FILE

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

L_VECTOR_DIR = './Data/Vectors/Sabre/Left/'
R_VECTOR_DIR = './Data/Vectors/Sabre/Right/'

class ClipVectorDataset(Dataset):
    def __init__(self, left_dir, right_dir):
        self.left_dir = left_dir
        self.right_dir = right_dir

        # N: number of clips
        # F: number of frames in that clip
        # D: 1000, the number of features coming out of the CNN

        left_files = [os.path.join(left_dir, f) for f in os.listdir(left_dir)]
        right_files = [os.path.join(right_dir, f) for f in os.listdir(right_dir)]
        self.files = left_files + right_files # [(F, D), ... N times]


    def __len__(self):
        left_files = os.listdir(self.left_dir)
        right_files = os.listdir(self.right_dir)
        return len(left_files) + len(right_files)

    def __getitem__(self, idx):
        vec = torch.load(self.files[idx])
        # label = 1 if 'Left' in self.files[idx] else 0
        # label = np.zeros(2, dtype='float32')
        if 'Left' in self.files[idx]:
            label = 0
        else:
            label = 1
        # label[0] += float('Left' in self.files[idx])
        # label[1] += float('Right' in self.files[idx])

        return vec, label

    def get(self, idx):
        vec, label = self[idx]
        file_name = self.files[idx]

        return vec, label, file_name

    def collate(self, rows):
        # each sample's shape is (F, D)
        seq_lengths = [sample.shape[0] for sample, label in rows]
        padded = torch.nn.utils.rnn.pad_sequence([sample for sample, label in rows], batch_first=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        labels = torch.tensor([label for sample, label in rows])

        return packed, labels

if __name__ == '__main__':
    #! THE FOLLOWING IS JUST FOR TESTING PURPOSES. DO NOT RUN THIS.
    clip_vectors = ClipVectorDataset(L_VECTOR_DIR, R_VECTOR_DIR)
    loader = DataLoader(
        dataset=clip_vectors,
        batch_size=32,
        shuffle=True,
        collate_fn=clip_vectors.collate
    )

    for i, batch in enumerate(loader):
        if i > 10:
            break

        data, labels = batch
        print(data)
        pass
