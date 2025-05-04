#! python3

import os
import pandas as pd

CSV_DIR = './Data/ScoreInfo/Sabre/'

csvs = os.listdir(CSV_DIR)

nominal = 0
exceptional = 0

for csv_file in csvs:
    info = pd.read_csv(CSV_DIR + csv_file, header=0)
    if 'nominal' not in info.columns:
        info.insert(6, 'nominal', True)

    info = info.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    info['frame_no'] = info['frame_no'].astype(int)

    prev_lscore, prev_rscore = 0, 0
    for i, row in info.iterrows():
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

    csv_text = info.to_csv(index=False)
    with open(CSV_DIR + csv_file, 'w') as f:
        f.write(csv_text)

print(f'{exceptional = }, {nominal = }, %exceptional = {exceptional / (exceptional + nominal) * 100 :.2f}%, %nominal = {nominal / (exceptional + nominal) * 100 :.2f}%')
