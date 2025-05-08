# Fencing AI Referee Notes

This file will be used to keep track of notes and other significant things in order to write a report at the end of the project.
I will track things such as:
* experiments
* design choices
* difficulties
* references
* etc.

## Project Structure Philosophy
* Keep everything modular (i.e. separate module for loading data, for preprocessing, etc.)
* Use a notebook for experimentation, testing, etc.
* Combine into an application file at the end

## Steps
* Started by [forking](https://github.com/Nick0915/FencingAIRef) and cloning the [original repo](https://github.com/sholtodouglas/fencing-AI)
  * This serves as inspiration/reference, but we take slightly different approaches
  * I aim to improve his performance by using a new model

0) Prepare workspace organization with a script (`0_prepare.py`)
1) Download videos from YouTube (`1_download_data.py`)
    * We already have a list of sabre videos of the same format, so we'll start with this for now
    * Used a multiprocessing pool to download many videos in parallel (100 videos in ~ 1m20s)
2) Label key frames in video where the score changes, and what the score is at that point (`2_label_vids.py`)
    * First, assume we have a function that can spit out the score based on reading one frame
    * We develop an algorithm that binary-searches for the exact frame these scores change1
    * Then, implement the assumed function using an off-the-shelf huggingface model trained on MNIST digits
    * References:
      * https://pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/
      * https://huggingface.co/prithivMLmods/Mnist-Digits-SigLIP2
    * With this naive algorithm (minimal changes), I can label 20 videos in 25 minutes (data set is on the order of ~600 videos)
      * Would take >20 hours to label everything
      * Ended up doing this on hopper with multiprocessing
        * Forgot to time it, but it took <8 hours (that's how long I salloc'd for)
    * This algorithm assumes that scores only ever increase monotonically (ideally, they do), but they may jump around weirdly in the video
      * Due to ref mistakes, tech issues, cards, overturning points, etc.
      * So we are losing some potentially useful training points, but it's pretty uncommon so not a big deal
    * Finally, check the validity of these labels (in `2.1_clean_up_labels.py`) and clean them up
      * Fix up some stuff with the csv formatting
      * Importantly: added `nominal` column which is true if that score change made sense. We want to weed out the following:
        * Scores going down (ref mistake, we shouldn't train on this)
        * Scores jumping up by more than one (ref mistake OR card given, we shouldn't train on this)
        * Both fencers scoring a point (ref mistake OR card given, we shouldn't train on this)
        * Low confidence score detection (don't want to train on potentially incorrect data)
      * Also importantly: crop away the end of the clip based on the difference between when the light went off and when the ref increased the score
        * Will improve performance because there's less redundant frames
        * Also throw out ones where the time difference between the lights going off and the score incrementing is very large, since it may be something weird
      * Statistics:
        * 84.71% of clips were nominal (AKA nothing weird happened), rest were labeled not nominal and thrown out
        * Shaved off an average of 2.65 seconds from the end of clips (useless info since fencing stops in this time)
        * We totaled 10,600 clips to train on
          * With 625 total videos, that's around 17 clips per video
3) Cut up the video into clips based on the previous step (`3_cut_clips.py`)
    * Also flip the videos horizontally to increase the number of datapoints
      * NOTE: currently this is not implemented as I cannot figure out how to get ffmpeg to use the h264 codec on hopper (if I flip the video, I need to use an actual codec, not just streamcopy)
        * ALSO NOTE: somehow I did flip all the videos??? I guess I forgot I did that?
          I uncommented some lines in `3_cut_clips.py` to reflect this but I'm not gonna run the code again so no guarantee that file works as is. Check previous commits for a working one that doesn't have the flipping.
          * My thinking is I somehow accidentally ran this in the background of my desktop computer, which does all the codec stuff fine, just a little slowly. I probably forgot I ran it and forgot to stop running it overnight. Ended up transferring all the clips to hopper using GLOBUS which was actually pretty fast (~3.5 min).
          * Now, our clip count is up to 21,200 (not sure why its not an exact x2 of the original count but I'm not complaining too much)
    * This step (with only the copy encoding, no flipping) took only 30 seconds on hopper! 64 cores
4) Downsample the clips (`4_downsample.py`)
    * Specifically, downsample the first part, but keep original framerate for the last second or so
      * The last part of the clip contains a lot of bladework which is useful for determining who gets the point.
      The rest of the clip is important too, but can be at a lower framerate and still make sense
    * Again, couldn't use complex filtergraphs on hopper, so I did this on my PC and just file transferred it to hopper through GLOBUS
      * ~7 mins to process, ~3.75 mins to transfer
    * NOTE: the ffmpeg command from this file is HEAVILY inspired (read: basically copied) from the reference project since I don't know ffmpeg commands that well. Full credit to Sholto Douglas
5) Overlay optical flow onto the clips (`5_overlay_flow.py`)
    * I dreaded this part because, again, I can't do it on hopper (due to the codec stuff).
      It also seems like more core would've helped speed this up!!!
      ETA on my destop PC: 3+ hours... I guess I can use the time to get started on my report/presentation
      * Transfer time to hopper: ~4 min
    * Also, in the previous step I accidentally encoded the videos in the original FPS, so although they are downsampled, it just speeds up the end instead of slowing down the beginning
      * This wouldn't matter for the model since it is fed frame-by-frame anyway, but it just makes manually watching the videos take longer
      * So, I fix that in this step right before calculating the optical flow, so now the beginning of clips is fast whereas the end is normal speed, making watching the clips go faster for manual checks
6) Use Inception-v4 model to turn videos into feature-vector sequences (`6_videos_to_features.py`)
    * Manually removed a couple vectors for being too small (not enough frames), brought the count down to 21,194 clips
    * Finally a step I can use hopper for again, took around 3 hours
    * Also created a pytorch `Dataset` class (`6.8_dataloader.py`)
7) The magnum opus: actually classify all the vectors into left (1) or right (0) with an RNN (`7_classify.py`)
    * The RNN is actually in (`6.9_rnn.py`)
      * Justification is that `7` should depend on smaller numbers lol
      * First architecture: LSTM + Multi-layer RNN (just like Douglas's original idea)

