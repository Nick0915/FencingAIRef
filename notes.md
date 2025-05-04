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
  * This serves as inspiration, but we take different approaches

0) Prepare workspace organization with a script (`0_prepare.py`)
1) Download videos from YouTube (`1_download_data.py`)
  * We already have a list of sabre videos of the same format, so we'll start with this for now
  * Used a multiprocessing pool to download many videos in parallel (100 videos in ~ 1m20s)
2) Label keyframes in video where the score changes, and what the score is at that point (`2_label_vids.py`)
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
  * This algorithm assumes that scores only ever increase monotonicly (ideally, they do), but they may jump around weirdly in the video
    * Due to ref mistakes, tech issues, cards, overturning points, etc.
    * So we are losing some potentially useful training points, but it's pretty uncommon so not a big deal
  * Finally, check the validity of these labels (in `2.1_check_labels.py`)
    * Fix up some stuff with the csv formatting
    * Importantly: added `nominal` column which is true if that score change made sense. We want to weed out the following:
      * Scores going down (ref mistake, we shouldn't train on this)
      * Scores jumping up by more than one (ref mistake OR card given, we shouldn't train on this)
      * Both fencers scoring a point (ref mistake OR card given, we shouldn't train on this)
      * Low confidence score detection (don't want to train on potentially incorrect data)
3) Use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose?tab=readme-ov-file)'s 2D multi-person estimation to get keypoints for fencers
