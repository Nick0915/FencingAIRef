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

## Beginning
* Started by [forking](https://github.com/Nick0915/FencingAIRef) and cloning the [original repo](https://github.com/sholtodouglas/fencing-AI)
* First order of business is to clean everything up and convert from Python 2 to Python 3
  0) Prepare workspace organization with a script
  1) Download videos from YouTube
    * We already have a list of sabre videos of the same format, so we'll start with this for now
    * Used a multiprocessing pool to download many videos in parallel (100 videos in ~ 1m20s)
  2) Label keyframes in video where the score changes, and what the score is at that point
    * First, assume we have a function that can spit out the score based on reading one frame
    * We develop an algorithm that binary-searches for the exact frame these scores change1
    * Then, implement the assumed function using an off-the-shelf huggingface model trained on MNIST digits
    * References:
      * https://pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/
      * https://huggingface.co/prithivMLmods/Mnist-Digits-SigLIP2
    * With this naive algorithm (minimal changes), I can label 20 videos in 25 minutes (data set is on the order of ~1000 videos)
      * Would take >20 hours to label everything