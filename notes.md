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