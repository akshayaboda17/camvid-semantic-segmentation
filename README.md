# Semantic Segmentation of Urban Driving Scenes

## Project Overview

This project investigates semantic segmentation of urban driving scenes using the CamVid dataset.

The objective is to assign a semantic class to every pixel in an input road-scene image.

## Dataset

Dataset: CamVid
Dataset protocol: 11 semantic classes

Repository:
https://github.com/lih627/CamVid

Verified split-file counts:
- Training: 367 images
- Validation: 101 images
- Testing: 233 images
- Total: 701 images

The dataset is not stored in this repository. It must be downloaded separately.

## Proposed Models

- Traditional baseline: Random Forest pixel classifier
- Deep-learning model 1: U-Net++
- Deep-learning model 2: PSPNet

## Planned Experiments

- Five individual segmentation loss functions
- Composite loss optimization
- Class-aware crop sampling
- Agent-based parameter selection
- Ablation and final evaluation

## Technology

- Python 3.10
- PyTorch
- Segmentation Models PyTorch
- Scikit-learn
- OpenCV
- Google Colab GPU

## Project Structure

- configs/: Experiment configurations and dataset splits
- data/: Dataset files (not committed)
- notebooks/: Dataset inspection and result analysis
- src/: Source code
- results/: Metrics, plots and experiment logs
- reports/: Project documentation

## Current Status

Initial research, environment setup and CamVid dataset acquisition completed.

Implementation and experiments are in progress.
