# Siamese Project Docs

## Overview

This repository is a compact PyTorch reimplementation of *Siamese Networks for One-Shot Learning* on Omniglot. The homepage gives the quick signal. The docs area exists to explain how the implementation is actually organized in source code.

## Chapter map

- `architecture.html`: how `model.py` builds the shared encoder, the embedding path, and the final similarity score
- `dataset.html`: how `mydataset.py` loads Omniglot into memory, samples positive and negative pairs, and constructs one-shot episodes
- `training.html`: how `train.py` wires flags, dataloaders, optimization, checkpointing, and periodic evaluation

## How to read the system

The repo is small enough that the implementation boundaries match the top-level Python files:

1. `model.py` defines the model function
2. `mydataset.py` defines the data contract
3. `train.py` defines runtime orchestration

The docs therefore follow those same ownership boundaries rather than introducing a parallel information architecture.

## Verified project facts

- dataset: Omniglot
- framework: PyTorch
- evaluation mode: 20-way one-shot matching
- reported final accuracy: about 89.5%
- proof artifact: `loss.png`
