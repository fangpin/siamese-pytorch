# Training Chapter

## Module boundary

`train.py` owns runtime orchestration. It is responsible for reading flags, constructing datasets and loaders, creating the model, optimizing it, evaluating precision, and saving artifacts.

## Main classes and functions

- `gflags.DEFINE_*`
  - declares runtime configuration such as paths, batch size, learning rate, evaluation cadence, and GPU ids
- `OmniglotTrain(...)` / `OmniglotTest(...)`
  - instantiate the data sources used by the loaders
- `DataLoader(...)`
  - wraps datasets for batched iteration
- `Siamese()`
  - constructs the model
- `torch.nn.BCEWithLogitsLoss`
  - defines the binary similarity loss
- `torch.optim.Adam`
  - updates the model parameters
- `torch.nn.DataParallel`
  - optionally wraps the model for multiple GPUs

## Control flow

1. parse flags
2. configure augmentation and visible GPUs
3. create training and test datasets
4. create dataloaders
5. build the model and optionally wrap it with `DataParallel`
6. run the training loop
7. periodically print loss, save checkpoints, and evaluate precision
8. persist sampled loss history and report final averaged accuracy

## Data flow

- input paths: `train_path`, `test_path`
- runtime batches: image pairs and labels from dataloaders
- model output: logits
- loss path: logits plus labels into `BCEWithLogitsLoss`
- evaluation path: per-episode logits into `np.argmax`
- artifacts: checkpoint files and `train_loss` pickle

## Design tradeoffs

- `gflags` keeps the script configurable without introducing a larger config system
- Adam is simpler to run than the paper's tuned SGD regime but contributes to metric differences
- checkpointing and evaluation are time-step based rather than wall-clock or epoch based, which matches the synthetic endless-pair loader design

## Current limitations

- the script mixes configuration, training, evaluation, and artifact writing in one file
- no resume-from-checkpoint path
- no explicit separation between train and eval helpers for easier testing
- final accuracy is aggregated from a rolling queue rather than a standalone evaluation command
