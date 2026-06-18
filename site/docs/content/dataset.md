# Dataset Chapter

## Module boundary

`mydataset.py` owns all dataset logic. It defines `OmniglotTrain` for optimization-time pair sampling and `OmniglotTest` for one-shot evaluation episodes.

## Main classes and functions

- `OmniglotTrain.__init__`
  - loads the training split into memory
- `OmniglotTrain.loadToMem`
  - walks the directory tree and expands classes with four rotations
- `OmniglotTrain.__getitem__`
  - emits either a positive pair or a negative pair plus a float label
- `OmniglotTest.__init__`
  - loads the evaluation split into memory and stores `times` and `way`
- `OmniglotTest.loadToMem`
  - walks the evaluation directory and caches grayscale images
- `OmniglotTest.__getitem__`
  - builds one episode as one true pair plus distractors

## Control flow

### Training data

1. preload all classes into memory
2. rotate every class by `0`, `90`, `180`, `270`
3. on odd indices, sample same-class pairs
4. on even indices, sample different-class pairs
5. convert to tensors and emit a float label

### Test data

1. choose an anchor class at the start of each episode
2. emit one true pair at episode index `0`
3. emit `way - 1` distractor pairs for the rest of the episode
4. let the training loop find the highest logit within the episode

## Data flow

- raw source: Omniglot image files grouped by alphabet and character
- in-memory representation: dictionaries from class id to PIL images
- train output: `(image1, image2, label)`
- test output: `(img1, img2)` with episode semantics implied by position

## Design tradeoffs

- full in-memory loading reduces disk churn but increases memory usage
- rotation expansion is simple and effective for Omniglot but is hard-coded rather than configurable
- the test episode contract is positional, which keeps the implementation short but couples evaluation correctness to loader ordering

## Current limitations

- no lazy loading or streaming mode
- no explicit episode object for test-time semantics
- no file-level validation for malformed Omniglot directories
