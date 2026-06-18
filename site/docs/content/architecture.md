# Architecture Chapter

## Module boundary

`model.py` owns the entire similarity model. There is one public class, `Siamese`, and two key methods, `forward_one` and `forward`.

## Main classes and functions

- `Siamese.__init__`
  - defines `self.conv`, `self.liner`, and `self.out`
- `Siamese.forward_one`
  - runs one image through the shared feature extractor and projection path
- `Siamese.forward`
  - computes two embeddings, takes the absolute difference, and returns the final logit

## Control flow

1. each input image enters the same convolution stack
2. the feature map is flattened
3. the flattened vector is projected to a 4096-dimensional embedding
4. the two embeddings are compared by element-wise absolute difference
5. the difference vector is mapped to one scalar logit

## Tensor flow

- input shape: grayscale image tensor
- conv stack output: spatial feature map
- flattened vector: `9216`
- embedding vector: `4096`
- pairwise comparison: `abs(out1 - out2)`
- output: one logit for `BCEWithLogitsLoss`

## Design tradeoffs

- weight sharing keeps both branches in one embedding space and matches the canonical Siamese pattern
- the model keeps everything in one file, which is easy to inspect in a small repo but reduces modular test boundaries
- the output head returns raw logits instead of probabilities, which is correct for `BCEWithLogitsLoss` but means probability semantics live outside the model

## Current limitations

- no explicit abstraction boundary between encoder and matching head
- no separate inference helper for probability output
- architecture choices are compact but not parameterized for experimentation inside the model file
