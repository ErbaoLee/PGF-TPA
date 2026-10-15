# Third-party material and attribution

## SolarBench / SKIPP'D data

Source: [solarbench/SKIPPD](https://huggingface.co/datasets/solarbench/SKIPPD).
Pinned experiment revision: `f51690de73d34cc409670ef6119fd69cc3aade8d`.
The upstream dataset card currently declares CC-BY-4.0. Credit the dataset
creators and SolarBench when using the data. Processed power/time metadata,
sample indices and derived predictions do not change the upstream data terms.
Original camera images are downloaded from upstream, not bundled in Git.

## Clear-sky reference function

`src/relative_op_func_reference.py` is the original attributed reference helper
with its `@author: ynie` and 2019 date retained. It is preserved for numerical
compatibility with the existing experiment caches. No separate license header
was present in the recovered copy; this notice does not assert that third-party
authorship or terms are replaced by the project GPL license. The precomputed
clear-sky arrays are included so weight inference can use the recorded inputs.

## Baseline implementations

The repository contains project adaptations inspired by SUNSET, ConvLSTM,
ResLSTM, Transformer, PatchTST, DLinear, CrossGNN and Sky-BiMamba. They run on
this repository's window and training protocols. They are not guaranteed to
be the original authors' released implementations or exact paper settings.
Architecture differences are documented in source docstrings and
`docs/PROTOCOL.md`; the smaller ResLSTM is not a torchvision ResNet-18.
Sky-BiMamba uses a project PyTorch selective-scan implementation with an
optional external mamba-ssm acceleration path. No external pretrained image
backbone is downloaded by the primary model.

## Project license

The existing repository's GNU GPL v3 text is retained in `LICENSE` and applies
to project-authored code and project-trained model weights. Third-party
attributions and upstream dataset terms remain applicable.
