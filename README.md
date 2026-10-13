# SRNet E-NAS — new MATLAB reference implementation

**Status: implementation draft; not executed in MATLAB in the authoring environment.**
This package extends the supplied short training script with evolutionary
architecture search. It is newly written code, not a recovered copy of the code
that produced a published paper. No published accuracy or architecture is claimed
to be reproduced. Validate it on MATLAB and real data before a public release.

## Requirements and quick start

Target API: MATLAB R2023b, Deep Learning Toolbox, Global Optimization Toolbox,
Image Processing Toolbox, and Statistics and Machine Learning Toolbox
(`confusionmat`). A compatible GPU with Parallel Computing Toolbox is optional.
New MATLAB releases may change GA callback interfaces; validate before use.

Place prepared images in exactly two class folders under `data/`, with at least
20 images per class. Folder names become labels. Images are resized to 100 by
100 and converted to grayscale. There is no augmentation, no normalization
beyond the network's explicit `Normalization='none'`, and no ELPV label mapping
is assumed. Document your source data, ambiguous-label handling, and exclusions.
Dataset images are not distributed in this package.

```matlab
cd('path/to/SRNet_ENAS')
addpath('tests')
test_graphs
smoke_test
% First run with real images, tiny search and one epoch:
r = srnet_enas('D:\my_data\data','results_smoke','smoke');
% Only after successful checks: expensive full search and final fitting.
r = srnet_enas('D:\my_data\data','results_full','full');
```

Output folders must not already exist, preventing accidental overwrite. Full
mode can take days or longer and large candidates can exhaust GPU memory.
No hardware-specific runtime has been measured. Errors stop execution rather
than being silently converted to bad fitness scores. Set batch size and execution
environment in `enas_config.m`; report any changed configuration.

## Search and evaluation

The solver is MATLAB `ga`, with 10 integer variables (`IntCon=1:10`), population
20, at most 50 generations, tournament selection of size 2, single-point
crossover fraction 0.8, one elite, tolerance 1e-6 and stall limit 50.
Both crossover offspring and mutation-only offspring receive two-stage mutation:
individual probability 0.1, then per-gene probability 0.01; a selected gene is
uniformly reassigned within its integer bounds. The replacement may equal its
previous value. The elite is not mutated. GA's integer feasibility machinery
still applies. These callback details are explicitly part of this implementation.

Each candidate is trained from scratch with Adam, learning rate 0.001, batch
size 128 and 50 epochs, without early stopping. Fitness uses last-epoch weights:

`fitness = 0.85 * (1 - validation_accuracy) + 0.15 * (block_count / 160)`

The complexity term is a block-count proxy, not parameter count, FLOPs, memory,
or latency. Repeated identical chromosomes reuse fitness within the current run.
Candidate fitting saves/restores the CPU random stream to avoid changing GA
randomness. GPU training is not guaranteed bitwise reproducible.

The split is stratified and randomized: 20% held-out test, then 10% of remaining
development images for search validation (approximately 72/8/20 overall).
All candidates use the same split. After search, the selected graph is initialized
again and trained on all development images for 500 fixed epochs; the held-out
test set is evaluated only afterwards. Split rounding is per class.

This is **a holdout workflow, not 10-fold cross-validation**. Image-level random
splitting does not ensure panel/module/subject independence. For grouped data,
replace splitting with a verified group-disjoint protocol before reporting
generalization. Do not repeatedly tune against held-out test results. To report
10-fold results, architecture search must be confined to each outer training fold
or use an independently fixed architecture; that workflow is not supplied here.

## Exact chromosome and newly defined search space

Historical sequence catalogues and connectivity maps were not provided.
The following finite catalogues are explicit implementation choices and must not
be described as the original paper's exact search space.

| Gene | Meaning and catalogue |
|---|---|
| x1 | Stack count: 1–8 |
| x2 | Block sequence index: constant rows 1–20; extra row [4,3,2,2,2,2,2,2] |
| x3 | Intermediate channels: constant rows 16:16:128; extra [16,32,64,64,64,64,64,64] |
| x4 | Output channels: constant rows 16:16:256; extra [32,64,128,128,128,128,128,128] |
| x5–x7 | Separate kernel sequence indices for conv1, conv2, conv3: constant rows 1–11 |
| x8 | Stack-entry stride: constant rows 1–5; extra [1,2,2,1,1,1,1,1] |
| x9 | Block residuals: 1 off, 2 on, shared across all blocks |
| x10 | Stack-spanning residuals: 1 off, 2 on, shared across all stacks |

Only the first x1 entries of each row are used. Kernels are square k-by-k.
This finite catalogue does **not** enumerate all possible per-stack sequences;
extend the catalogue if needed and record the change. Stack links are local
stack-input-to-stack-output skips; arbitrary cross-stack topology is not searched.
Main-path stack connections remain sequential. A 1x1 projection aligns channels
and spatial dimensions whenever necessary. Each bottleneck has three convolutions,
batch normalization, intermediate ReLUs and output ReLU. The stem has 16 channels;
the head uses global average pooling and a two-class classifier.

## Outputs

- `configuration.mat`: complete settings, codebooks, toolbox versions.
- `split.csv`: exact paths, labels and assigned partitions (may contain local paths).
- `candidate_log.csv`: chromosome, validation accuracy, blocks, fitness, elapsed time.
- `best_candidate.mat`: best architecture observed so far, not trained weights.
- `ga_checkpoint.mat`: GA state and CPU RNG snapshot for inspection. Automatic
  resume is not implemented; cache and candidate weights are not checkpointed.
- `search_result.mat`: selected chromosome, decoded architecture and solver output.
- `final_model.mat`: final trained network, training info, test metrics and settings.
- `test_predictions.mat`: labels, predictions, probabilities and their class order.

Do not upload dataset files, local path manifests or trained outputs accidentally.

## Relationship to the input script

The fixed `resnetLayers` construction is replaced by a graph constructed from
the chromosome. Conflicting hard-coded stack definitions are removed. Adam,
0.001 learning rate, 128 batch size and 500 final epochs follow the supplied
script; the new search budget is separate. The original input remains unchanged
outside this package. No trained weights, historical splits, GA logs, original
codebooks, pretrained-model comparisons or published performance are supplied.

## Before GitHub / Zenodo release

1. Run both MATLAB checks and a real-data smoke run; inspect saved graphs/results.
2. Resolve the historical-method differences above and document your actual run.
3. Confirm code authors, affiliations, ORCIDs, title and copyright ownership;
   choose a software license. No license or author metadata is fabricated here.
4. Add `LICENSE` and a completed `CITATION.cff`. Do not insert a fictitious DOI.
5. Upload source to GitHub, connect the repository in Zenodo, then publish a
   release such as `v0.1.0`. Cite the DOI Zenodo actually assigns.

Official references:
- https://www.mathworks.com/help/gads/genetic-algorithm-options.html
- https://www.mathworks.com/help/gads/mixed-integer-optimization.html
- https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layergraph.connectlayers.html
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
