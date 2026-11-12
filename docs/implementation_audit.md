# Implementation Audit: Paper vs. Uploaded Code

This audit compares the uploaded Python source files with the uploaded IEEE UV2022 paper. It is intended to prevent the repository README and future resume descriptions from claiming that the supplied scripts reproduce components that are not present in the code snapshot.

## 1. Important architecture discrepancy

| Published paper | Uploaded code snapshot |
|---|---|
| CNN image-level classifier followed by U-Net++ segmentation | U-Net segmentation model only |
| Leather-to-carpet transfer-learning pipeline | No explicit source-domain pretraining / target-domain transfer step is visible in the supplied `train.py` |
| Classification threshold 0.3 | No CNN classifier or classification threshold |
| U-Net++ segmentation | Standard U-Net with four downsampling blocks and four upsampling blocks |
| Paper input resolution 256 × 256 | `dataset.py` resizes to 256 then random-crops to 224 × 224 |

`unet_model.py` defines a conventional U-Net with `DoubleConv`, `Down`, `Up`, and `OutConv` blocks. It is not a U-Net++ nested skip architecture. The current source files should therefore be documented as a U-Net segmentation implementation or training experiment, not as the complete published two-stage model.

## 2. Paper experiment settings vs. code settings

| Item | Paper | Uploaded scripts |
|---|---|---|
| Dataset split | 7:2:1 train/validation/test | 5-fold split from a CSV manifest |
| Augmentation | Flipping, rotation, cropping, color-shading adjustment; 10× expansion | Resize + RandomCrop in `dataset.py`; no complete paper augmentation pipeline visible |
| Segmentation architecture | U-Net++ in proposed model; U-Net baseline | Standard U-Net |
| Optimizer | Adam | RMSprop |
| Learning rate | 0.001 in paper's experiment description | 0.00001 |
| Segmentation batch size | 4 for U-Net++ | 6 |
| Epochs | 100 direct-training; 50 per stage in transfer-learning scenario | 100 |
| Main paper metrics | mIoU and Pixel Accuracy | Training script attempts an accuracy calculation; no mIoU implementation is present in supplied files |

Do not combine these settings into one experiment configuration. They describe different artifacts.

## 3. Issues to fix before claiming reproducible training results

### 3.1 Image and mask random crops are not synchronized

In `dataset.py`, the same `RandomCrop` transform pipeline is called independently on the image and its mask. Each call samples a new crop location. This can misalign the input image and segmentation label.

**Recommended fix:** apply paired geometric transformations using the same sampled parameters to both image and mask. Use bilinear interpolation for the image and nearest-neighbor interpolation for the mask. Avoid applying color transforms to the mask.

### 3.2 `argmax` is not a valid binary metric for a one-channel output

`unet_model.py` is instantiated with `n_classes=1`. In `train.py`, the metric uses `y_hat.argmax(dim=1)`. Since the channel dimension has length one, `argmax` always returns zero. This does not calculate binary pixel accuracy.

**Recommended fix:** convert logits to probabilities with `sigmoid`, threshold probabilities, and compare the resulting binary mask with the ground truth. Calculate TP, TN, FP, and FN over the intended evaluation population. Add mIoU and Dice if those are the reported metrics.

### 3.3 Validation uses random cropping

The validation branch in `dataset.py` also uses `RandomCrop(224)`. This means validation examples can change spatial content between epochs.

**Recommended fix:** use deterministic resizing/center cropping for validation and test, with the same image-mask geometry.

### 3.4 Prediction threshold is applied to logits

`predict.py` thresholds the raw model output at `0.05`. The training loss applies sigmoid internally, so the output of the model is logits. A probability threshold should be applied after sigmoid.

**Recommended fix:** use `probabilities = torch.sigmoid(logits)` and select a documented probability threshold on the validation set before applying it to the held-out test set.

### 3.5 Checkpoint selection and evaluation protocol

The training script selects the best checkpoint using a custom validation accuracy calculation. Because the current `argmax` metric is invalid for a one-channel binary segmentation output, checkpoint selection is also unreliable.

**Recommended fix:** first correct the metric implementation; select checkpoints on a validation set; keep the final test set untouched until final evaluation.

### 3.6 Cross-validation split details

`k_fold.py` divides the CSV rows into contiguous blocks and does not shuffle the rows internally. It assumes the manifest has already been shuffled. It also writes fixed filenames (`train_k.txt`, `test_k.txt`) to the current working directory.

**Recommended fix:** shuffle deterministically before fold assignment, use a standard K-fold splitter, preserve class balance where feasible, and save fold manifests in a dedicated directory. If multiple images originate from the same physical sample or acquisition sequence, split by sample/group to avoid leakage.

### 3.7 Hard-coded machine-specific paths

`predict.py` contains absolute Windows paths to a checkpoint and dataset. These paths prevent execution on another machine.

**Recommended fix:** move dataset, checkpoint, output, and inference settings into command-line arguments or a configuration file.

### 3.8 Preprocessing consistency

The training path uses `Resize(256)` followed by `RandomCrop(224)` and `ToTensor()`. The prediction path uses `Resize(224)` and `ToTensor()`. The spatial transforms differ, and the training script's image/mask crop alignment also needs correction.

**Recommended fix:** define shared, explicit preprocessing configurations for training, validation, and inference. Document the normalization and interpolation behavior.

## 4. Additional code hygiene

- `data_loader.py` contains unused imports and several preprocessing helpers that are not used by the supplied `train.py` pipeline.
- `RescaleT` computes `new_h` and `new_w` but resizes using `self.output_size` for both dimensions; this helper should be corrected if it is used.
- The training script creates `best_loss` but does not use it.
- The training script's printed labels and final summary include RMSE terminology even though the task is segmentation and the loss is Soft Dice.
- Ensure the output `results/` directory exists before saving checkpoints.
- Close manifest files with context managers and validate that every image path has a corresponding mask path.
- Pin tested package versions in `requirements.txt` after confirming the environment used for successful execution.

## 5. Suggested maintenance order

1. Keep the published paper and current code as separate, clearly identified artifacts.
2. Correct paired image-mask augmentation and deterministic validation transforms.
3. Correct binary metric computation and probability thresholding.
4. Replace hard-coded paths with configuration/CLI arguments.
5. Add a reproducible dataset manifest and deterministic split procedure.
6. Implement or recover the missing CNN classifier and U-Net++ code only if the original source is available.
7. Re-run evaluation and publish new results separately from the paper's reported results.
