# Phase 4 Segmentation Model Summary (In Progress)

## Objective
Build and train a CNN-based segmentation model that locates buildings in pre-disaster satellite imagery, producing the building-location foundation Phase 5 will use to guide damage classification.

## 4.1 — CNN & Segmentation Fundamentals (Concept)
- **Convolution:** a small learned kernel slides across the image, producing a feature map highlighting where a pattern appears — the same kernel is reused everywhere, letting the model recognize a pattern regardless of position.
- **Pooling:** max pooling shrinks feature maps by keeping the strongest value in small regions, reducing computation and adding positional tolerance.
- **Encoder:** stacked (convolution → pooling) rounds shrink spatial size while growing feature richness — compressing toward understanding, at the cost of precise pixel-location detail.
- **Decoder:** reverses the encoder, upsampling back toward the original size to produce a full pixel-by-pixel output mask.
- **Skip connections (U-Net's defining feature):** encoder feature maps are copied directly to the matching decoder stage before further shrinking, giving the decoder both deep understanding and precise spatial detail — preventing a blurry output.

## 4.2 — U-Net Architecture (Built & Verified)
Built incrementally in PyTorch (`src/unet.py`), verifying each component's output shape against hand-calculated expectations before assembly:

| Stage | Shape (base_channels=64) |
|---|---|
| Input | (1, 3, 512, 512) |
| After `block1`/`block2`/`block3` (pooled) | (1, 64, 256, 256) → (1, 128, 128, 128) → (1, 256, 64, 64) |
| After `bottleneck` | (1, 512, 64, 64) |
| After `decoder_block3`/`2`/`1` | (1, 256, 128, 128) → (1, 128, 256, 256) → (1, 64, 512, 512) |
| After `final_conv` (assembled `UNet`) | (1, 1, 512, 512) |

Parameterized by `base_channels` (default 64) so a lighter comparison variant can be created for Phase 4.5 without duplicating the class.

## 4.3 — Loss Function, Optimizer, and Training Loop
- **Loss: `BCEWithLogitsLoss`** — matches the model's raw-logit output (unbounded values from `final_conv`) and the binary per-pixel ground truth; applies sigmoid + BCE in one numerically stable step.
- **Optimizer: `Adam`**, `lr=1e-4`, given `model.parameters()`.
- **Training loop, per batch:** `optimizer.zero_grad()` → `model(images)` → `criterion(prediction, mask)` → `loss.backward()` → `optimizer.step()`.

**Bug found and fixed:** `SegmentationDataset.__getitem__` returned PIL Image objects, not tensors. Unnoticed in Phase 3.3's single-item testing (`dataset[0]` doesn't require batching); surfaced when `DataLoader` tried to batch multiple PIL Images together (`TypeError: default_collate: batch must contain tensors...`). Fixed by adding `torchvision.transforms.ToTensor()` as the final step in `__getitem__`, after resizing and augmentation.

**Verification results:**
- First-batch loss: 0.7508 — close to the theoretical random-guessing baseline for BCE (≈0.693, i.e. -ln(0.5)), confirming the full pipeline (Dataset → DataLoader → model → loss → backprop → optimizer) is correctly wired, not just error-free.
- Measured per-batch time: ~2 minutes (5 batches in ~10 minutes) on local CPU.
- Projected: ~3 hours per epoch, ~90 hours for a full ~30-epoch run — confirms local CPU training is infeasible for real experimentation, exactly the scenario anticipated in `docs/scope_and_assumptions.md`.

**Decision:** local training stopped after confirming pipeline correctness (its actual purpose). Real training moves to free-tier Colab/Kaggle GPU in Phase 4.4. Training data for the 3 confirmed disasters (~2.5GB: images, labels, targets) plus `train_ids.txt`/`val_ids.txt`/`src/unet.py`/`src/segmentation_dataset.py` uploaded to a personal Google Drive account for Colab access — kept private, not linked publicly, per the dataset's licensing terms (the xBD dataset is not covered by this repo's MIT license).

## Remaining Phase 4 Tasks
- **4.4** — Training on Colab/Kaggle GPU
- **4.5** — Training multiple candidate segmentation architectures (multi-model comparison)
- **4.6** — Monitoring training/validation loss and overfitting
- **4.7** — Selecting the best segmentation model
- **4.8** — Saving trained model weights
- **4.9** — Visual sanity check on validation images

*This document will be updated as Phase 4 continues.*
