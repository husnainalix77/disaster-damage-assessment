# Phase 4 Segmentation Model Summary (In Progress)

## Objective
Build and train a CNN-based segmentation model that locates buildings in pre-disaster satellite imagery, producing the building-location foundation Phase 5 will use to guide damage classification.

## 4.1 — CNN & Segmentation Fundamentals (Concept)
Documented before writing any model code:

- **Convolution:** a small learned kernel slides across the image, producing a feature map that highlights where a specific pattern appears. The same kernel is reused across every position, letting the model recognize a pattern (e.g. an edge) regardless of where it appears in the image.
- **Pooling:** max pooling shrinks feature maps by keeping the strongest value in small regions, reducing computation and adding tolerance to small positional shifts.
- **Encoder:** stacked (convolution → pooling) rounds progressively shrink spatial size while growing feature richness, compressing the image into a dense understanding — at the cost of precise pixel-location detail.
- **Decoder:** reverses the encoder, upsampling back toward the original size to produce a full pixel-by-pixel output mask (required for segmentation, unlike classification's single-label output).
- **Skip connections (U-Net's defining feature):** at each encoder stage, the feature map is copied across to the matching decoder stage before further shrinking occurs. This gives the decoder both the deep, compressed understanding AND the precise spatial detail that would otherwise be lost — producing a sharp, accurate mask instead of a blurry one.

## 4.2 — U-Net Architecture (Built & Verified)
Built incrementally in PyTorch, verifying each component's output shape against hand-calculated expectations on a dummy input before proceeding to the next.

**Components** (all in `src/unet.py`):
- `EncoderBlock(in_channels, out_channels)` — conv→relu→conv→relu→[save skip]→pool
- `Bottleneck(in_channels, out_channels)` — same conv pattern, no pooling, no skip (deepest point of the U)
- `DecoderBlock(in_channels, skip_channels, out_channels)` — upsample (`ConvTranspose2d`) → concatenate with matching skip connection (`torch.cat`, channel dimension) → conv→relu→conv→relu
- `UNet(base_channels=64)` — assembles 3 encoder blocks, 1 bottleneck, 3 decoder blocks, and a final 1×1 output convolution collapsing to a single channel

**Design decisions:**
- **Parameterized `base_channels`** (default 64, the standard convention from the original U-Net paper) rather than hardcoded channel counts, so a lighter variant (e.g. `base_channels=32`) can be created for the Phase 4.5 multi-architecture comparison by reusing the same class.
- **Verification-first construction** — each block was tested independently on a dummy tensor immediately after being written, confirming exact output shapes before assembly, rather than writing the full architecture and debugging shape errors afterward.

**Verified shape progression** (input 512×512×3, `base_channels=64`):

| Stage | Shape |
|---|---|
| Input | (1, 3, 512, 512) |
| After `block1` (pooled) | (1, 64, 256, 256) |
| After `block2` (pooled) | (1, 128, 128, 128) |
| After `block3` (pooled) | (1, 256, 64, 64) |
| After `bottleneck` | (1, 512, 64, 64) |
| After `decoder_block3` | (1, 256, 128, 128) |
| After `decoder_block2` | (1, 128, 256, 256) |
| After `decoder_block1` | (1, 64, 512, 512) |
| After `final_conv` (assembled `UNet`) | (1, 1, 512, 512) |

The assembled `UNet` class reproduces the exact same output shape as the manually-chained component test, confirming correctness after moving the code to `src/unet.py`.

## Remaining Phase 4 Tasks
- **4.3** — Loss function, optimizer, and training loop
- **4.4** — Training on Colab/Kaggle GPU (per hardware-scoping decision)
- **4.5** — Training multiple candidate segmentation architectures (per the multi-model comparison decision)
- **4.6** — Monitoring training/validation loss and overfitting
- **4.7** — Selecting the best segmentation model
- **4.8** — Saving trained model weights
- **4.9** — Visual sanity check on validation images

*This document will be updated as Phase 4 continues.*
