# Semantic Segmentation for Urban Scene Understanding

A CamVid-based comparison of a traditional pixel classifier and two deep-learning segmentation architectures: Random Forest, U-Net++, and PSPNet.

## 1. Project overview

Semantic segmentation assigns a semantic category to every pixel in an image. In urban-driving scenes, this allows a system to distinguish road, sidewalk, buildings, vehicles, pedestrians, signs, and other scene elements.

This project evaluates three approaches on the 11-class CamVid protocol:

- **Traditional baseline:** Random Forest using RGB and pixel-position features.
- **Deep model 1:** U-Net++ with an ImageNet-pretrained ResNet18 encoder.
- **Deep model 2:** PSPNet with an ImageNet-pretrained ResNet18 encoder.

The study compares individual loss functions, a weighted composite loss, data-sampling changes, computational cost, and final held-out test performance.

## 2. Dataset

- **Dataset:** CamVid
- **Dataset source:** [lih627/CamVid](https://github.com/lih627/CamVid)
- **Original image dimensions:** 960 × 720 pixels
- **Input dimensions used by the models:** 192 × 256 pixels (height × width)
- **Total image-mask pairs:** 701

| Split | Images |
|---|---:|
| Training | 367 |
| Validation | 101 |
| Test | 233 |
| **Total** | **701** |

The dataset audit found all 701 listed image-mask pairs valid, with no missing images, missing masks, dimension mismatches, invalid class IDs, or image overlap between splits.

### Class IDs

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | Sky | 6 | SignSymbol |
| 1 | Building | 7 | Fence |
| 2 | Pole | 8 | Car |
| 3 | Road | 9 | Pedestrian |
| 4 | Sidewalk | 10 | Bicyclist |
| 5 | Tree | 255 | Ignore / void |

Label `255` is excluded from the semantic-class metrics and training loss where the loss implementation supports the ignore index.

### Dataset location

The code's default dataset root is:

```text
data/camvid_source/
```

Place the CamVid dataset at that path. The dataset root must contain the split text files and the image/mask directories expected by `src/datasets/camvid.py`. Alternatively, pass a different location using `--dataset-root` to the training, inference, or audit-related code where that option is available.

The dataset itself is not committed to this repository. Download it from the source above and check its usage terms before redistribution.

## 3. Methodology

### 3.1 Preprocessing

- RGB images are resized to 192 × 256 using bilinear interpolation.
- Segmentation masks are resized using nearest-neighbour interpolation so that integer class IDs remain unchanged.
- Deep-learning inputs are converted to tensors and normalized using ImageNet channel statistics.
- Training augmentation includes horizontal flipping and modest brightness, contrast, and saturation changes.
- Validation and test data use the non-augmented pipeline.

The optional crop modes are `none`, `random`, and `class_aware`. Cropping is applied only when training augmentation is enabled; validation and test images remain uncropped. The crop size is specified as height and width, with a default of 360 × 480 in the original-image coordinate space.

### 3.2 Traditional baseline: Random Forest

The Random Forest baseline predicts a class for each pixel using five features:

1. Red channel
2. Green channel
3. Blue channel
4. Normalized horizontal coordinate
5. Normalized vertical coordinate

The images and masks are resized to 192 × 256. Training pixels are sampled by class to limit memory use and reduce the effect of the original class imbalance. The implementation samples at most 100 pixels per class per training image and caps the total training sample at 5,000 pixels per class.

| Parameter | Value |
|---|---:|
| Trees (`n_estimators`) | 100 |
| Maximum depth | 18 |
| Minimum samples per leaf | 2 |
| Maximum features | `sqrt` |
| Bootstrap | True |
| Random seed | 42 |

The baseline training script uses the training split and evaluates on validation only. The final test metrics reported later come from the separate held-out test evaluation. The saved `.joblib` model is a local artifact and is excluded from Git.

### 3.3 Deep-learning architectures

Both deep models use an ImageNet-pretrained ResNet18 encoder, 3 input channels, and 11 output classes.

| Model | Configuration |
|---|---|
| U-Net++ | SMP `UnetPlusPlus`, encoder depth 5 |
| PSPNet | SMP `PSPNet`, encoder depth 3, upsampling 8, dropout 0.2 |

The models produce per-pixel class logits. The training pipeline uses the same dataset splits, input dimensions, optimizer family, batch size, learning rate, and epoch budget for the comparative experiments.

### 3.4 Training configuration

The main loss-comparison experiments use:

| Setting | Value |
|---|---:|
| Input size | 192 × 256 |
| Batch size | 4 |
| Epochs | 10 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Mixed precision | FP16 where supported |
| Individual-loss experiment seed | 42 |

A `ReduceLROnPlateau` scheduler is used by the training pipeline. Final composite configurations were also evaluated using seeds 7 and 21 in addition to seed 42.

### 3.5 Loss functions

Five individual losses were evaluated for each deep architecture:

- **Cross Entropy:** pixel-wise multiclass classification loss.
- **Dice:** overlap-oriented loss that directly encourages better region agreement.
- **Focal:** reduces the relative contribution of well-classified examples and emphasizes harder examples.
- **Lovasz:** a surrogate objective designed to optimize intersection-over-union.
- **Boundary:** emphasizes segmentation-boundary agreement.

A weighted composite loss combines Cross Entropy, Dice, and Lovasz:

\[
L_{composite}=w_{CE}L_{CE}+w_{Dice}L_{Dice}+w_{Lovasz}L_{Lovasz}
\]

where the three non-negative weights sum to one.

### 3.6 Agent-based composite-weight search

The weight-search script selects composite-loss weights using validation mIoU as its objective. The recorded search used five trials, three epochs per trial, and a fixed compute budget of 15 model-epochs for each architecture. It includes an equal-weight baseline and seeded Dirichlet random-search candidates. The test split is not used for weight selection.

Selected weights:

| Model | Cross Entropy | Dice | Lovasz |
|---|---:|---:|---:|
| U-Net++ | 0.204267 | 0.636507 | 0.159226 |
| PSPNet | 0.333333 | 0.333333 | 0.333334 |

The selected PSPNet weights are effectively equal weights. The selected U-Net++ weights are the best candidate in its recorded search.

## 4. Evaluation metrics

The project reports:

- **Mean Intersection over Union (mIoU)**
- **Mean Dice / F1**
- **Mean precision**
- **Mean recall**
- **Pixel accuracy**
- **Boundary F1**
- Per-class IoU and confusion matrices

The boundary metric uses a two-pixel spatial tolerance at the evaluated 256 × 192 resolution in the Random Forest evaluation implementation. The `255` ignore label is not treated as one of the 11 semantic classes.

## 5. Experimental results

### 5.1 Individual-loss validation experiments

All individual-loss experiments used seed 42 and the same 10-epoch training configuration described above. Results below are validation results, not held-out test results.

| Model | Loss | mIoU (%) | Mean Dice (%) | Boundary F1 (%) |
|---|---|---:|---:|---:|
| U-Net++ | Cross Entropy | 53.00 | 60.67 | 47.65 |
| U-Net++ | Dice | **64.98** | **74.49** | **60.21** |
| U-Net++ | Focal | 54.81 | 63.24 | 51.86 |
| U-Net++ | Lovasz | 63.31 | 73.33 | 58.47 |
| U-Net++ | Boundary | 36.84 | 40.60 | 21.36 |
| PSPNet | Cross Entropy | 53.63 | 63.30 | 45.76 |
| PSPNet | Dice | 50.72 | 59.53 | 36.68 |
| PSPNet | Focal | **54.83** | 64.84 | **46.64** |
| PSPNet | Lovasz | 54.36 | **65.15** | 41.45 |
| PSPNet | Boundary | 17.02 | 20.84 | 6.15 |

The best individual-loss validation mIoU was achieved by Dice loss for U-Net++ and Focal loss for PSPNet. Boundary loss performed poorly in these experiments; this result is retained rather than omitted.

### 5.2 Composite-loss validation results across three seeds

The final composite configuration was evaluated with seeds 7, 21, and 42. The table reports mean and **sample** standard deviation across the three runs. Values are percentages; standard deviations are percentage points.

| Metric | U-Net++ mean ± SD | PSPNet mean ± SD |
|---|---:|---:|
| mIoU | 59.35 ± 3.99 | 56.85 ± 1.47 |
| Mean Dice | 67.24 ± 5.34 | 67.19 ± 1.50 |
| Precision | 68.70 ± 5.93 | 68.77 ± 3.32 |
| Recall | 68.25 ± 5.38 | 69.89 ± 2.57 |
| Pixel accuracy | 91.55 ± 0.94 | 88.99 ± 0.90 |
| Boundary F1 | 53.42 ± 5.24 | 47.09 ± 0.75 |

For seed 42, the best recorded validation mIoU was 63.94% for U-Net++ and 56.03% for PSPNet. The three-seed means are the more appropriate summary of variability; the best single-seed value should not be presented as the expected performance.

### 5.3 Class-aware crop ablation

A class-aware crop experiment was performed for U-Net++ using the selected composite weights, seed 42, and a 360 × 480 crop size. Validation images were not cropped.

| Configuration | Validation mIoU (%) | Mean Dice (%) | Pixel accuracy (%) |
|---|---:|---:|---:|
| Full-image pipeline | 63.94 | 73.37 | 92.16 |
| Class-aware crop | 59.72 | 69.58 | 91.29 |

The crop experiment did not improve overall validation performance. It is reported as an unsuccessful ablation rather than a claimed improvement.

### 5.4 Final held-out test results

The following results are from the held-out test split. These results are for final evaluation and were not used for loss selection or composite-weight tuning.

| Model | mIoU (%) | Mean Dice (%) | Precision (%) | Recall (%) | Pixel accuracy (%) | Boundary F1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 26.37 | 35.52 | 36.39 | 42.54 | 58.47 | 26.26 |
| U-Net++ | **53.32** | **64.07** | 64.90 | **67.35** | **87.11** | **48.06** |
| PSPNet | 48.59 | 60.04 | **65.93** | 57.45 | 84.80 | 39.17 |

U-Net++ has the highest test mIoU, mean Dice, recall, pixel accuracy, and Boundary F1 among the three models. PSPNet has slightly higher mean precision. Both deep models outperform the Random Forest baseline on test mIoU.

### 5.5 Test-time augmentation

Horizontal-flip test-time augmentation (TTA) was also evaluated:

| Model | Standard test mIoU (%) | Horizontal-flip TTA mIoU (%) |
|---|---:|---:|
| U-Net++ | 53.32 | 54.91 |
| PSPNet | 48.59 | 48.51 |

TTA increased U-Net++ test mIoU by approximately 1.59 percentage points. It did not improve PSPNet in this run.

## 6. Computational analysis

Model complexity was estimated at an input size of 1 × 3 × 192 × 256 using THOP. The reported FLOPs use the convention 1 MAC = 2 FLOPs.

| Model | Parameters | MACs | Estimated FLOPs |
|---|---:|---:|---:|
| U-Net++ | 15,971,899 | 12.0913 G | 24.1826 G |
| PSPNet | 11,375,915 | 1.1272 G | 2.2544 G |

GPU inference was benchmarked on a Colab T4 using batch size 1, 20 warm-up passes, and 100 timed forward passes with CUDA synchronization.

| Model | Mean forward latency (ms/image) | Throughput (images/s) | Peak GPU memory |
|---|---:|---:|---:|
| U-Net++ | 12.568 | 79.57 | 296.40 MB |
| PSPNet | 5.202 | 192.25 | 104.06 MB |

These timings measure the deep-model GPU forward pass. The Random Forest test inference measurement (382.15 ms/image) includes CPU feature extraction and classifier prediction, so it is not a directly comparable hardware measurement.

## 7. Repository structure

```text
SemanticSegmentation/
├── data/
│   └── camvid_source/              # Local dataset; not committed
├── models/
│   └── random_forest_baseline.joblib  # Local model; ignored by Git
├── results/
│   ├── audit/                      # Dataset audit outputs
│   ├── experiments/                # Loss, composite, seed, ablation and test results
│   ├── preprocessing/              # Pipeline preview
│   ├── random_forest/              # Random Forest validation outputs
│   ├── experiment_summary.csv      # Consolidated experiment metrics
│   └── three_seed_summary.csv     # Mean and sample SD across seeds
├── scripts/
│   ├── build_experiment_summary.py
│   └── build_seed_summary.py
├── src/
│   ├── baselines/random_forest.py
│   ├── datasets/                   # CamVid dataset and transforms
│   ├── losses/                     # Individual and composite losses
│   ├── models/                     # U-Net++ and PSPNet builders
│   ├── data_audit.py
│   ├── inference.py
│   ├── metrics.py
│   ├── search_composite_weights.py
│   └── train.py
├── tests/
│   └── improvements/               # Crop-mode tests
├── .gitignore
├── README.md
└── requirements.txt
```

Training checkpoints (`.pth`, `.pt`, `.ckpt`), the serialized Random Forest model (`.joblib`), the downloaded dataset, virtual environments, and temporary archives are excluded from Git by `.gitignore`. Checkpoints are not included in this repository; obtain or generate the required checkpoint before running inference.

## 8. Installation

### Windows CPU environment

The following commands reflect the local Windows CPU environment used for development and import verification (Python 3.10):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The project dependency file intentionally does not pin the PyTorch wheels, because CPU and CUDA installations differ. Install the PyTorch and Torchvision pair appropriate for the target machine before installing the remaining requirements.

### Google Colab GPU

The deep-learning training experiments were run on a Google Colab T4 GPU. In Colab, use the runtime's compatible PyTorch and Torchvision installation; do not replace it with the Windows CPU wheel command above. Install the remaining project dependencies with:

```python
!pip install -r requirements.txt
```

Verify the runtime before training:

```python
import torch, torchvision, segmentation_models_pytorch
print(torch.__version__)
print(torchvision.__version__)
print(segmentation_models_pytorch.__version__)
print(torch.cuda.is_available())
```

## 9. Running the project

Run commands from the repository root. Ensure the dataset is present at `data/camvid_source/`, or supply `--dataset-root` explicitly.

### 9.1 Dataset audit

```powershell
python -m src.data_audit
```

The audit writes its summary, split checks, pair audit, class distribution, and sample visualizations under `results/audit/`. This command performs the audit; it does not expose a `--help` option.

### 9.2 Random Forest baseline

```powershell
python -m src.baselines.random_forest
```

This trains the baseline from the training split and evaluates it on validation. It writes the model to `models/` and validation outputs to `results/random_forest/`. Running it again retrains and overwrites those default outputs.

### 9.3 Train a deep-learning model

Example: U-Net++ with the selected composite-loss weights:

```powershell
python -m src.train --model unetplusplus --loss composite --loss-weights '{"cross_entropy":0.204267,"dice":0.636507,"lovasz":0.159226}' --epochs 10 --batch-size 4 --learning-rate 0.0001 --weight-decay 0.0001 --seed 42 --dataset-root data/camvid_source --output-dir results/experiments/unetplusplus_composite_seed42
```

Example: PSPNet with equal composite-loss weights:

```powershell
python -m src.train --model pspnet --loss composite --loss-weights '{"cross_entropy":0.333333,"dice":0.333333,"lovasz":0.333334}' --epochs 10 --batch-size 4 --learning-rate 0.0001 --weight-decay 0.0001 --seed 42 --dataset-root data/camvid_source --output-dir results/experiments/pspnet_composite_seed42
```

Individual losses can be selected using `--loss cross_entropy`, `dice`, `focal`, `lovasz`, or `boundary`. Use `--crop-mode none`, `random`, or `class_aware` to select a training crop strategy. See all available arguments with:

```powershell
python -m src.train --help
```

### 9.4 Search composite-loss weights

Example for U-Net++:

```powershell
python -m src.search_composite_weights --model unetplusplus --trials 5 --search-epochs 3 --batch-size 4 --seed 42 --dataset-root data/camvid_source --output-root results/experiments/unetplusplus_composite_seed42_agent_evidence
```

For PSPNet, replace `unetplusplus` with `pspnet` and use a separate output directory. Search results are based on validation mIoU; do not use the test split for tuning.

### 9.5 Inference and evaluation

The inference script accepts a compatible training checkpoint and supports `validation` or `test` splits. It compares standard inference with horizontal-flip TTA by default.

Validation example:

```powershell
python -m src.inference --checkpoint <path-to-best_model.pth> --split validation --dataset-root data/camvid_source --output-dir results/inference/unetplusplus_validation
```

Held-out test evaluation should be performed only for the final selected configuration, not repeatedly during experimentation:

```powershell
python -m src.inference --checkpoint <path-to-best_model.pth> --split test --dataset-root data/camvid_source --output-dir results/inference/unetplusplus_test
```

Replace the checkpoint placeholder with the actual path to the trained checkpoint. The inference script's options are available with:

```powershell
python -m src.inference --help
```

### 9.6 Rebuild result summaries

After the experiment result folders are available, regenerate the CSV summaries:

```powershell
python scripts/build_experiment_summary.py
python scripts/build_seed_summary.py
```

Outputs:

- `results/experiment_summary.csv`
- `results/three_seed_summary.csv`

## 10. Reproducibility and evaluation protocol

- Random seeds are set explicitly in the training and search scripts.
- The individual-loss comparison uses seed 42 for each model/loss combination.
- The final composite configurations were evaluated with seeds 7, 21, and 42 on validation.
- Composite weights were selected using validation mIoU.
- The held-out test split is reserved for final reporting and is not used for tuning.
- The dataset audit found no overlapping image names across the three split lists.
- The experiment JSON and CSV files under `results/experiments/` retain the recorded configuration and evaluation outputs.

For a reproducible run, use the same dataset version, package versions, image dimensions, model settings, loss weights, random seed, batch size, optimizer, learning rate, weight decay, and epoch count. GPU timing and training duration can vary with runtime hardware and load.

## 11. Limitations

- CamVid has substantial class imbalance. Rare classes, particularly Pole, remain difficult for the deep models.
- The class-aware crop ablation did not improve overall validation performance in the recorded run.
- Boundary loss produced substantially lower validation scores than the other individual losses in these experiments.
- Validation and test performance differ; model selection should be based on validation, while the held-out test results should be reported once for the chosen final configuration.
- Checkpoints and the dataset are not distributed in this repository, so inference requires locally available model weights and dataset files.
- Reported compute measurements depend on the hardware and timing protocol. Random Forest CPU inference and deep-model GPU forward latency should not be compared as equivalent measurements.

## 12. Main implementation files

| File | Purpose |
|---|---|
| `src/datasets/camvid.py` | CamVid split loading and dataset interface |
| `src/datasets/transforms.py` | Resizing, augmentation, crop modes and tensor conversion |
| `src/models/build_models.py` | U-Net++ and PSPNet construction |
| `src/losses/segmentation_losses.py` | Individual segmentation losses |
| `src/losses/composite.py` | Weighted composite loss |
| `src/train.py` | Deep-learning training pipeline |
| `src/search_composite_weights.py` | Validation-based composite-weight search |
| `src/inference.py` | Standard and TTA inference evaluation |
| `src/metrics.py` | Segmentation metrics and confusion matrix |
| `src/data_audit.py` | Dataset integrity and class-distribution audit |
| `src/baselines/random_forest.py` | Traditional Random Forest baseline |

---

