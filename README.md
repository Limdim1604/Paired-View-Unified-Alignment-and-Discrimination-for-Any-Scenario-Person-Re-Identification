# PV-UAD: Paired-View Unified Alignment and Discrimination for Any-Scenario Person Re-Identification

PV-UAD is a two-stage framework for **Any-Scenario Person Re-Identification (AS-ReID)** across heterogeneous imaging modalities and camera platforms.

The model is designed for the **WHU-MARS** benchmark, where RGB, NIR, and TIR observations are captured from both **ground** and **aerial** platforms. Instead of training separate models for different modality or viewpoint pairs, PV-UAD learns a **single unified representation** for retrieval across all scenarios.

>  **Research Project:** *PV-UAD: Paired-View Unified Alignment and Discrimination for Any-Scenario Person Re-Identification* (submission pending)
> **Authors:** Thien-Bao Nguyen, Kim-Hai-Anh Cap, Tien-Dung Mai  
> University of Information Technology, Vietnam National University Ho Chi Minh City

---

## Overview

Person Re-Identification becomes substantially harder when both **spectral modality** and **camera viewpoint** change at the same time.

PV-UAD extends the UAD baseline by explicitly exploiting the paired multispectral structure of WHU-MARS through:

- **Paired-view sampling** for synchronized RGB/NIR/TIR observations.
- **View-Factorized Progressive Center Alignment (VF-ProCA)** to separate modality alignment from aerial-ground alignment.
- A **residual invariant representation branch** for identity-stable features.
- **RGB/NIR-to-TIR cross-spectral distillation** with reliability-aware local supervision.
- **Scenario-aware prototype discrimination** for hard cross-scenario positive and negative mining.
- Optional **SwinIR-S ×2** offline resolution enhancement.

The complete system uses a **ViT-B/16** backbone initialized from TransReID.

---

## Method

PV-UAD follows a two-stage training strategy.

### Stage I — Paired-View Representation Learning

Stage I learns a unified identity representation while preserving useful scenario information.

Main components:

1. **TransReID-initialized ViT-B/16**
2. **Paired-view sampling**
3. **Residual invariant branch**
4. **Six-class modality-view scenario head**
5. **View-Factorized Progressive Center Alignment**
6. **Global Prototype Discrimination**

The six scenarios are formed from:

- Modalities: `RGB`, `NIR`, `TIR`
- Views: `Ground`, `Aerial`

### Stage II — Targeted Cross-Spectral Refinement

Stage II starts from the final Stage-I model and prototype bank.

It introduces:

- **Global RGB/NIR → TIR distillation**
- **Agreement-filtered local token distillation**
- **3×3 neighborhood matching** to tolerate sensor misregistration
- **Feature preservation** using a frozen Stage-I teacher
- **Identity-by-scenario prototype bank**
- **Cross-scenario hard-negative ranking**

This stage targets difficult spectral and viewpoint shifts without discarding the representation learned in Stage I.

---

## Dataset

Experiments are conducted on **WHU-MARS-1000**, the paired frame-synchronized subset of WHU-MARS.

| Property | Value |
|---|---:|
| Identities | 1,000 |
| Modalities | RGB / NIR / TIR |
| Platforms | Ground / Aerial |
| Total images | 185,922 |
| Training identities | 500 |
| Test identities | 500 |
| Training images | 92,313 |
| Gallery images | 93,609 |
| Query images | 6,405 |

The same AS-ReID-trained model is also evaluated under:

- **WHU-MARS-1000-GD** — ground-view daytime retrieval
- **AG-ReID** — Aerial → Ground and Ground → Aerial
- **VI-ReID** — NIR → RGB and TIR → RGB

No separate model is trained for these protocol-specific evaluations.

---

## Main Results

### WHU-MARS-1000

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| UAD | 11.00 | 29.50 | 46.30 | 54.80 |
| **PV-UAD** | **11.81** | **30.80** | **47.42** | **56.10** |
| **PV-UAD + TTA** | **11.88** | **30.88** | **47.60** | **56.05** |

PV-UAD improves UAD by **+0.81 mAP** and **+1.30 Rank-1** without re-ranking or test-time augmentation.

### WHU-MARS-1000-GD

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| UAD | 14.50 | 31.90 | 45.00 | 52.50 |
| **PV-UAD** | **17.09** | **36.59** | **49.90** | **56.81** |
| **PV-UAD + TTA** | **17.13** | **36.48** | **50.10** | **57.01** |

The largest gain over UAD is observed in this setting: **+2.59 mAP**.

### Aerial-Ground Re-ID

| Direction | Method | mAP | Rank-1 | Rank-10 |
|---|---|---:|---:|---:|
| A → G | UAD | 11.50 | 31.90 | 58.30 |
| A → G | **PV-UAD** | **13.02** | **35.41** | **60.44** |
| G → A | UAD | 13.30 | 30.50 | 54.70 |
| G → A | **PV-UAD** | **15.30** | **32.87** | **58.59** |

PV-UAD improves mAP by **+1.52** for A→G and **+2.00** for G→A.

### Visible-Infrared Re-ID

| Direction | Method | Rank-1 | mAP | mINP |
|---|---|---:|---:|---:|
| NIR → RGB | UAD | 25.53 | 16.58 | 3.10 |
| NIR → RGB | PV-UAD + TTA | 24.78 | **17.25** | **3.33** |
| TIR → RGB | UAD | **9.04** | **6.32** | **1.36** |
| TIR → RGB | PV-UAD + TTA | 7.63 | 5.78 | 1.32 |

The NIR→RGB setting improves in mAP and mINP, while **TIR→RGB remains a challenging limitation**.

---

## Implementation Details

| Setting | Value |
|---|---|
| Backbone | ViT-B/16 |
| Input size | 256 × 128 |
| Feature dimension | 768 |
| Initialization | TransReID pretrained on MSMT17 |
| Hardware | 2 × NVIDIA T4 |
| Stage I | 60 epochs |
| Stage II | 25 epochs |
| Stage-I optimizer | SGD |
| Stage-I learning rate | `8e-3` |
| Stage-II learning rate | `1e-4` |
| Paired keys per identity | `K = 4` |
| Global paired-key batch size | 24 |
| Test batch size | 256 |
| Prototype EMA coefficient | 0.2 |
| Re-ranking | No |

For offline resolution enhancement, the experiments use pretrained **SwinIR-S ×2** without ReID-specific fine-tuning.

---

## Inference

The final retrieval descriptor combines the global backbone representation and residual invariant representation:

```text
h_test = 0.5 * h + 0.5 * h_inv
```

The standard result uses a single forward pass.

An optional horizontal-flip TTA setting sums normalized descriptors from the original and flipped images before retrieval.

**No re-ranking is used.**

---

## Key Findings

PV-UAD improves the UAD baseline on:

- Full **Any-Scenario Re-ID**
- **Ground-daytime** retrieval
- **Aerial → Ground** retrieval
- **Ground → Aerial** retrieval
- **NIR → RGB** mAP and mINP

The experiments indicate that **paired-view alignment** and **scenario-aware discrimination** are the primary contributors, while thermal-visible matching remains the main unresolved challenge.

---

## Repository Structure

The exact folder structure may vary as the implementation is organized. A typical layout is:

```text
PV-UAD/
├── configs/
├── datasets/
├── models/
├── losses/
├── engine/
├── utils/
├── scripts/
├── README.md
└── ...
```

> Update this section to match the final repository structure before release.

---

## Usage

Training and evaluation commands should match the final code organization of this repository.

Recommended documentation to add once the implementation is finalized:

```text
1. Environment setup
2. WHU-MARS dataset preparation
3. SwinIR offline preprocessing
4. Stage-I training
5. Stage-II refinement
6. Evaluation on AS-ReID / GD / AG-ReID / VI-ReID
```

This README intentionally does not provide unverified commands before the final repository structure and configuration files are fixed.

---
## Acknowledgements

This research was supported by the **VNUHCM-University of Information Technology's Scientific Research Support Fund**.

PV-UAD builds upon ideas and benchmarks from prior work including **WHU-MARS**, **UAD**, **TransReID**, and **SwinIR**. Please refer to the paper for the complete list of references.

---

## Contact

For questions about this project, please open an issue in this repository or contact the authors through the information provided in the paper.
