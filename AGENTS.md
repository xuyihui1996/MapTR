# AGENTS.md for Map\_TRv2

## Overview

MapTRv2 is an advanced, end-to-end framework designed for online vectorized High-Definition (HD) map construction for autonomous driving applications. It leverages a structured Transformer-based approach to achieve state-of-the-art performance in real-time.

## Project Structure

* `mmdetection3d/`: Core 3D detection modules.
* `projects/`: Specific MapTRv2 configurations and plugins.
* `tools/`: Training, evaluation, and visualization scripts.
* `configs/`: Configuration files for model training and testing.
* `ckpts/`: Pretrained checkpoints.
* `data/`: Dataset directories (nuScenes, Argoverse2).

## Environment Setup

Ensure you have created a dedicated conda environment:

```bash
conda create -n maptr python=3.8 -y
conda activate maptr
```

Install all dependencies:

```bash
pip install torch torchvision torchaudio mmcv-full mmdet mmsegmentation timm
```

Set up project-specific modules:

```bash
cd /path/to/MapTR/mmdetection3d
python setup.py develop

cd /path/to/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install

cd /path/to/MapTR
pip install -r requirement.txt
```

## Data Preparation

Download datasets:

* [nuScenes Dataset](https://www.nuscenes.org/download)
* [Argoverse2 Dataset](https://www.argoverse.org/av2.html#download-link)

Prepare nuScenes data:

```bash
python tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Prepare Argoverse2 data:

```bash
python tools/maptrv2/custom_av2_map_converter.py --data-root ./data/argoverse2/sensor/
```

## Training and Evaluation

### Training

Run training on 8 GPUs:

```bash
./tools/dist_train.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py 8
```

### Evaluation

Evaluate model performance:

```bash
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py ./path/to/ckpts.pth 8
```

## Visualization

Visualization scripts are under `tools/maptrv2/`:

```bash
python tools/maptrv2/nusc_vis_pred.py /path/to/experiment/config /path/to/experiment/ckpt
python tools/maptrv2/av2_vis_pred.py /path/to/experiment/config /path/to/experiment/ckpt
```

To generate qualitative benchmark videos:

```bash
python tools/maptr/generate_video.py /path/to/visualization/directory
```

## Model & Architecture

MapTRv2 adopts a hierarchical Transformer encoder-decoder structure:

* **Encoder:** Converts multi-view images into BEV (Bird's Eye View) representations using methods like BEVPoolv2.
* **Decoder:** Uses hierarchical queries (instance-level and point-level) and decoupled self-attention mechanisms to efficiently predict vectorized map elements.

## Contribution Guidelines

* Ensure coding standards by running:

```bash
flake8
black .
```

* Run tests locally before submitting a PR:

```bash
pytest tests/
```

* Maintain clarity and concise commit messages and PR titles: `[MapTRv2] Short description`

## Testing and Validation

* Unit tests and integration tests should be written for new functionalities.
* Test coverage should remain above 80%.
* Continuously validate accuracy and efficiency benchmarks (mAP, FPS).

## PR Instructions

* Format: `[MapTRv2] Brief Description`
* Clearly document changes and testing results.
* Ensure all CI checks pass before merging.

---

This structured guide aims to enhance collaboration efficiency, improve code quality, and ensure consistent development practices for MapTRv2.
