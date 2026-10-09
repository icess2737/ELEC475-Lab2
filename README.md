# ELEC475 Lab 2

PyTorch implementations of base LeNet, modern LeNet, AlexNet for 64 x 64 RGB inputs, and a torchvision AlexNet adapter. AlexNet supports no normalization, LRN, and BatchNorm.

## Setup

Create a Python 3.12 virtual environment. For the measured CUDA 12.4 environment:

```text
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Run commands from the repository root. CIFAR-10 and Tiny ImageNet download into `data/` on demand. Tiny ImageNet validation uses its separate official validation images and annotations.

## Experiments

```text
python -m models.alexnet
python compare_models.py --device cuda
python demo_transforms.py
python run_postlab.py
```

The restartable experiment queue covers the seven-model ladder on both datasets, three-seed normalization and augmentation comparisons on CIFAR-10, and a three-seed dropout ablation. It saves per-epoch CSV histories, final JSON metrics, best-validation weights, final weights, and curves. Completed runs are skipped when their final metrics match the requested epoch count.

Training uses Adam with learning rate 0.001, weight decay 0.0001, batch size 128, and seeds 42, 475, and 2035. LeNet runs use five epochs; the normalization ladder uses ten; the augmentation and dropout comparisons use fifteen. The full Tiny ImageNet ladder uses one seed. Training and clean training evaluation drop the final incomplete batch. Validation includes every validation image.

Results tables use final-epoch metrics. A best-validation checkpoint can produce different metrics from the final epoch. Training curves record training-mode accuracy; generalization gaps use clean inputs in evaluation mode. FLOPs are twice the MAC estimate, with the operation coverage of THOP.

The submission archive separately contains the report and required checkpoints. Datasets, virtual environments, reports, and checkpoints are excluded from this source repository.
