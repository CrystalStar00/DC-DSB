This repository contains the implementation of the paper:  
**Prediction of Single-Cell Perturbation Response Based on Direction-Constrained Diffusion Schrödinger Bridge**

---

## Project Description

**DC-DSB** (Direction-Constrained Diffusion Schrödinger Bridge) predicts transcriptional responses of single cells under external perturbations by learning probabilistic trajectories between unperturbed and perturbed distributions.  
The model integrates experimental factors and prior knowledge via multi-level conditional representations, and applies **direction-constrained conditioning**—guidance only along the generative direction—to improve biological plausibility and training stability.

## Environment Installation

We recommend using **conda**:

```bash
# Create environment
conda create -n dcdsb python=3.10

# Activate
conda activate dcdsb

# Install PyTorch 2.0.1 (match CUDA/CPU per your system)
# Example A (CUDA 11.8):
conda install -y -c pytorch -c nvidia pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8

# Install dependencies
conda install -y -c pyg pyg==2.3.1
pip install -r requirements.txt
```

## Dataset and Checkpoints

Download datasets and checkpoints from the following links and place them under：
https://drive.google.com/drive/folders/10yKqU3BAjKcTudMHsxshlkqJQyMl55lu?usp=sharing

## Training & Inference

Run training with:

```
python train.py \
  --data norman \
  --exp_name norman \
  --epoch 300 \
  --batch_size 512 \
  --training_timesteps 50 \
  --inference_timesteps 50 \
  --lr 1e-5
```

Run inference with:

```
python inference.py \
  --data norman \
  --exp_name norman \
  --batch_size 512 \
  --training_timesteps 50 \
  --inference_timesteps 50 \
  --ckpt ./ckpt/norman.pth
```

