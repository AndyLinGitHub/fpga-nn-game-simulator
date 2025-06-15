# Environment Setup
```
# 1. Create a new conda environment with Python 3.9
conda create -n fngs python=3.9 -y

# 2. Activate the environment
conda activate fngs

# 3. Install required packages from requirements.txt
pip install -r requirements.txt
```

# Description
The model preparation procedure consists of six steps:
- Collect game frame–action pairs as training data.
- Train a Variational Autoencoder (VAE) with an encoder–decoder structure to reconstruct game frames.
- Fine-tune the trained VAE using post-quantization training.
- Train a Convolutional Neural Network (CNN) that takes image latents (from the VAE encoder) and binary-encoded player actions to predict the latent representation of the next frame.
- Fine-tune the trained CNN using post-quantization training.
- The decoder part of the VAE is combined with the CNN to form a complete model, which is then exported to ONNX format and converted into a .coe file for initializing Block RAM on the FPGA.

## Collect Game Frames
```
python game.py
``` 

## Train VAE
```
python vae.py
```

## Train CNN
```
python cnn.py
```

## Export Model
