# Environment Setup
```
# 1. Create a new conda environment with Python 3.9
conda create -n fngs python=3.9 -y

# 2. Activate the environment
conda activate fngs

# 3. Install required packages from requirements.txt
pip install -r requirements.txt
```

# Description and Usage
The model preparation procedure consists of six steps:
- Collect game frame–action pairs as training data.
- Train a Variational Autoencoder (VAE) with an encoder–decoder structure to compress and reconstruct game frames.
- Fine-tune the trained VAE with post-training quantization.
- Train a Convolutional Neural Network (CNN) that takes image latents (from the VAE encoder) and binary-encoded player actions to predict the latent representation of the next frame.
- Fine-tune the trained CNN with post-training quantization.
- The decoder part of the VAE is combined with the CNN to form a complete model, which is then exported to ONNX format and converted into a .coe file for initializing Block RAM on the FPGA.

## Collect Game Frames
The game frames are generated and collected by running a Pong game implemented in Pygame, using pre-generated random keyboard actions as input. In this project, because the CNN output is deterministic, the game should avoid randomness and sudden full-frame changes to maintain good demo quality. This limitation can be addressed in future work by using sample-based models such as diffusion models.
```
python game.py
``` 

## Train VAE
After the VAE is trained, it is used to encode the essential information of each game frame into a compact latent representation. The CNN then operates on this encoded data to predict the latent representation of the next frame. Finally, the VAE decoder reconstructs the game frame from the predicted latent. This approach significantly reduces the input and output dimensions for the CNN, resulting in a smaller model size while maintaining high prediction quality.
```
python vae.py
```

Use TensorBoard to view the log
```
tensorboard --logdir ./runs
```

## Train Unet
In this application, we use a U-Net model to predict the next frame. For the demo, to fit within the Block RAM constraints, the U-Net is simplified to a standard CNN. You can increase the ```num_layers``` parameter in the ```UNetConfig``` class (in config.py) to create a more complex U-Net model.
```
python unet.py --vae.checkpoint [QUANTIZED_VAE_CHECKPOINT_PATH] --vae.sigmoid_base 4 --vae.quant True
```

Use TensorBoard to view the log
```
tensorboard --logdir ./runs
```
## Export Model
