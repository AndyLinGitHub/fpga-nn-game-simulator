from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from unet import FiLMUNet, LatentSequenceDataset
from config import *

class CombinedModel(nn.Module):
    def __init__(self, conv_model, vae_decoder):
        super(CombinedModel, self).__init__()
        self.conv_model = conv_model
        self.vae_decoder = vae_decoder

    def forward(self, x, cond):
        features = self.conv_model(x, cond)
        output = self.vae_decoder(features)
        return features, output

if __name__ == "__main__":
    unet_config = UNetConfig()
    vae_config = VAEConfig()
    parser = auto_argparse_from_config(unet_config)
    parser = auto_argparse_from_config(vae_config, parser)
    args = parser.parse_args()
    unet_config = update_config_from_args(unet_config, args)
    vae_config = update_config_from_args(vae_config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LatentSequenceDataset(unet_config, vae_config, device)
    quant_args = SimpleNamespace(quant=unet_config.quant, clip_min=unet_config.clip_min, clip_max=unet_config.clip_max, decimals=unet_config.decimals)
    model = FiLMUNet(unet_config, quant_args)
    if unet_config.checkpoint is not None:
        model.load_state_dict(torch.load(unet_config.checkpoint))
    model.to(device)

    model.eval()
    dataset.vae_model.eval()
    full_model = CombinedModel(model, dataset.vae_model.decoder).to("cuda")

    latents, _, _ = dataset.__getitem__(0)
    latents = latents.to(device).unsqueeze(0)
    ref = latents[:, :4, ]
    np.save('initial_input.npy', latents.cpu().numpy())


    dummy_cond= torch.randint(0, 2, (1, 8)).float()
    with torch.no_grad():
        export = torch.onnx.export(
            full_model.to("cpu"),
            (latents.to("cpu"), dummy_cond.to("cpu")),
            "model.onnx",
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

    print("ONNX model exported as model.onnx")