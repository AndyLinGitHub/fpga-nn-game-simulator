import os
import json
from collections import OrderedDict
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.io import read_image
import torch.ao.quantization as tq
from datetime import datetime
from ptflops import get_model_complexity_info
from PIL import Image

from config import VAEConfig, auto_argparse_from_config, update_config_from_args

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, decimals):
        scale = 10 ** decimals
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradients straight through
        return grad_output, None
        
class ClipRound(nn.Module):
    def __init__(self, clip_min=-1.0, clip_max=1.0, decimals=4):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.decimals = decimals

    def forward(self, x):
        x = torch.clamp(x, self.clip_min, self.clip_max)
        x = RoundSTE.apply(x, self.decimals)
        
        return x

def insert_clip_round(seq_model, clip_min=-1.0, clip_max=1.0, decimals=4):
    new_layers = OrderedDict()
    for name, layer in seq_model.named_children():
        new_layers[name] = layer
        clip_name = f"clip_{name}"
        new_layers[clip_name] = ClipRound(clip_min, clip_max, decimals)
        
    return nn.Sequential(new_layers)

def round_to_decimal(x, decimals=4):
    scale = 10 ** decimals
    return torch.round(x * scale) / scale
    
def clip_and_round_params(model, clip_min=-1.0, clip_max=1.0, decimals=4):
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                round_to_decimal(torch.clamp(param, clip_min, clip_max), decimals)
            )

class SigmoidBaseK(nn.Module):
    def __init__(self, k):
        super(SigmoidBaseK, self).__init__()
        self.k = k

    def forward(self, x):
        return 1 / (1 + torch.pow(self.k, -x))

class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        self.config = config

        # Qn.m Quantization
        if config.quant:
            self.clip_round = ClipRound(config.clip_min, config.clip_max, config.decimals)
        
        # Encoder
        enc_layers = []
        in_ch = config.in_channels
        H, W = config.image_size
        layer_count = 0
        for out_ch in config.encoder_channels:
            enc_layers.append(
                (f"conv{layer_count}", nn.Conv2d(in_ch, out_ch, config.kernel_size, config.stride, config.padding))
                             )
            enc_layers.append(
                (f"relu{layer_count}", nn.ReLU())
            )
            in_ch = out_ch
            H = (H + 2 * config.padding - config.kernel_size) // config.stride + 1
            W = (W + 2 * config.padding - config.kernel_size) // config.stride + 1
            layer_count += 1

        #print("Latent Dimension (C, H, W): ", config.latent_dim, H, W)

        self.encoder = nn.Sequential(OrderedDict(enc_layers))
        if config.quant:
            self.encoder = insert_clip_round(self.encoder, config.clip_min, config.clip_max, config.decimals)
        
        self.H, self.W = H, W
        self.mu = nn.Conv2d(out_ch, config.latent_dim, 1)
        self.logvar = nn.Conv2d(out_ch, config.latent_dim, 1)

        dec_layers = []
        in_ch = config.latent_dim
        layer_count = 0
        for out_ch in config.decoder_channels[1:]:
            dec_layers.append(
                (f"convt{layer_count}", nn.ConvTranspose2d(in_ch, out_ch, config.kernel_size, config.stride, config.padding))
            )
            dec_layers.append(
                (f"relu{layer_count}", nn.ReLU())
            )
            in_ch = out_ch
            layer_count += 1
            
        dec_layers.append(
            (f"convt{layer_count}", nn.ConvTranspose2d(in_ch, config.in_channels, config.kernel_size, config.stride, config.padding))
        )

        if config.sigmoid_base is not None:
            dec_layers.append(
                ("sigmoid", SigmoidBaseK(config.sigmoid_base))
            )
        else:
            dec_layers.append(
                ("sigmoid", nn.Sigmoid())
            )

        self.decoder = nn.Sequential(OrderedDict(dec_layers))
        if config.quant:
            self.decoder = insert_clip_round(self.decoder, config.clip_min, config.clip_max, config.decimals)

    def encode(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def decode(self, z):
        x = z
        if self.config.quant:
            x = self.clip_round(x)
            
        return self.decoder(x)

    def forward(self, x):
        if self.config.quant:
            x = self.clip_round(x)
        mu, logvar = self.encode(x)

        if self.config.quant:
            mu = self.clip_round(mu)
            logvar = self.clip_round(logvar)
        z = self.reparameterize(mu, logvar)

        if self.config.quant:
            z = self.clip_round(z)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def load_and_preprocess_images(file_list, transform, device):
    imgs = []
    for file in file_list:
        img = Image.open(file)
        if transform:
            img = transform(img)
        imgs.append(img)
    batch = torch.stack(imgs).to(device)
    return batch

def train_vae(config: VAEConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Save config to log directory
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor()
    ])

    from torch.utils.data import ConcatDataset
    dataset1 = datasets.ImageFolder(root=config.data_dir, transform=transform)
    dataset2 = datasets.ImageFolder(root=config.data_dir2,transform=transform)
    dataset = ConcatDataset([dataset1, dataset2])

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = VAE(config).to(device)
    #print(model)
    print("VAE Summary:")
    with torch.cuda.device(0) if torch.cuda.is_available() else torch.cpu.device(0):
        get_model_complexity_info(model, (config.in_channels, *config.image_size), 
                                      as_strings=True, print_per_layer_stat=True)
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint))

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        avg_train_loss = 0.0
        if config.retrain:
            model.train()
            model.to(device)
            total_loss = 0
            for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} - Training"):
                x = x.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x)
                loss = loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
                if config.quant:
                    clip_and_round_params(model, config.clip_min, config.clip_max, config.decimals)
    
            avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} - Validation"):
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss = loss_function(recon, x, mu, logvar)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
        
        # Custom reconstruction targets
        custom_files = [
            os.path.join(config.data_dir, config.game, f"frame_{i}.png")
            for i in [1, 10, 100, 1000]
        ]
        with torch.no_grad():
            sample = load_and_preprocess_images(custom_files, transform, device)
            recon, _, _ = model(sample)
            sample_grid = make_grid(torch.cat([sample, recon]), nrow=4)
            writer.add_image("Reconstruction", sample_grid, epoch)

        if not config.retrain:
            break

    writer.close()

    return os.path.join(log_dir, "best_model.pt")


if __name__ == "__main__":
    config = VAEConfig()
    parser = auto_argparse_from_config(config)
    args = parser.parse_args()
    config = update_config_from_args(config, args)
    
    # Training
    best_model_dir = train_vae(config)
    print(best_model_dir)

    # Post-training quantization
    config.checkpoint = best_model_dir
    config.sigmoid_base = 4
    config.quant = True
    config.learning_rate = config.learning_rate/10
    best_model_dir = train_vae(config)
    print(best_model_dir)