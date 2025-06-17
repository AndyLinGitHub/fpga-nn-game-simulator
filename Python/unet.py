import os
import re
import gc
import json
import pickle
from types import SimpleNamespace

from tqdm import tqdm
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info

from vae import VAE, ClipRound, insert_clip_round, clip_and_round_params
from config import *

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, quant_args):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*4, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if quant_args.quant:
            self.conv = insert_clip_round(self.conv, quant_args.clip_min, quant_args.clip_max, quant_args.decimals)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, quant_args):
        super().__init__()

        self.down = nn.Sequential(
            #nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels,  in_channels, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            DoubleConv(in_channels, out_channels, quant_args),
        )

        if quant_args.quant:
            self.down = insert_clip_round(self.down, quant_args.clip_min, quant_args.clip_max, quant_args.decimals)

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, quant_args):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels, quant_args)
        self.quant = quant_args.quant
        if quant_args.quant:
            self.clip_round = ClipRound(quant_args.clip_min, quant_args.clip_max, quant_args.decimals)

    def forward(self, x, x_skip):
        x = self.up(x)
        if self.quant:
            x = self.clip_round(x)
        x = torch.cat([x_skip, x], dim=1)
        
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, quant_args):
        super().__init__()

        self.in_conv = DoubleConv(in_channels, in_channels*2, quant_args)
        self.downs = nn.ModuleList()
        ch = in_channels*2
        for _ in range(num_layers - 1):
            self.downs.append(Down(ch, ch * 2, quant_args))
            ch *= 2

        self.ups = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.ups.append(Up(ch, ch // 2, quant_args))
            ch //= 2

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.quant = quant_args.quant
        if quant_args.quant:
            self.clip_round = ClipRound(quant_args.clip_min, quant_args.clip_max, quant_args.decimals)

    def forward(self, x):
        dwon_features = []

        x = self.in_conv(x)
        if self.quant:
            x = self.clip_round(x)
        dwon_features.append(x)
        for down in self.downs:
            x = down(x)
            dwon_features.append(x)

        for i, up in enumerate(self.ups):
            x = up(x, dwon_features[-(i + 2)])

        x = self.out_conv(x)
        if self.quant:
            x = self.clip_round(x)
        return x

class FiLM(nn.Module):
    def __init__(self, cond_dim, in_channels, quant_args):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(cond_dim, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels * 2)
        )

        self.quant = quant_args.quant
        if quant_args.quant:
            self.projector = insert_clip_round(self.projector, quant_args.clip_min, quant_args.clip_max, quant_args.decimals)
            self.clip_round = ClipRound(quant_args.clip_min, quant_args.clip_max, quant_args.decimals)

    def forward(self, x, cond):
        gamma, beta = self.projector(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        x = x * (1 + gamma) + beta
        if self.quant:
            x = self.clip_round(x)

        return x
        
class FiLMUNet(nn.Module):
    def __init__(self, config, quant_args):
        super().__init__()
        in_channels = config.sequence_length * config.latent_dim

        self.film = FiLM(config.cond_dim, in_channels, quant_args)
        self.unet = UNet(in_channels * 2, config.latent_dim, config.num_layers, quant_args)

        self.quant = config.quant
        if config.quant:
            self.clip_round = ClipRound(config.clip_min, config.clip_max, config.decimals)

    def forward(self, x, cond):
        if self.quant:
            x = self.clip_round(x)
            cond = self.clip_round(cond)

        x_film = self.film(x, cond)
        x = torch.cat([x_film, x], dim = 1)
        
        return self.unet(x)

class LatentSequenceDataset(Dataset):
    def __init__(self, unet_config, vae_config, device):
        self.vae_model = VAE(vae_config)
        self.vae_model.load_state_dict(torch.load(vae_config.checkpoint))
        self.vae_model.to(device)
        self.sequence_length = unet_config.sequence_length

        conds_path = os.path.join(unet_config.conds_dir, unet_config.conds_file)
        with open(conds_path, 'rb') as f:
            self.conds = pickle.load(f)

        print(len(self.conds))

        if not os.path.exists(unet_config.image_latent):
            image_folder = os.path.join(unet_config.data_dir, unet_config.game)
            image_files = os.listdir(image_folder)
            image_files = [f for f in image_files if f.endswith(".png")]
            image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))

            transform = transforms.Compose([
                transforms.Resize(vae_config.image_size),
                transforms.ToTensor()
                ])

            image_list = []
            image_latent = []
            for i in tqdm(range(len(image_files))):
                image = Image.open(os.path.join(image_folder, image_files[i])).convert("RGB")
                image_list.append(transform(image))
                
                if len(image_list) == unet_config.batch_size or i == len(image_files) - 1:
                    image_tensor = torch.stack(image_list).to(device)
                    with torch.no_grad():
                        mu, logvar = self.vae_model.encode(image_tensor)
                        image_latent.append(self.vae_model.reparameterize(mu, logvar).cpu())

                    image_list = []

            image_latent = torch.cat(image_latent, dim=0)
            torch.save(image_latent, unet_config.image_latent)
            del mu, logvar, image_list, image_latent
            gc.collect()
            torch.cuda.empty_cache()

        self.image_latent = torch.load(unet_config.image_latent)

    def __len__(self):
        return len(self.image_latent) - 1

    def __getitem__(self, idx):
        end_idx = idx + 1
        start_idx = max(0, end_idx - self.sequence_length)
        latents = self.image_latent[start_idx:end_idx]
        next_latent = self.image_latent[end_idx]
        
        while latents.shape[0] < self.sequence_length:
            latents = torch.concat([latents[0].unsqueeze(0), latents], dim=0)

        latents = latents.view(-1, *latents.shape[2:])

        cond = self.conds[end_idx-1]
        cond = torch.tensor(cond, dtype=torch.float32)

        return latents, cond, next_latent

def add_gaussian_noise(x: torch.Tensor, scale: float = 0.7) -> torch.Tensor:
    noise = (torch.randn_like(x) * scale).to(x.device)
    
    return x + noise
    
def train(model, quant_args, dataloader, optimizer, device, epoch, loss_fn=nn.MSELoss()):
    model.train()
    total_loss = 0
    for latents, cond, next_latent in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        latents = latents.to(device)
        with torch.no_grad():
            latents = add_gaussian_noise(latents)
            
        cond = cond.to(device)
        next_latent = next_latent.to(device)
        pred_latent = model(latents, cond)

        loss = loss_fn(pred_latent, next_latent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if quant_args.quant:
            clip_and_round_params(model, quant_args.clip_min, quant_args.clip_max, quant_args.decimals)

    avg_loss = total_loss / len(dataloader)
    
    return avg_loss

def validate(model, dataloader, device, epoch, loss_fn=nn.MSELoss()):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for latents, cond, next_latent in tqdm(dataloader, desc=f"Validating Epoch {epoch}"):
            latents = latents.to(device)
            latents = add_gaussian_noise(latents)
            cond = cond.to(device)
            next_latent = next_latent.to(device)
            pred_latent = model(latents, cond)

            loss = loss_fn(pred_latent, next_latent)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    
    return avg_loss

def train_unet(unet_config, vae_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(unet_config.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    with open(os.path.join(log_dir, "unet_config.json"), "w") as f:
        json.dump(unet_config.__dict__, f, indent=4)

    with open(os.path.join(log_dir, "vae_config.json"), "w") as f:
        json.dump(vae_config.__dict__, f, indent=4)
    
    dataset = LatentSequenceDataset(unet_config, vae_config, device)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=unet_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=unet_config.batch_size, shuffle=True)

    quant_args = SimpleNamespace(quant=unet_config.quant, clip_min=unet_config.clip_min, clip_max=unet_config.clip_max, decimals=unet_config.decimals)
    model = FiLMUNet(unet_config, quant_args)
    if unet_config.checkpoint is not None:
        model.load_state_dict(torch.load(unet_config.checkpoint))
    model.to(device)
    #print(model)
    print("Unet Summary:")

    def input_constructor(input_res):
        # input_res is a tuple of input shapes
        x1_shape, x2_shape = input_res
        x1 = torch.zeros((1, *x1_shape)).to("cuda")
        x2 = torch.zeros((1, *x2_shape)).to("cuda")
        print(x1.shape, x2.shape)
        return dict(x=x1, cond=x2)
    
    input_shapes = ((unet_config.latent_dim*unet_config.sequence_length, int(vae_config.image_size[0]/2**(len(vae_config.encoder_channels))), int(vae_config.image_size[1]/2**(len(vae_config.encoder_channels)))), 
                    (unet_config.cond_dim,))  # shapes for x1 and x2
    with torch.cuda.device(0) if torch.cuda.is_available() else torch.cpu.device(0):
        get_model_complexity_info(model, input_shapes, input_constructor=input_constructor, 
                                  as_strings=True, print_per_layer_stat=True)

    optimizer = optim.Adam(model.parameters(), lr=unet_config.learning_rate, weight_decay=unet_config.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=unet_config.start_factor, end_factor=unet_config.end_factor, total_iters=unet_config.epochs)

    best_val_loss = float('inf')
    for epoch in range(1, unet_config.epochs + 1):
        avg_train_loss = train(model, quant_args, train_loader, optimizer, device, epoch)
        avg_val_loss = validate(model, val_loader, device, epoch)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{unet_config.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))

        scheduler.step()

    return os.path.join(log_dir, "best_model.pt")

if __name__ == "__main__":
    unet_config = UNetConfig()
    vae_config = VAEConfig()
    parser = auto_argparse_from_config(unet_config)
    parser = auto_argparse_from_config(vae_config, parser)
    args = parser.parse_args()
    unet_config = update_config_from_args(unet_config, args)
    vae_config = update_config_from_args(vae_config, args)
    
    # Training
    best_model_dir = train_unet(unet_config, vae_config)
    print(best_model_dir)

    # Post-training quantization
    unet_config.checkpoint = best_model_dir
    unet_config.quant = True
    unet_config.learning_rate = unet_config.learning_rate/10
    best_model_dir = train_unet(unet_config, vae_config)
    print(best_model_dir)