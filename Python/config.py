import argparse
from ast import literal_eval

def smart_type(v):
    try:
        return literal_eval(v)
    except Exception:
        return v
    
def auto_argparse_from_config(config, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        
    for k, v in config.__dict__.items():
        parser.add_argument(f"--{config.name}.{k}", type=smart_type, help=f"(default: {v})")
    return parser

def update_config_from_args(config, args):
    for key, value in vars(args).items():
        config_name, attr = key.split(".")
        if value is not None and config_name == config.name:
            setattr(config, attr, value)
            
    return config

class VAEConfig:
    def __init__(self):
        self.name = "vae"
        self.scale = 4
        self.image_size = (64, 64)  # (H, W)
        self.in_channels = 3
        self.latent_dim = 4
        self.encoder_channels = [8*self.scale, 16*self.scale, 32*self.scale, 64*self.scale]
        self.decoder_channels = [64*self.scale, 32*self.scale, 16*self.scale, 8*self.scale]
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.epochs = 1
        self.data_dir = "pong_frames"
        self.data_dir2 = "pong_frames"        
        self.log_dir = "runs/pong_vae"
        self.game = "pong"
        self.checkpoint = None
        self.retrain = True
        self.sigmoid_base = None
        self.quant = False
        self.clip_min = -4.0
        self.clip_max = 4.0
        self.decimals = 4

class UNetConfig:
    def __init__(self):
        self.name = "unet"
        self.data_dir = "pong_frames"        
        self.game = "pong"
        self.log_dir = "runs/unet"
        self.conds_dir = "pong_frames/pong"
        self.conds_file = "player_encoding.pkl"
        self.latent_dim = 4
        self.sequence_length = 4
        self.cond_dim = 8
        self.num_layers = 1
        self.image_latent = "image_latent.pt"
        self.learning_rate = 1e-3
        self.weight_decay = 0
        self.start_factor = 1
        self.end_factor = 0.1
        self.batch_size = 4096
        self.epochs = 1024
        self.checkpoint = None
        self.quant = False
        self.clip_min = -4.0
        self.clip_max = 4.0
        self.decimals = 4