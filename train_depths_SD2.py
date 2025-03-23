import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from torch.amp import GradScaler
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

# ========= CONFIG =========
PRETRAINED_MODEL = "stabilityai/stable-diffusion-2-1-base"
DATASET_ROOT = "/home/exp1/zihan2/DIODE_dataset"
SPLIT = "train"
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/home/exp1/zihan2/DIODE_dataset/output"
PROMPT = "depth map"
LATENT_SCALING_FACTOR = 0.13025
IMAGE_SIZE = 768

torch.set_default_dtype(torch.float32)

# ========= DATASET =========
class DepthDataset(Dataset):
    def __init__(self, data_root, split="train", size=IMAGE_SIZE):
        self.data_root = Path(data_root)
        self.split = split
        self.image_paths = []
        self.gt_paths = []

        meta_path = self.data_root / "diode_meta3.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        for scene in meta[split]["indoors"]["images"]:
            image_path = self.data_root / split / "indoors" / scene
            scene_name = scene.replace(".png", "")
            depth_path = self.data_root / split / "indoors" / f"{scene_name}_depth.npy"

            if image_path.exists() and depth_path.exists():
                self.image_paths.append(image_path)
                self.gt_paths.append(depth_path)

        self.image_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_tensor = self.image_transform(image)

        depth = np.load(self.gt_paths[idx]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (depth > 0).astype(np.float32)
        depth_tensor = torch.from_numpy(depth)[None, None, ...]
        depth_tensor = F.interpolate(depth_tensor, size=(self.size, self.size), mode='bilinear', align_corners=False)[0]
        mask_tensor = torch.from_numpy(mask)[None, None, ...]
        mask_tensor = F.interpolate(mask_tensor, size=(self.size, self.size), mode='nearest')[0]

        max_depth = torch.max(depth_tensor[mask_tensor.bool()]) if mask_tensor.any() else 1.0
        depth_tensor = depth_tensor / (max_depth + 1e-6)

        return image_tensor, depth_tensor, mask_tensor

train_loader = DataLoader(DepthDataset(DATASET_ROOT, SPLIT), batch_size=BATCH_SIZE, shuffle=True)

# ========= LOAD MODELS =========
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL, subfolder="vae").to(DEVICE)
unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL, subfolder="unet").to(DEVICE)
text_encoder = CLIPTextModelWithProjection.from_pretrained(PRETRAINED_MODEL, subfolder="text_encoder").to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL, subfolder="tokenizer")

vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

# ========= SETUP LORA =========
lora_attn_procs = {}
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name.split(".")[1])
        hidden_size = unet.config.block_out_channels[::-1][block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name.split(".")[1])
        hidden_size = unet.config.block_out_channels[block_id]
    else:
        hidden_size = unet.config.block_out_channels[0]

    lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=8)

unet.set_attn_processor(lora_attn_procs)
attn_lora_layers = AttnProcsLayers(unet.attn_processors).to(DEVICE)
optimizer = torch.optim.AdamW(attn_lora_layers.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# ========= PROMPT CONDITIONING =========
text_input = tokenizer([PROMPT], padding="max_length", return_tensors="pt", truncation=True, max_length=77)
text_embeds = text_encoder(text_input.input_ids.to(DEVICE)).last_hidden_state

# ========= TRAIN LOOP =========
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_losses = []
all_step_losses = []

for epoch in range(EPOCHS):
    unet.train()
    epoch_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, depths, masks in progress:
        images = images.to(DEVICE)
        depths = depths.to(DEVICE)
        masks = masks.to(DEVICE)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.mode() * LATENT_SCALING_FACTOR

        timesteps = torch.full((images.size(0),), 999, device=DEVICE, dtype=torch.long)

        with autocast():
            pred_latents = unet(latents, timesteps, encoder_hidden_states=text_embeds.expand(images.size(0), -1, -1)).sample
            decoded_images = vae.decode(pred_latents / LATENT_SCALING_FACTOR, return_dict=False)[0]
            pred_depths = decoded_images[:, 0:1]

            resized_masks = F.interpolate(masks, size=pred_depths.shape[-2:], mode="nearest")
            diff = (pred_depths - depths) * resized_masks
            loss = (diff ** 2).sum() / resized_masks.sum().clamp(min=1.0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step_loss = loss.item()
        epoch_loss += step_loss
        all_step_losses.append(step_loss)
        progress.set_postfix(loss=epoch_loss / (progress.n + 1))

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    epoch_bin_path = os.path.join(OUTPUT_DIR, f"lora_SD2_depths_epoch_{epoch+1}.bin")
    unet.save_attn_procs(epoch_bin_path)
    print(f"Saved LoRA Depths weights for epoch {epoch+1} to {epoch_bin_path}")

final_bin_path = os.path.join(OUTPUT_DIR, "lora_SD2_depths_final.bin")
unet.save_attn_procs(final_bin_path)
print(f"Training complete. Final LoRA Depths weights saved to {final_bin_path}")

# ========= PLOT LOSS CURVES =========
plt.figure(figsize=(10, 4))
plt.plot(all_step_losses, label="Step Loss", linewidth=0.7)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Per Step")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lora_SD2_depth_loss_per_step.png"))
plt.close()
print(f"Step loss plot saved to {OUTPUT_DIR}/lora_SD2_depth_loss_per_step.png")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.title("Training Loss Per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lora_SD2_depth_loss_curve.png"))
plt.close()
print(f"Epoch loss plot saved to {OUTPUT_DIR}/lora_SD2_depth_loss_curve.png")
