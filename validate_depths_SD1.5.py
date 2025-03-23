import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch.nn.functional as F

# ========== CONFIG ==========
DATASET_ROOT = "/home/exp1/zihan2/DIODE_dataset"
VAL_META_PATH = os.path.join(DATASET_ROOT, "diode_meta3.json")
CKPT_PATH = os.path.join(DATASET_ROOT, "SD1.5_outputs", "/home/exp1/zihan2/DIODE_dataset/output/lora_depths_final.bin")
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
OUTPUT_DIR = os.path.join(DATASET_ROOT, "val_depths_predicted_SD1.5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== DATASET ==========
class DepthValDataset(Dataset):
    def __init__(self, data_root, meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.samples = list(zip(meta["val"]["indoors"]["images"], meta["val"]["indoors"]["depth_maps"]))
        self.data_root = data_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, depth_rel = self.samples[idx]
        img_path = os.path.join(self.data_root, "val", "indoors", img_rel)
        depth_path = os.path.join(self.data_root, "val", "indoors", depth_rel)

        image = Image.open(img_path).convert("RGB")
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        depth_tensor = torch.tensor(np.load(depth_path), dtype=torch.float32)  # [H, W]

        return image_tensor, depth_tensor, img_rel

# ========== LOAD MODEL ==========
print("Loading model...")
pipeline = StableDiffusionPipeline.from_pretrained(PRETRAINED_MODEL)
unet = pipeline.unet.to(DEVICE, dtype=torch.float16)
vae: AutoencoderKL = pipeline.vae.to(DEVICE, dtype=torch.float16)
text_encoder = pipeline.text_encoder.to(DEVICE, dtype=torch.float16)
tokenizer = pipeline.tokenizer
unet.load_attn_procs('/home/exp1/zihan2/DIODE_dataset/output/lora_depths_final.bin')
pipeline.to(DEVICE)

# ========== LOAD DATA ==========
val_dataset = DepthValDataset(DATASET_ROOT, VAL_META_PATH)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ========== VALIDATION ==========
print("Running depth validation...")
mse_scores = []
rmse_scores = []
delta1_scores = []

for image, target, rel_path in tqdm(val_loader):
    image = image.to(DEVICE, dtype=torch.float16)
    target = target.to(DEVICE, dtype=torch.float32)

    pred_filename = rel_path[0].replace("/", "_").replace(".png", ".npy")
    pred_path = os.path.join(OUTPUT_DIR, pred_filename)

    # If cached prediction exists, load it
    if os.path.exists(pred_path):
        pred = torch.tensor(np.load(pred_path)).to(DEVICE, dtype=torch.float32)
    else:
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
            text_input = tokenizer(["intrinsic extraction depth"], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
            encoder_hidden_states = text_encoder(text_input)[0].to(dtype=torch.float16)
            timestep = torch.randint(0, 1000, (1,), device=DEVICE).long()
            pred_latents = unet(latents, timestep, encoder_hidden_states).sample
            pred = pred_latents.squeeze(0).mean(0).to(torch.float32)  # [H, W]

        np.save(pred_path, pred.cpu().numpy())

    # Resize GT if shape mismatch
    if target.shape[-2:] != pred.shape[-2:]:
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # [1, 1, H, W] → [1, H, W]
        elif target.dim() == 4 and target.shape[-1] == 1:
            target = target.squeeze(-1)  # [1, H, W, 1] → [1, H, W]
        elif target.dim() == 3 and target.shape[0] == 1:
            target = target.squeeze(0)  # [1, H, W] → [H, W]

        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)  # [H, W] → [1, 1, H, W]
        elif target.dim() == 3:
            target = target.unsqueeze(0)  # [1, H, W] → [1, 1, H, W]

        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)  # → [1, H, W]

    # Normalize to [0, 1]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)

    # Optional: clip predicted values
    pred = pred.clamp(0, 1)

    # Calculate MSE
    mse = torch.mean((pred - target) ** 2).item()
    mse_scores.append(mse)

    # Calculate RMSE
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    rmse_scores.append(rmse)

    # Calculate δ < 1.25
    max_ratio = torch.max(pred / target, target / pred)
    delta1 = (max_ratio < 1.25).float().mean().item()
    delta1_scores.append(delta1)

# ========== RESULTS ==========
mean_mse = np.mean(mse_scores)
mean_rmse = np.mean(rmse_scores)
mean_delta1 = np.mean(delta1_scores)

print(f"Depth validation completed.")
print(f"Mean MSE: {mean_mse:.6f}")
print(f"Mean RMSE: {mean_rmse:.6f}")
print(f"Threshold Accuracy (δ < 1.25): {mean_delta1:.2f}%")

 
