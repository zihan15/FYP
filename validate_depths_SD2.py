import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

# ========== CONFIG ==========
DATASET_ROOT = "/home/exp1/zihan2/DIODE_dataset"
VAL_META_PATH = os.path.join(DATASET_ROOT, "diode_meta3.json")
CKPT_PATH = os.path.join(DATASET_ROOT, "SD2.1_outputs", "/home/exp1/zihan2/DIODE_dataset/output/lora_SD2_depths_final.bin")
PRETRAINED_MODEL = "stabilityai/stable-diffusion-2-1-base"
TOKENIZER_NAME = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
IMAGE_SIZE = 768
LATENT_SCALING_FACTOR = 0.13025
OUTPUT_DIR = os.path.join(DATASET_ROOT, "val_depths_predicted_SD2.1")
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
        image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        depth_tensor = torch.tensor(np.load(depth_path), dtype=torch.float32)  # [H, W]

        return image_tensor, depth_tensor, img_rel

# ========== LOAD MODEL ==========
print("Loading SD 2.1 model...")
pipeline = StableDiffusionPipeline.from_pretrained(
    PRETRAINED_MODEL,
    torch_dtype=torch.float16,
    revision="fp16"
)
unet: UNet2DConditionModel = pipeline.unet.to(DEVICE)
vae: AutoencoderKL = pipeline.vae.to(DEVICE)
text_encoder: CLIPTextModelWithProjection = pipeline.text_encoder.to(DEVICE)
tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_NAME)

unet.load_attn_procs('/home/exp1/zihan2/DIODE_dataset/output/lora_SD2_depths_final.bin')
pipeline.to(DEVICE)

# ========== LOAD DATA ==========
val_dataset = DepthValDataset(DATASET_ROOT, VAL_META_PATH)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ========== VALIDATION ==========
print("Running depth validation (SD 2.1)...")
mse_scores = []
rmse_scores = []
delta1_scores = []

for image, target, rel_path in tqdm(val_loader):
    image = image.to(DEVICE, dtype=torch.float16)
    target = target.to(DEVICE, dtype=torch.float32)

    pred_filename = rel_path[0].replace("/", "_").replace(".png", ".npy")
    pred_path = os.path.join(OUTPUT_DIR, pred_filename)

    if os.path.exists(pred_path):
        pred = torch.tensor(np.load(pred_path)).to(DEVICE, dtype=torch.float32)
    else:
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * LATENT_SCALING_FACTOR
            text_input = tokenizer(["depth map"], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
            encoder_hidden_states = text_encoder(text_input).last_hidden_state.to(dtype=torch.float16)
            timestep = torch.randint(0, 1000, (1,), device=DEVICE).long()
            pred_latents = unet(latents, timestep, encoder_hidden_states).sample
            pred = pred_latents.squeeze(0).mean(0).to(torch.float32)

        np.save(pred_path, pred.cpu().numpy())

    # Resize GT if shape mismatch
    if target.shape[-2:] != pred.shape[-2:]:
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        elif target.dim() == 4 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        elif target.dim() == 3 and target.shape[0] == 1:
            target = target.squeeze(0)

        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3:
            target = target.unsqueeze(0)

        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

    # Normalize to [0, 1]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)
    pred = pred.clamp(0, 1)

    # Calculate MSE
    mse = torch.mean((pred - target) ** 2).item()
    mse_scores.append(mse)

    # Calculate RMSE
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    rmse_scores.append(rmse)

    # Calculate delta < 1.25
    epsilon = 1e-6
    pred_safe = pred.clamp(min=epsilon)
    target_safe = target.clamp(min=epsilon)
    max_ratio = torch.max(pred_safe / target_safe, target_safe / pred_safe)
    delta1 = (max_ratio < 1.25).float().mean().item()
    delta1_scores.append(delta1)

# ========== RESULTS ==========
mean_mse = np.mean(mse_scores)
mean_rmse = np.mean(rmse_scores)
mean_delta1 = np.mean(delta1_scores)

print("Depth validation completed (SD 2.1).")
print(f"Mean MSE: {mean_mse:.6f}")
print(f"Mean RMSE: {mean_rmse:.6f}")
print(f"Threshold Accuracy (δ < 1.25): {mean_delta1 * 100:.2f}%")
