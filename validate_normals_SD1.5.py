import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ========== CONFIG ==========
DATASET_ROOT = "/home/exp1/zihan2/DIODE_dataset"
VAL_META_PATH = os.path.join(DATASET_ROOT, "diode_meta3.json")
CKPT_PATH = os.path.join(DATASET_ROOT, "SD1.5_outputs", "/home/exp1/zihan2/DIODE_dataset/output/lora_normal_final.bin")
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
OUTPUT_DIR = os.path.join(DATASET_ROOT, "val_normals_predicted_SD1.5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== DATASET ==========
class NormalValDataset(Dataset):
    def __init__(self, data_root, meta_path):
        self.data_root = data_root
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.samples = self.meta["val_normals"]["indoors"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        relative_path = self.samples[idx]
        img_path = os.path.join(self.data_root, "val", "indoors", relative_path.replace("_normal.npy", ".png"))
        normal_path = os.path.join(self.data_root, "val_normals", "indoors", relative_path)

        image = Image.open(img_path).convert("RGB")
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        normal_gt = torch.tensor(np.load(normal_path), dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]

        return image, normal_gt, relative_path

# ========== METRICS ==========
def angular_error(pred, gt):
    cos_sim = F.cosine_similarity(pred, gt, dim=0).clamp(-1, 1)
    angles = torch.acos(cos_sim) * (180.0 / np.pi)  # degrees
    return angles

# ========== DATALOADER ==========
val_dataset = NormalValDataset(DATASET_ROOT, VAL_META_PATH)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ========== MODEL ==========
print("🔧 Loading pretrained components...")
pipeline = StableDiffusionPipeline.from_pretrained(PRETRAINED_MODEL)
unet = pipeline.unet.to(DEVICE, dtype=torch.float16)
vae: AutoencoderKL = pipeline.vae.to(DEVICE, dtype=torch.float16)
text_encoder = pipeline.text_encoder.to(DEVICE, dtype=torch.float16)
tokenizer = pipeline.tokenizer
unet.load_attn_procs('/home/exp1/zihan2/DIODE_dataset/output/lora_normal_final.bin')
pipeline.to(DEVICE)

# ========== VALIDATION LOOP ==========
print("Starting validation...")
angular_errors = []
mae_scores = []
l1_scores = []

for image, target, rel_path in tqdm(val_loader):
    image = image.to(DEVICE, dtype=torch.float16)
    target = target.to(DEVICE, dtype=torch.float32)

    pred_filename = rel_path[0].replace("/", "_").replace("_normal.npy", ".png")
    pred_path = os.path.join(OUTPUT_DIR, pred_filename)

    if os.path.exists(pred_path):
        pred_img = Image.open(pred_path).convert("RGB")
        pred = TF.to_tensor(pred_img).to(DEVICE, dtype=torch.float32)
    else:
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
            prompt = ["depth map"]
            text_input = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
            encoder_hidden_states = text_encoder(text_input)[0].to(dtype=torch.float16)
            timesteps = torch.randint(0, 1000, (1,), device=DEVICE).long()
            pred_latents = unet(latents, timesteps, encoder_hidden_states).sample
            pred = vae.decode(pred_latents / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(0, 1).to(torch.float32)

        save_img = (pred * 255).byte().cpu().permute(1, 2, 0).numpy()
        Image.fromarray(save_img).save(pred_path)

    # Resize GT if needed
    if target.shape[-2:] != pred.shape[-2:]:
        print(f"Raw target shape before reshape: {target.shape}")
        if target.dim() == 4 and target.shape[-1] == 3:
            target = target.squeeze(0).permute(2, 0, 1)  # (1, H, W, 3) → (3, H, W)
        elif target.dim() == 3 and target.shape[-1] == 3:
            target = target.permute(2, 0, 1)  # (H, W, 3) → (3, H, W)

        target = F.interpolate(target.unsqueeze(0), size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

    # Normalize to [-1, 1] for angular comparison
    pred = (pred * 2) - 1
    pred = F.normalize(pred, dim=0)
    target = F.normalize(target, dim=0)

    # Compute angular error
    angles = angular_error(pred, target)
    angular_errors.append(angles.mean().item())
    mae_scores.append(angles.abs().mean().item())

    # L1 error
    l1 = torch.abs(pred - target).mean().item()
    l1_scores.append(l1)

# ========== RESULTS ==========
print(f"Validation complete.")
print(f"Mean Angular Error: {np.mean(angular_errors):.4f}°")
print(f"Median Angular Error: {np.median(angular_errors):.4f}°")
print(f"Mean L1 Error: {np.mean(l1_scores)*100:.6f}")