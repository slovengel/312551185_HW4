import random
import os
import numpy as np
import math
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm.auto import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, hflip, vflip
from PromptIR.net.model import PromptIR

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

BATCH_SIZE = 1
LEARNING_RATE = 2e-4
NUM_EPOCH = 25


class RealSEImageRestorationDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = 'hw4_realse_dataset\\train\\'
        self.degraded_dir = self.root_dir + 'degraded'
        self.clean_dir = self.root_dir + 'clean'
        self.transform = transform

        self.degraded_images = sorted([
            f for f in os.listdir(self.degraded_dir)
            if f.endswith('.png') and ('rain-' in f or 'snow-' in f)
        ])

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        degraded_filename = self.degraded_images[idx]

        if degraded_filename.startswith('rain-'):
            clean_filename = degraded_filename.replace('rain-', 'rain_clean-')
        elif degraded_filename.startswith('snow-'):
            clean_filename = degraded_filename.replace('snow-', 'snow_clean-')

        degraded_path = os.path.join(self.degraded_dir, degraded_filename)
        clean_path = os.path.join(self.clean_dir, clean_filename)

        degraded_img = Image.open(degraded_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        if random.random() > 0.5:
            degraded_img = hflip(degraded_img)
            clean_img = hflip(clean_img)

        if random.random() > 0.5:
            degraded_img = vflip(degraded_img)
            clean_img = vflip(clean_img)

        if self.transform:
            degraded_img = self.transform(degraded_img)
            clean_img = self.transform(clean_img)

        return degraded_img, clean_img


transform = Compose([
    ToTensor()
])

dataset = RealSEImageRestorationDataset(transform=transform)
train_dataflow = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = PromptIR(decoder=True)
model = model.cuda()


def train(
    model: nn.Module,
    dataflow: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler,
):
    model.train()

    total_loss = 0
    count = 0
    for degraded, clean in tqdm(dataflow, desc='train', leave=False):

        degraded = degraded.cuda()
        clean = clean.cuda()

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            output = model(degraded)
            loss = criterion(output, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        count += 1

    scheduler.step()

    avg_loss = total_loss / count
    return avg_loss


def calculate_psnr(restored, ground_truth, max_val=1.0):
    mse = F.mse_loss(restored, ground_truth)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr


def evaluate(
    model: nn.Module,
    dataflow: DataLoader,
):

    model.eval()
    total_psnr = 0.0
    num_images = 0

    count = 0

    with torch.no_grad():
        for degraded, clean in tqdm(dataflow, desc='eval', leave=False):
            count += 1

            degraded = degraded.cuda()
            clean = clean.cuda()

            restored = model(degraded)
            restored = torch.clamp(restored, 0.0, 1.0)

            for i in range(restored.size(0)):
                psnr = calculate_psnr(restored[i], clean[i])
                total_psnr += psnr
                num_images += 1

    avg_psnr = total_psnr / num_images
    return avg_psnr


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=NUM_EPOCH,
    eta_min=1e-6
)
scaler = torch.amp.GradScaler()

train_losses = []
train_PSNRs = []

model.cuda()

for epoch in range(NUM_EPOCH):
    print("Epoch", epoch + 1, ":")

    torch.cuda.empty_cache()
    train_loss = train(
        model,
        train_dataflow,
        criterion,
        optimizer,
        scheduler,
        scaler
    )
    torch.cuda.empty_cache()
    train_PSNR = evaluate(model, train_dataflow)
    torch.cuda.empty_cache()

    train_losses.append(train_loss)
    train_PSNRs.append(train_PSNR)

    print(
        f"Epoch {epoch + 1}:\t"
        f"Train Loss: {train_loss:.4f} "
        f"Train PSNR: {train_PSNR:.4f}"
    )

    torch.save(model.state_dict(), f"promptIR_{epoch+1}.pt")


class RealSEImageRestorationDataset_test(Dataset):
    def __init__(self, transform=None):
        self.root_dir = 'hw4_realse_dataset\\test\\'
        self.degraded_dir = self.root_dir + 'degraded'
        self.transform = transform

        self.degraded_images = sorted([
            f for f in os.listdir(self.degraded_dir)
            if f.endswith('.png')
        ])

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        degraded_filename = self.degraded_images[idx]
        degraded_path = os.path.join(self.degraded_dir, degraded_filename)
        degraded_img = Image.open(degraded_path).convert("RGB")

        if self.transform:
            degraded_img = self.transform(degraded_img)

        image_id = degraded_filename.split('.')[0]

        return degraded_img, image_id


test_dataset = RealSEImageRestorationDataset_test(transform=transform)
test_dataflow = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def inference(
    model: nn.Module,
    dataflow: DataLoader,
):

    save_dir = "results\\"
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    count = 0

    with torch.no_grad():
        for degraded, image_id in tqdm(dataflow, desc='eval', leave=False):
            count += 1

            degraded = degraded.cuda()

            restored = model(degraded)
            restored = torch.clamp(restored, 0.0, 1.0)

            for i in range(restored.size(0)):
                img = to_pil_image(restored[i].cpu())
                img.save(os.path.join(save_dir, f"{image_id[i]}.png"))


model.load_state_dict(torch.load('promptIR_25.pt', weights_only=True))
model.cuda()
inference(model, test_dataflow)
