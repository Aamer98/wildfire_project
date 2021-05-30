import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

import os
from PIL import Image
import numpy as np
import albumentations as A
from tqdm import tqdm
import zipfile
from albumentations.pytorch.transforms import ToTensorV2
import logging
import random
from efficientunet import *
import albumentations.augmentations.crops.transforms as AT
import shutil
import urllib
from config import *
from utils import *
from train import *
from test import *
from model import *
from data import *


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    loss_total = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        preds = (predictions > 0.5).float()
        num_correct += (preds == targets).sum()
        num_pixels += torch.numel(preds)
        dice_score += (2 * (preds * targets).sum()) / (
                (preds + targets).sum() + 1e-8
        )
        loss_total += loss

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm
        loop.set_postfix(loss=loss.item())
    print("Dice score: {}, Loss: {}".format((dice_score / len(loader)), (loss_total / len(loader))))


    return ((dice_score / len(loader)), (loss_total / len(loader)))
