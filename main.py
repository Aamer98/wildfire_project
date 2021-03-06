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
from test import *

def main(model):
    train_transform = A.Compose(
        [
            AT.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            # A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            AT.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    set_seed(0)

    
    loss_fn = nn.BCEWithLogitsLoss()  # dont need sigmoid
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter()
    if not os.path.exists('/content/saved_images'):
      os.makedirs('/content/saved_images')
    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS))
        train_dice, train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # check accuracy
        dice_sc = check_accuracy(val_loader, model, device=DEVICE)

        writer.add_scalar('dice/train', train_dice, epoch)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('dice/val', dice_sc, epoch)

        # print examples
        save_predictions_as_imgs(
            val_loader, model, folder='/content/saved_images/', device=DEVICE
        )

    writer.close()
    if not os.path.exists('/content/drive/MyDrive/Wildfire_project/challenge1/{}/'.format(MODEL_NAME)):
      os.makedirs('/content/drive/MyDrive/Wildfire_project/challenge1/{}/'.format(MODEL_NAME))
    
    if not os.path.exists('/content/drive/MyDrive/Wildfire_project/challenge1/{}/saved_images'.format(MODEL_NAME)):

      shutil.copytree('/content/saved_images',
                    '/content/drive/MyDrive/Wildfire_project/challenge1/{}/saved_images'.format(MODEL_NAME))
      shutil.copy('/content/my_checkpoint.pth.tar',
                '/content/drive/MyDrive/Wildfire_project/challenge1/{}/my_checkpoint.pth.tar'.format(MODEL_NAME))




# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    main(model)
    
    _,test_loader = get_loaders(
      TRAIN_IMG_DIR,
      TRAIN_MASK_DIR,
      TEST_IMG_DIR,
      TEST_IMG_DIR,
      BATCH_SIZE,
      val_transforms,
      val_transforms,
      NUM_WORKERS,
      PIN_MEMORY
    )
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    save_test_pickle()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
