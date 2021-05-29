import shutil

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

    os.makedirs('/content/drive/MyDrive/Wildfire_project/challenge1/{}/'.format(MODEL_NAME))
    shutil.copytree('/content/saved_images',
                    '/content/drive/MyDrive/Wildfire_project/challenge1/{}/saved_images'.format(MODEL_NAME))
    shutil.copy('/content/my_checkpoint.pth.tar',
                '/content/drive/MyDrive/Wildfire_project/challenge1/{}/my_checkpoint.pth.tar'.format(MODEL_NAME))

    return ((dice_score / len(loader)), (loss_total / len(loader)))

