from data import get_loaders
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from config import *


val_transforms = A.Compose(
      [
       A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
       #AT.RandomCrop(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
       A.Normalize(
           mean = [0.0, 0.0, 0.0],
           std = [1.0, 1.0, 1.0],
           max_pixel_value = 255.0,
       ),
       ToTensorV2(),
      ]
  )


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

def save_test_predictions(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            x, f"{folder}/{idx}.png"
        )


def save_test_pickle():
    for file_stem in fnames:
        img_dir = file_stem + '.png'
        test_img = cv2.imread(img_dir)

        name_pickles = fnames.copy
        img_dir = '/content/wildfire_dataset/data_zip(2)/test/images/' + file_stem + '.png'
        lbl_dir = '/content/wildfire_dataset/data_zip(2)/test/results/' + names[file_stem]

        test_img = cv2.imread(img_dir)
        pred_lbl = cv2.imread(lbl_dir, 0)

        test_img.shape
        ht = test_img.shape[0]
        wt = test_img.shape[1]

        resize_lbl = cv2.resize(pred_lbl, (ht, wt))  # , interpolation=cv2.INTER_CUBIC)

        (thresh, resize_lbl) = cv2.threshold(resize_lbl, 127, 255, cv2.THRESH_BINARY)
        resize_lbl[resize_lbl == 255] = 1

        final_lbl = np.reshape(np.array(resize_lbl), (1, ht, wt))
        mask = final_lbl == 1

        with open('{}.pickle'.format(file_stem), 'wb') as handle:
            pickle.dump(mask, handle)

