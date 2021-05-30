import torch

# Hyperparameters etc
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '/content/wildfire_dataset/data_zip(2)/train/images'
TRAIN_MASK_DIR = '/content/wildfire_dataset/data_zip(2)/train/corrected_labels'
VAL_IMG_DIR = '/content/wildfire_dataset/data_zip(2)/val/images'
VAL_MASK_DIR = '/content/wildfire_dataset/data_zip(2)/val/corrected_labels'
TEST_IMG_DIR = '/content/wildfire_dataset/data_zip(2)/test/images'
TEST_RES_DIR = '/content/wildfire_dataset/data_zip(2)/test/results'
MODEL_NAME = 'test_model1'
DATA_LINK = 'https://www.dropbox.com/s/eso7s52w73s0h5f/wildfire_dataset.zip?dl=0'

fnames = ['JORDAN_235_P1_201901281204_MGA94_55',
       'JORDAN_294_P1_201902011150_MGA94_55',
       'WALHALLA_313_P1_201902020733_MGA94_55',
       'WALHALLA_353_P1_201902031625_MGA94_55',
       'MACALISTER91_648_P1_201903070444_MGA94_55']

names = {'MACALISTER91_648_P1_201903070444_MGA94_55': 'pred_1.png',
         'JORDAN_294_P1_201902011150_MGA94_55': 'pred_3.png',
         'JORDAN_235_P1_201901281204_MGA94_55': 'pred_4.png',
         'WALHALLA_353_P1_201902031625_MGA94_55': 'pred_0.png',
         'WALHALLA_313_P1_201902020733_MGA94_55': 'pred_2.png'}
