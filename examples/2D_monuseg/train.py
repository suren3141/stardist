from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

import cv2

import sys, os
sys.path.append('/workspace/hover_net')
sys.path.append('/workspace/stardist')

from dataloader.utils import get_file_list
from dataloader.train_loader import MoNuSegDataset

from torch.utils.data import DataLoader

np.random.seed(42)
lbl_cmap = random_label_cmap()


from PIL import Image
def read_img(filename, mode='RGB'):
    img = Image.open(filename)
    img_rgb = img.convert(mode)
    img_array = np.array(img_rgb)
    return img_array

import nvidia_smi

def get_total_mem():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)

    free_mem = info.free

    nvidia_smi.nvmlShutdown()

    return free_mem / 1e6

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y


def sem_to_inst_map(sem_map):
    num_labels, labels = cv2.connectedComponents(sem_map)

    inst_map = np.zeros_like(sem_map)

    for label in range(1, num_labels):
        inst_map[labels==label] = label

    return inst_map


def get_file_label(gt_dirs, gt=True):
    file_list = []
    file_labels = []


    for k, v in gt_dirs.items():
        if gt:
            f = get_file_list(v, ".png")
        else:
            f = get_file_list(v, ".png", img_path="samples", ann_path="labels")
        file_list.extend(f)
        file_labels.extend([f"{k}"] * len(f))
    
    return file_list, file_labels

if __name__ == "__main__":

    gt_dirs = {
        # "all": ["/mnt/dataset/MoNuSeg/patches_valid_256x256_128x128/MoNuSegTrainingData"],
        "train": ["/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4/MoNuSegTrainingData"],
        # "test": ["/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4/MoNuSegTestData"],
    }

    gt_file_list, _ = get_file_label(gt_dirs, gt=True)

    print(f"gt_file -> {len(gt_file_list)}")

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(gt_file_list))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    gt_file_val = [gt_file_list[i] for i in ind_val]
    gt_file_train = [gt_file_list[i] for i in ind_train]


    syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm/monuseg_patches_128.64CH_200st_1e-4lr_8bs_hv_ResNet18_kmeans_10_v1.1_4/ResNet18_kmeans_10_v1.1/*/"
    syn_dirs = sorted(glob(os.path.join(syn_pardir, "*")))

    def get_syn_name(x):
        c, x = os.path.split(x)
        c = os.path.split(c)[-1]
        return f"{c}_{x}"

    syn_dirs = { get_syn_name(x): x for x in syn_dirs}

    syn_file_list, _ = get_file_label(syn_dirs, gt=False)
    # syn_file_list = []

    print(f"syn_file -> {len(syn_file_list)}")

    train_file_list = gt_file_train + syn_file_list
    val_file_list = gt_file_val

    n_channel = 3
    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    img_preprocess = lambda x : normalize(x,1,99.8,axis=axis_norm)
    label_preprocess = lambda x : fill_label_holes(sem_to_inst_map(x))

    '''
    train_set = MoNuSegDataset(
        train_file_list, file_type=".png", mode="train", with_type=False, preprocess=(img_preprocess, label_preprocess),
        target_gen=(None, None), input_shape=(256,256), mask_shape=(256,256))

    val_set = MoNuSegDataset(
        val_file_list, file_type=".png", mode="test", with_type=False, preprocess=(img_preprocess, label_preprocess),
        target_gen=(None, None), input_shape=(256,256), mask_shape=(256,256))

    train_loader = DataLoader(train_set, num_workers= 8, batch_size= 8, shuffle=True, drop_last=False, )
    val_loader = DataLoader(val_set, num_workers= 8, batch_size= 8, shuffle=False, drop_last=False, )

    '''
    
    '''
    X_name = sorted(glob('/mnt/dataset/MoNuSeg/patches_valid_256x256_128x128/MoNuSegTrainingData/images/*.png'))
    Y_name = sorted(glob('/mnt/dataset/MoNuSeg/patches_valid_256x256_128x128/MoNuSegTrainingData/bin_masks/*.png'))
    print(len(X_name), len(Y_name))
    assert all(Path(x).name==Path(y).name for x,y in zip(X_name,Y_name))


    X = list(map(lambda x: read_img(x, 'RGB'), X_name))
    Y = list(map(lambda x: read_img(x, 'L'), Y_name))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(sem_to_inst_map(y)) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))
    '''

    X_trn = list(map(lambda x: img_preprocess(read_img(x[0], 'RGB')), train_file_list))
    Y_trn = list(map(lambda x: label_preprocess(read_img(x[1], 'L')), train_file_list))

    X_val = list(map(lambda x: img_preprocess(read_img(x[0], 'RGB')), val_file_list))
    Y_val = list(map(lambda x: label_preprocess(read_img(x[1], 'L')), val_file_list))

    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = gputools_available()
    print(use_gpu)

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2,2)

    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel,
    )
    print(conf)
    vars(conf)

    free_mem = get_total_mem()

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8, total_memory=free_mem)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    model = StarDist2D(conf, name='stardist_gt_syn', basedir='models_monuseg_4')

    '''
    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)
        
    '''

    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

