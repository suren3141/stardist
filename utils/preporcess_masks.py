import numpy as np
from PIL import Image, ImageDraw 
import os,sys, glob
from pathlib import Path
from matplotlib import pyplot as plt 
sys.path.append('/workspace/stardist')

from utils import *
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available

from hover_net.misc.embeddings import write_embedding

from tqdm import tqdm

def get_nuclie_feature(img, inst_mask, mode="mean"):
    """Each nuclei will be defined by a feature so it could be clustered and anomalies removed"""
    if mode=="mean":
        f = np.mean
    elif mode=="median":
        f = np.median
    else:
        raise NotImplementedError

    vals = np.unique(inst_mask)
    feature = [f(img[inst_mask == v], axis=0) for v in vals]

    return vals, feature

def scatter_feature(feat, label, col=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(feat[:, 0], feat[:, 1], feat[:, 2], color=col)
    for i, l in enumerate(label):
        ax.text(feat[i, 0], feat[i, 1], feat[i, 2], l, fontsize='xx-small')

    return fig
    
def plot_overlay(image, mask, col=[0, 255, 0], mark_id=True):
    img = np.array(image)
    edge = get_edges(mask)
    ind = np.where(edge)
    img[ind[0], ind[1], :] = col
    pil_img = Image.fromarray(img)

    if mark_id:
        draw = ImageDraw.Draw(pil_img)
        uniq = np.unique(mask)
        for u in uniq:
            if u == 0: continue
            pos = np.where(mask==u)
            x, y = np.mean(pos, axis=1)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((int(y), int(x)),f"{u}",(255,255,255))

    return pil_img


def get_edges(t):
    edge = np.zeros_like(t).astype(np.bool8)
    if edge.ndim == 2: # only h, w
        edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
        edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
        edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
        edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
    else:
        raise NotImplementedError
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge

def update_mask(inst_path, drop_mask, out_path=None):
    file_name = os.path.split(inst_path)[-1]

    inst_mask = read_img(inst_path, 'I')
    vals = np.unique(inst_mask)
    assert len(vals) == len(drop_mask)
    for v, m in zip(vals, drop_mask):
        if m:
            inst_mask = np.where(inst_mask == v, 0, inst_mask)

    if out_path is not None:
        if not os.path.exists(out_path): os.makedirs(out_path)
        Image.fromarray(inst_mask).save(os.path.join(out_path, file_name))

    return inst_mask
            





if __name__ == "__main__":
    use_inst_mask = True
    filters = [f'*_{i}' for i in range(5)]

    gt_dirs = {
        "train": ["/mnt/dataset/MoNuSeg/patches_valid_inst_256x256_128x128/25ss/MoNuSegTrainingData"],
    }

    gt_file_list, _ = get_file_label(gt_dirs, gt=True)

    print(f"gt_file -> {len(gt_file_list)}")

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(gt_file_list))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    gt_file_val = [gt_file_list[i] for i in ind_val]
    gt_file_train = [gt_file_list[i] for i in ind_train]


    # syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm/monuseg_patches_128.64CH_200st_1e-4lr_8bs_hv_ResNet18_kmeans_10_v1.1_4/ResNet18_kmeans_10_v1.1/*/"
    syn_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/25ss"
    syn_dirs = sorted(glob.glob(os.path.join(syn_pardir, "*")))

    def get_syn_name(x):
        c, x = os.path.split(x)
        c = os.path.split(c)[-1]
        return f"{c}_{x}"

    syn_dirs = { get_syn_name(x): x for x in syn_dirs}

    syn_dirs = {
        "syn" : syn_pardir
    }

    syn_file_list = []
    for f in filters:
        s, _ = get_file_label(syn_dirs, img_path='images_in_silico_inst', inst_path='masks_in_silico_inst', filt=f)
        syn_file_list.extend(s)

    syn_file_list_filtered = []

    for f in syn_file_list:
        valid = any([Path(f[0]).stem.startswith(Path(i[0]).stem) for i in gt_file_list])
        if valid : syn_file_list_filtered.append(f)

    print(f"syn_file -> {len(syn_file_list_filtered)}")

    train_file_list = gt_file_train + syn_file_list_filtered
    val_file_list = gt_file_val


    n_channel = 3
    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    img_preprocess = lambda x : normalize(x,1,99.8,axis=axis_norm)

    if use_inst_mask:
        label_preprocess = lambda x : fill_label_holes(x)
    else:
        raise NotImplementedError

    x_id = 0
    y_id = 2 if use_inst_mask else 1
    mask_dtype = 'I' if use_inst_mask else 'L'

    X_gt = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(gt_file_list)))
    Y_gt = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(gt_file_list)))

    X_syn = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(syn_file_list_filtered)))
    Y_syn = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(syn_file_list_filtered)))


    # vals, features = [], []

    for i, (x, y) in tqdm(enumerate(zip(X_syn, Y_syn))):
        val, feat = get_nuclie_feature(img=x, inst_mask=y, mode="mean")
        # vals += list(val)
        # features += feat
        name = Path(syn_file_list_filtered[i][0]).stem

        var_feat = np.var(feat[1:], axis=0)
        dist2_0 = np.square(np.abs(feat - feat[0]))
        drop_mask = np.all(dist2_0 < var_feat, axis=-1)

        if True:
            img = read_img(syn_file_list_filtered[i][x_id], 'RGB')
            msk = Y_syn[i]
            img_pil = plot_overlay(img, msk)
            img_pil.save(f'tmp/image_with_edge_{i}.png')

            fig = scatter_feature(np.array(feat), val, np.where(drop_mask, 'r', 'b'))
            fig.savefig(f'tmp/scatter_with_idx_{i}.png')

        update_mask(syn_file_list_filtered[i][y_id], drop_mask, out_path=os.path.join(syn_pardir, 'masks_in_silico_inst_filt'))

        write_embedding(f'/tmp/nuclei_feature_{name}', None, feat, list(val))






