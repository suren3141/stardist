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

def hstack_pil_image(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im
    

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
            

def get_GT_syn_files(gt_dir=None, syn_pardir=None, inst_path='masks_in_silico_inst'):
    use_inst_mask = True
    filters = [f'*_{i}' for i in range(5)]

    gt_dirs = {
        "train": [gt_dir],
    }

    gt_file_list, _ = get_file_label(gt_dirs, gt=True)

    print(f"gt_file -> {len(gt_file_list)}")

    # syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm/monuseg_patches_128.64CH_200st_1e-4lr_8bs_hv_ResNet18_kmeans_10_v1.1_4/ResNet18_kmeans_10_v1.1/*/"
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
        s, _ = get_file_label(syn_dirs, img_path='images_in_silico_inst', inst_path=inst_path, filt=f)
        syn_file_list.extend(s)

    syn_file_list_filtered = []

    for f in syn_file_list:
        valid = any([Path(f[0]).stem.startswith(Path(i[0]).stem) for i in gt_file_list])
        if valid : syn_file_list_filtered.append(f)

    print(f"syn_file -> {len(syn_file_list_filtered)}")

    return gt_file_list, syn_file_list_filtered



def main():
    use_inst_mask=True
    gt_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/10ss"
    syn_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/10ss"

    gt_file_list, syn_file_list_filtered = get_GT_syn_files(gt_pardir, syn_pardir)


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

    # X_gt = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(gt_file_list)))
    # Y_gt = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(gt_file_list)))

    X_syn = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(syn_file_list_filtered)))
    Y_syn = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(syn_file_list_filtered)))


    # vals, features = [], []
    var_scale = 2

    for i, (x, y) in tqdm(enumerate(zip(X_syn, Y_syn)), total=len(X_syn)):
        val, feat = get_nuclie_feature(img=x, inst_mask=y, mode="mean")
        # vals += list(val)
        # features += feat
        name = Path(syn_file_list_filtered[i][0]).stem

        var_feat = np.var(feat[1:], axis=0)
        dist2_0 = np.square(np.abs(feat - feat[0]))
        drop_mask = np.all(dist2_0 < var_feat * var_scale, axis=-1)

        if False:
            img = read_img(syn_file_list_filtered[i][x_id], 'RGB')
            msk = Y_syn[i]
            img_pil = plot_overlay(img, msk)
            img_pil.save(f'tmp/filt2/image_with_edge_{i}.png')

            fig = scatter_feature(np.array(feat), val, np.where(drop_mask, 'r', 'b'))
            fig.savefig(f'tmp/filt2/scatter_with_idx_{i}.png')

            # write_embedding(f'/tmp/nuclei_feature_{name}', None, feat, list(val))
    
        update_mask(syn_file_list_filtered[i][y_id], drop_mask, out_path=os.path.join(syn_pardir, 'masks_in_silico_inst_filt2'))



def visualize_processed_output():
    use_inst_mask=True
    gt_pardir = "/mnt/dataset/MoNuSeg/patches_valid_inst_256x256_128x128/25ss/MoNuSegTrainingData"
    syn_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/25ss"

    gt_file_list, syn_file_list_filtered = get_GT_syn_files(gt_pardir, syn_pardir)
    gt_file_list, syn_file_list_filtered_updated = get_GT_syn_files(gt_pardir, syn_pardir, inst_path='masks_in_silico_inst_filt')

    x_id = 0
    y_id = 2 if use_inst_mask else 1
    mask_dtype = 'I' if use_inst_mask else 'L'

    for i, (syn, syn_new) in enumerate(zip(syn_file_list_filtered, syn_file_list_filtered_updated)):
        img = read_img(syn[x_id], 'RGB')
        mask1 = read_img(syn[y_id], mask_dtype)
        mask2 = read_img(syn_new[y_id], mask_dtype)

        img1 = plot_overlay(img, mask1, mark_id=False)
        img2 = plot_overlay(img, mask2, mark_id=False)
        images = hstack_pil_image([img1, img2])

        images.save(f'tmp/masks/mask_update_{i}.png')






if __name__ == "__main__":
    main()


