import numpy as np
from matplotlib import pyplot as plt
import json, os, glob

def get_json_data(path, file_name='stats.json', metric_name=['f1', 'dice', 'sample']):

    json_path = os.path.join(path, file_name)

    files = glob.glob(json_path)

    dic = {}

    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            vals = {}
            for m in metric_name:
                if m == "dice":
                    val = data['dice']
                elif m == "sample":
                    val = None if 'sample' not in data.keys() else data['sample']
                else:
                    val = data['0.5'][m]
                vals[m] = val
            
            exp = os.path.split(os.path.split(f)[0])[-1]
            dic[exp] = vals

    return dic

import seaborn as sns

def plot_samples(metrics, out_name):

    exp_128_128_05 = ['stardist_128_128_05gt_inst', 'stardist_128_128_05gt_05syn_inst', 'stardist_128_128_05gt_05syn.x2_inst', 'stardist_128_128_05gt_05syn.x3_inst', 'stardist_128_128_05gt_05syn.x4_inst', 'stardist_128_128_05gt_05syn.x5_inst']

    plot_sample(metrics, "exp_128_128_05", exp_128_128_05)

    exp_128_128_05_v3 = ['stardist_128_128_05gt_inst_v3', 'stardist_128_128_05gt_05syn_inst_v3', 'stardist_128_128_05gt_05syn.x2_inst_v3', 'stardist_128_128_05gt_05syn.x3_inst_v3', 'stardist_128_128_05gt_05syn.x4_inst_v3', 'stardist_128_128_05gt_05syn.x5_inst_v3']

    plot_sample(metrics, "exp_128_128_05_v3", exp_128_128_05_v3)


    

def plot_sample(metric, out_name, exps):

    f1 = dict(
        F1 = [],
        exp = [],
        mean = []
    )

    dice = dict(
        Dice = [],
        exp = [],
        mean = []
    )

    x_vals = []

    for exp in exps:
        if exp not in metric or metric[exp]['sample'] is None: continue
        f1_test = [stat['f1'] for stat in metric[exp]['sample']['stats']]
        dice_test = metric[exp]['sample']['dice']

        f1['F1'].extend(f1_test)
        f1['exp'].extend([exp] * len(f1_test))
        f1['mean'].append(np.mean(f1_test))

        dice['Dice'].extend(dice_test)
        dice['exp'].extend([exp] * len(dice_test))
        dice['mean'].append(np.mean(dice_test))

        x_vals.append(exp)

    x_vals = range(len(x_vals))



    # metrics = dict(
    #     test = metric_test,
    #     test_size = test_label,
    #     syn = metric_syn,
    #     syn_size = syn_label,
    # )

    # metrics = dict(
    #     F1 = metric_test + metric_syn,
    #     type = ['test'] * len(metric_test) + ['syn'] * len(metric_syn),
    #     size = test_label + syn_label,
    # )

    def plot_f1_sample(f1, x_vals):
        plt.figure(figsize=(16, 8))
        f1_mean = f1.pop('mean')
        sns.violinplot(data=f1, y="F1", x="exp", ax=plt.gca())
        ax = plt.gca()
        plt.title('F1 scores - Stardist trained on 128x128 images')
        # plt.xticks(rotation=30)
        plt.xticks(x_vals, [f'x{i}' for i in x_vals])

        for l in ax.lines:
            # print(l.get_data())
            if len(l.get_data()[1]) != 1 : continue
            ax.text(l.get_data()[0][0], l.get_data()[1][0], f'{l.get_data()[1][0]:.2f}',size='large')    

        plt.plot(x_vals, f1_mean, zorder=1, c='r', marker='x', label="mean")

        os.makedirs(f'tmp/metrics/{out_name}', exist_ok=True)
        plt.savefig(f'tmp/metrics/{out_name}/sample_f1.png')
        plt.close()



    def plot_dice_sample(dice, x_vals):

        plt.figure(figsize=(16, 8))
        dice_mean = dice.pop('mean')
        sns.violinplot(data=dice, y="Dice", x="exp", ax=plt.gca())
        ax = plt.gca()
        plt.title('Dice scores - Stardist trained on 256x256 images')
        plt.xticks(x_vals, [f'x{i}' for i in x_vals])

        for l in ax.lines:
            # print(l.get_data())
            if len(l.get_data()[1]) != 1 : continue
            ax.text(l.get_data()[0][0], l.get_data()[1][0], f'{l.get_data()[1][0]:.2f}',size='large')    

        # for l in ax.lines:
        #     # print(l.get_data())
        #     if len(l.get_data()[1]) != 1 : continue
        #     ax.text(l.get_data()[0][l.get_data()[0].nonzero()][0], l.get_data()[1][0], f'{l.get_data()[1][0]:.2f}',size='large')    
        plt.plot(x_vals, dice_mean, zorder=1, c='r', label="mean")

        plt.savefig(f'tmp/metrics/{out_name}/sample_dice.png')
        plt.close()

    plot_f1_sample(f1, x_vals)
    plot_dice_sample(dice, x_vals)

def plot_metrics(metrics, names=['f1', 'dice']):

    for m in names:
        metric = {exp : metrics[exp][m] for exp in metrics.keys()}
        plot_metric_128(metric, out_name = f"plot_{m}_128.png", ylabel=f"{m} score")
        plot_metric_256(metric, out_name = f"plot_{m}_256", ylabel=f"{m} score")

def get_exp_vals(metric, prefix, suffix=None, word=None, ignore=None):

    def get_size(name):
        if ".x2" in name: return 2
        elif ".x3" in name: return 3
        elif ".x4" in name: return 4
        elif ".x5" in name: return 5
        elif "syn" in name: return 1
        else: return 0

    filt = []
    for exp in metric.keys():
        if exp.startswith(prefix):
            if word is not None and word not in exp: continue
            if ignore is not None and ignore in exp: continue
            filt.append(exp)

    y = [metric[i] for i in filt]
    x = [get_size(i) for i in filt]
    xy = sorted(list(zip(x, y)))

    x, y = zip(*xy)

    return x, y


def plot_metric_256(metric, out_name=None, ylabel=None):
    ## number of training images 
    exp_10 = ['stardist_10gt_inst', 'stardist_10gt_10syn_inst', 'stardist_10gt_10syn.x2_inst', 'stardist_10gt_10syn.x3_inst', 'stardist_10gt_10syn.x4_inst', 'stardist_10gt_10syn.x5_inst']
    x_10 = range(len(exp_10))
    x_10 = [i*181 for i in x_10]
    y_10 = [metric[i] for i in exp_10]
    x_10, y_10 = get_exp_vals(metric, "stardist_10gt", ignore="filt")
    col_10 = "C0"

    exp_25 = ['stardist_25gt_inst', 'stardist_25gt_25syn_inst', 'stardist_25gt_25syn.x2_inst', 'stardist_25gt_25syn.x3_inst', 'stardist_25gt_25syn.x4_inst', 'stardist_25gt_25syn.x5_inst']
    x_25 = range(len(exp_25))
    x_25 = [i*453 for i in x_25]
    y_25 = [metric[i] for i in exp_25]
    col_25 = "C1"

    exp_05 = ['stardist_05gt_inst', 'stardist_05gt_05syn_inst', 'stardist_05gt_05syn.x2_inst', 'stardist_05gt_05syn.x3_inst', 'stardist_05gt_05syn.x4_inst', 'stardist_05gt_05syn.x5_inst']
    x_05 = range(len(exp_05))
    x_05 = [i*90 for i in x_05]
    y_05 = [metric[i] for i in exp_05]
    col_05 = "C4"

    exp_25_filt = ['stardist_25gt_inst', 'stardist_25gt_25syn_inst_filt', 'stardist_25gt_25syn.x2_inst_filt', 'stardist_25gt_25syn.x3_inst_filt', 'stardist_25gt_25syn.x4_inst_filt', 'stardist_25gt_25syn.x5_inst_filt']
    x_25 = range(len(exp_25_filt))
    x_25 = [i*453 for i in x_25]
    y_25_filt = [metric[i] for i in exp_25_filt]
    col_25_filt = "C2"

    exp_10_filt = ['stardist_10gt_inst', 'stardist_10gt_10syn_inst_filt', 'stardist_10gt_10syn.x2_inst_filt', 'stardist_10gt_10syn.x3_inst_filt', 'stardist_10gt_10syn.x4_inst_filt', 'stardist_10gt_10syn.x5_inst_filt']
    x_10 = range(len(exp_10_filt))
    x_10 = [i*453 for i in x_10]
    y_10_filt = [metric[i] for i in exp_10_filt]
    col_10_filt = "C3"


    print(metric)

    # x_50 = [1432, 1432*2]

    # colors
    fig, ax = plt.subplots()

    plt.plot(range(len(x_10)), y_10, c=col_10, label="10%")
    plt.plot(range(len(x_25)), y_25, c=col_25, label="25%")
    plt.plot(range(len(x_25)), y_25_filt, c=col_25_filt, label="25% (w/ postprocess)")
    plt.plot(range(len(x_10)), y_10_filt, c=col_10_filt, label="10% (w/ postprocess)")
    plt.plot(range(len(x_05)), y_05, c=col_05, label="05%")

    plt.legend()
    plt.xlabel('Number of augmentations')
    plt.xticks(range(len(x_10)), [f'x{i}' for i in range(len(x_10))])
    plt.ylabel(ylabel)


    extra = ['stardist_128_128_FT.2', 'stardist_128_128_FT.2_v1']
    y_extra = [metric[i] for i in extra if i in metric]
    col_extra = "C6"

    plt.scatter([1] * len(y_extra), y_extra, c=col_extra)

    os.makedirs('tmp/metrics', exist_ok=True)

    if out_name is not None:
        plt.savefig(f"tmp/metrics/{out_name}", dpi=200)

    plt.close()

def plot_metric_128(metric, out_name=None, ylabel=None):
    ## number of training images 
    exp_05 = ['stardist_05gt_inst', 'stardist_05gt_05syn_inst', 'stardist_05gt_05syn.x2_inst', 'stardist_05gt_05syn.x3_inst', 'stardist_05gt_05syn.x4_inst', 'stardist_05gt_05syn.x5_inst']
    x_05 = range(len(exp_05))
    x_05 = [i*90 for i in x_05]
    y_05 = [metric[i] for i in exp_05]

    exp_128_128_05 = ['stardist_128_128_05gt_inst', 'stardist_128_128_05gt_05syn_inst', 'stardist_128_128_05gt_05syn.x2_inst', 'stardist_128_128_05gt_05syn.x3_inst', 'stardist_128_128_05gt_05syn.x4_inst', 'stardist_128_128_05gt_05syn.x5_inst']
    y_128_128_05 = [metric[i] for i in exp_128_128_05]

    # This model was pretrained on syn data and fine tuned on GT data
    exp_128_128_05_syn2gt = get_exp_vals(metric, "stardist_128_128_05gt", word="v3", ignore="filt")
    x_128_128_05_syn2gt, y_128_128_05_syn2gt = exp_128_128_05_syn2gt

    exp_128_128_05_tstasval = get_exp_vals(metric, "stardist_128_128_05gt", word="v4", ignore="filt")
    x_128_128_05_tstasval, y_128_128_05_tstasval = exp_128_128_05_tstasval

    # This model was pretrained on syn data and fine tuned on GT data
    exp_128_128_02_syn2gt = get_exp_vals(metric, "stardist_128_128_02gt", word="v3", ignore="filt")
    x_128_128_02_syn2gt, y_128_128_02_syn2gt = exp_128_128_02_syn2gt

    # y_128_128_05 += y_extra

    print(metric)

    # x_50 = [1432, 1432*2]

    # colors
    fig, ax = plt.subplots()

    plt.plot(range(len(x_05)), y_05, label="05% (256x256)")
    plt.plot(range(len(y_128_128_05)), y_128_128_05, label="05% (128x128)")

    # This model was pretrained on syn data and fine tuned on GT data
    plt.plot(x_128_128_05_syn2gt, y_128_128_05_syn2gt, label="05% (128x128) FineTuned")

    # This model was pretrained on syn data and fine tuned on GT data
    # plt.plot(x_128_128_05_tstasval, y_128_128_05_tstasval, label="05% (128x128) TestAsVsl")

    # This model was pretrained on syn data and fine tuned on GT data
    plt.plot(x_128_128_02_syn2gt, y_128_128_02_syn2gt, label="02% (128x128) FineTuned")

    plt.legend()
    plt.xlabel('Number of augmentations')
    plt.xticks(range(len(x_05)), [f'x{i}' for i in range(len(x_05))])
    plt.ylabel(ylabel)


    if out_name is not None:
        plt.savefig(f"tmp/metrics/{out_name}", dpi=200)

    plt.close()



if __name__ == "__main__":

    metric_v0 = get_json_data("/mnt/dataset/stardist/models_monuseg/*")
    metric_v3 = get_json_data("/mnt/dataset/stardist/models_monuseg_v1.3_Syn2GT/*")
    metric_v4 = get_json_data("/mnt/dataset/stardist/models_monuseg_v1.3_TestAsVal/*")

    def comb_dict_with_suffix(src, dst, suffix):
        src = {k+suffix: v for k, v in src.items()}
        dst.update(src)

        return dst    
    
    metric_v0 = comb_dict_with_suffix(metric_v3, metric_v0, '_v3')
    metric_v0 = comb_dict_with_suffix(metric_v4, metric_v0, '_v4')


    # plot_metrics(metric_v0, names=['f1', 'dice'])

    # f1_v1 = get_json_data("/mnt/dataset/stardist/models_monuseg_v1.1/*")
    # f1_v1 = {k+'_v1': v for k, v in f1_v1.items()}

    # f1.update(f1_v1)    


    plot_samples(metric_v0, out_name='plot_sample.png')
    plot_metrics(metric_v0, names=['f1', 'dice'])


    # plot_metric(f1, out_name='plot_f1.png', ylabel='F1 score')
    # plot_metric(dice, out_name='plot_dice.png', ylabel='Dice')
    # plot_metric(acc, out_name='plot_acc.png', ylabel='Accuracy')

    # 10% experiment
    # plt.plot([0,np.max(x_10)], [f1_atb_m, f1_atb_m], c = "#1E88E5")
    # ax.fill_between([0,np.max(x_10)], f1_atb_m+f1_atb_std, f1_atb_m-f1_atb_std, color="#1E88E5", alpha=0.1)



    #     ax.fill_between(x_10, f1s_means+f1s_std, f1s_means-f1s_std, color="#D81B60", alpha=0.1)
    #     for i, f1 in enumerate(f1s):
    #         plt.scatter([x_10[i], x_10[i], x_10[i]], f1, marker= '^',color="#D81B60")
    # f1s_nm = [f1_0, f1_1, f1_2, f1_3, f1_4, f1_5, f1_6]
    # f1s_nm_means = np.mean(f1s_nm, axis = 1)
    # f1s_nm_std = np.std(f1s_nm, axis = 1)

    # # 25% experimen

    # for i,_ in enumerate(x2[:-1]): 
    #     plt.plot(x_25, f1s_nm_means, c = "#32BD71")
    #     ax.fill_between(x_25, f1s_nm_means+f1s_nm_std, f1s_nm_means-f1s_nm_std, color="#32BD71", alpha=0.1)
    #     for i, f1 in enumerate(f1s_nm):
    #         print(f1s_nm[i])
    #         plt.scatter([x_25[i], x_25[i], x_25[i]], f1s_nm[i], marker= '^',color="#32BD71")
    # plt.locator_params(axis='y', nbins=8)
    # # 50% experiment

    # for i,_ in enumerate(x_50[:-1]): 
    #     plt.plot(x_50, f1s_nm_50_means, c = "#AB32BD")
    #     ax.fill_between(x_50, f1s_nm_50_means+f1s_nm_50_std, f1s_nm_50_means-f1s_nm_50_std, color="#AB32BD", alpha=0.1)
    #     for i, f1 in enumerate(f1s_nm_50):
    #         plt.scatter([x_50[i], x_50[i], x_50[i]], f1s_nm_50[i], marker= '^',color="#AB32BD")

    # plt.locator_params(axis='y', nbins=8)
    # plt.show()

