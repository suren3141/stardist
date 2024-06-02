import numpy as np
from matplotlib import pyplot as plt
import json, os, glob

def get_json_data(path, file_name='stats.json'):

    json_path = os.path.join(path, file_name)

    files = glob.glob(json_path)

    dic = {}

    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            dice = data['0.5']['f1']
            exp = os.path.split(os.path.split(f)[0])[-1]
            dic[exp] = dice

    return dic


if __name__ == "__main__":

    dice = get_json_data("/mnt/dataset/stardist/models_monuseg/*")

    ## number of training images 
    exp_10 = ['stardist_10gt_inst', 'stardist_10gt_10syn_inst', 'stardist_10gt_10syn.x2_inst', 'stardist_10gt_10syn.x3_inst', 'stardist_10gt_10syn.x4_inst', 'stardist_10gt_10syn.x5_inst']
    x_10 = range(len(exp_10))
    x_10 = [i*181 for i in x_10]
    y_10 = [dice[i] for i in exp_10]
    col_10 = "#D81B60"

    exp_25 = ['stardist_25gt_inst', 'stardist_25gt_25syn_inst', 'stardist_25gt_25syn.x2_inst', 'stardist_25gt_25syn.x3_inst', 'stardist_25gt_25syn.x4_inst', 'stardist_25gt_25syn.x5_inst']
    x_25 = range(len(exp_25))
    x_25 = [i*453 for i in x_25]
    y_25 = [dice[i] for i in exp_25]
    col_25 = "#32BD71"

    print(dice)

    # x_50 = [1432, 1432*2]

    # colors
    fig, ax = plt.subplots()

    plt.plot(range(len(x_10)), y_10, c=col_10, label="10%")
    plt.plot(range(len(x_25)), y_25, c=col_25, label="25%")
    plt.legend()
    plt.xlabel('Number of augmentations')
    plt.xticks(range(len(x_10)), [f'x{i}' for i in range(len(x_10))])
    plt.ylabel('F1 score')

    os.makedirs('tmp', exist_ok=True)
    plt.savefig("tmp/plot.png", dpi=200)

    plt.close()





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

