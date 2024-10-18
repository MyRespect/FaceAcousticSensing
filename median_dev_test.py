import torch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

def load_embedding(file_path):
    # loading the training sample
    train_saved_embed_label = torch.load(file_path, map_location="cpu")
    train_rep, train_y = train_saved_embed_label
    train_rep = torch.cat(train_rep, dim=0).detach()
    train_y = torch.cat(train_y, dim=0).detach()

    embed_0 = []
    embed_1 = []
    embed_2 = []
    embed_3 = []
    embed_4 = []
    embed_5 = []

    for kk in range(len(train_rep)):
        if train_y[kk] == 0:
            embed_0.append(train_rep[kk])
        elif train_y[kk] == 1:
            embed_1.append(train_rep[kk])
        elif train_y[kk] == 2:
            embed_2.append(train_rep[kk])
        elif train_y[kk] == 3:
            embed_3.append(train_rep[kk])
        elif train_y[kk] == 4:
            embed_4.append(train_rep[kk])
        elif train_y[kk] == 5:
            embed_5.append(train_rep[kk])

    embed_all = [embed_0, embed_1, embed_2, embed_3, embed_4, embed_5]

    return embed_all

def data_statistic(embed_all):
    """
    MAD: median absolute deviation test
    """
    MAD_list = []
    centroid_list = []
    med_sample2cent_i_list = []
    for embed_i in embed_all:
        embed_i = np.stack(embed_i)
        centroid_i = np.mean(embed_i, 0)
        sample2cent_i = []
        for kk in range(len(embed_i)):
            sample2cent_i.append(np.linalg.norm(embed_i[kk] - centroid_i))
        sample2cent_i = np.stack(sample2cent_i)
        print(sample2cent_i)
        med_sample2cent_i = np.median(sample2cent_i)
        MAD_i = np.median(np.abs(sample2cent_i-med_sample2cent_i))
        MAD_list.append(MAD_i)
        centroid_list.append(centroid_i)
        med_sample2cent_i_list.append(med_sample2cent_i)
    return MAD_list, centroid_list, med_sample2cent_i_list

if __name__ == "__main__":
    train_file_path = "./results/data/5_train_embed_label.pt"
    test_file_path = "./results/data/5_test_embed_label.pt"
    train_embed_all = load_embedding(train_file_path)
    test_embed_all = load_embedding(test_file_path)
    MAD_list, centroid_list, med_sample2cent_i_list = data_statistic(train_embed_all)

    y_pred = []
    y_true = []
    for idx, test_embed_i in enumerate(test_embed_all):
        for kk in range(len(test_embed_i)):
            dist_list = []
            for i, centroid_i in enumerate(centroid_list):
                tmp_dist = np.linalg.norm(test_embed_i[kk]-centroid_i)
                tmp_dist = np.abs(tmp_dist - med_sample2cent_i_list[i])/MAD_list[i]
                dist_list.append(tmp_dist)
            y_pred.append(np.argmin(dist_list))
            y_true.append(idx)
    prfs = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(prfs, acc)

    # # linear classification test
    # train_saved_embed_label = torch.load(train_file_path, map_location=device)
    # train_rep, train_y = train_saved_embed_label
    # test_saved_embed_label = torch.load(test_file_path, map_location=device)
    # test_rep, test_y = test_saved_embed_label
    # clf2 = MLPClassifier(hidden_layer_sizes=(1024, 512), random_state=1, max_iter=1000)
    # clf2.fit(train_rep, train_y)
    # y_pred = clf2.predict(test_rep)
    # print(accuracy_score(test_y, y_pred))