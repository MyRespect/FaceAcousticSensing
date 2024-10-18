#! Code Structure Reference: https://github.com/HobbitLong/SupContrast
import os
import json
import itertools
import torch
import random
import time
import numpy as np

from torchvision import transforms
from resnet_big import SupCEConResNet
from losses import SupConLoss
from torch.utils.data import ConcatDataset, DataLoader
from dataset_construction import FaceEmotionDataset, XDomainDataset
from utils import AverageMeter, StratifiedSampler, WeightAverage, \
    TwoCropTransform, adjust_learning_rate, warmup_learning_rate, \
        accuracy, parse_option, set_optimizer, save_model, load_model

seed = 42
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') # to handle "too many files open" error msg

torch.autograd.set_detect_anomaly(True)

def set_loader(opt, idx=5):
    """
    idx is in case 1 to choose the index of test user
    """

    normalize = transforms.Normalize(mean=4.8874, std=17.5538) # computed from whole dataset
    train_transform = transforms.Compose([WeightAverage(), normalize])
    val_transform = transforms.Compose([normalize])

    if opt.dataset == 'facer':
        ppl_list1 = ['bochen', 'chenhui', 'chenning', 'geoff', 'guangjing',\
                      'han', 'juexing', 'maolin', 'mark', 'nick', 'pp', \
                        'shane', 'shengyi', 'shuyang', 'thinh', 'tony', \
                            'van', 'yidong', 'yu', 'ce'] # all users
        ppl_list2 = ['bochen', 'chenhui', 'geoff', 'guangjing', 'juexing', \
                     'nick', 'pp',  'tony', 'van', 'yidong']
        ppl_list3 = ['chenning'', maolin', 'mark', 'shane', 'shengyi', \
                     'shuyang', 'thinh', 'yu', 'ce', 'han']
        ppl_list4 = ['bochen', 'chenhui', 'chenning', 'geoff', 'guangjing', \
                     'juexing', 'mark', 'nick', 'pp', \
                    'shane', 'shengyi', 'thinh', 'tony', 'van', 'yidong', 'yu']
        ppl_list5 = ['maolin', 'han', 'ce', 'shuyang']

        rootpath = './dataset/'

        if opt.case == '6': # merge all, and split train-test
            face_data = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list1, \
                                           wear1 = "open", wear2 = "mask", use_synthesis = True, \
                                            local_data_name = 'dataset_all', saved = opt.data_saved, \
                                                transform = TwoCropTransform(train_transform))
            train_len = int(len(face_data)*0.8)
            val_len = len(face_data)-train_len
            train_dataset, val_dataset = torch.utils.data.random_split(face_data, [train_len, val_len], \
                                                                        generator=torch.Generator().manual_seed(seed))

        elif opt.case == '1': # train with 19 ppl, test with another 1 ppl
            ppl_list_new = ppl_list1[:idx]+ppl_list1[idx+1:]
            train_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list_new, \
                                               wear1 = 'open', wear2 = 'mask', use_synthesis = True, \
                                                local_data_name = 'dataset_19ppl', saved = opt.data_saved, \
                                                    transform=TwoCropTransform(train_transform))
            val_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = [ppl_list1[idx]], \
                                              wear1 = 'open', wear2 = 'mask', use_synthesis=False, \
                                                local_data_name = 'dataset_1ppl',  saved = opt.data_saved, \
                                                transform=val_transform
                                                    )

        elif opt.case == '2': # train with men's, test with women's
            train_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list4, \
                                               wear1 = 'open', wear2 = 'mask', use_synthesis=True, \
                                                local_data_name = 'dataset_men', saved = opt.data_saved, \
                                                    transform=TwoCropTransform(train_transform))
            val_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list5, \
                                              wear1 = 'open', wear2 = 'mask', use_synthesis=False, \
                                                local_data_name = 'dataset_women', saved = opt.data_saved, \
                                                transform=val_transform
                                                    )

        elif opt.case == '3': # train with 10 ppl, test with another 10 ppl
            train_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list3, \
                                              wear1 = 'open', wear2 = None, use_synthesis=True, \
                                                local_data_name = 'dataset_ppl_list3', saved = opt.data_saved, \
                                                    transform=TwoCropTransform(train_transform))
            val_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list2, \
                                              wear1 = 'open', wear2 = None, use_synthesis=False, \
                                                local_data_name = 'dataset_ppl_list2', saved = opt.data_saved, \
                                                transform=val_transform
                                                    )

        elif opt.case == '4': # train without mask, test with mask
            train_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list1, \
                                               wear1 = 'open', wear2 = None, use_synthesis=True, \
                                                local_data_name = 'dataset_open', saved = opt.data_saved, \
                                                    transform=TwoCropTransform(train_transform))
            val_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list1, \
                                              wear1 = None, wear2 = 'mask', use_synthesis=False, \
                                                local_data_name = 'dataset_mask', saved = opt.data_saved, \
                                                transform=val_transform
                                                    )

        elif opt.case == '5': # train with mask, test without mask
            train_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list1, \
                                              wear1 = None, wear2 = 'mask', use_synthesis=False, \
                                                local_data_name = 'dataset_mask', saved = opt.data_saved, \
                                                    transform=TwoCropTransform(train_transform))
            val_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list1, \
                                              wear1 = 'open', wear2 = None, use_synthesis=False, \
                                                local_data_name = 'dataset_open', saved=opt.data_saved, \
                                                transform=val_transform
                                                    )

        elif opt.case == '0': # for test purpose
            train_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list5, \
                                               wear1 = 'open', wear2 = None, use_synthesis=True, \
                                                local_data_name = 'dataset_men', saved = False, \
                                                    transform=TwoCropTransform(train_transform))
            val_dataset = FaceEmotionDataset(rootpath = rootpath, ppl_list = ppl_list5, \
                                              wear1 = 'open', wear2 = None, use_synthesis=False, \
                                                local_data_name = 'dataset_women', saved = False, \
                                                transform=val_transform
                                                    )          
    else:
        raise ValueError(opt.dataset)

    train_class_vector=[]
    for _, label in train_dataset:
        train_class_vector.append(label)
    train_class_vector = torch.from_numpy(np.array(train_class_vector))
    train_sampler = StratifiedSampler(class_vector=train_class_vector, batch_size=opt.batch_size)

    print("Training Dataset Size: ", len(train_dataset), len(train_class_vector))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset

def set_model(opt):
    model = SupCEConResNet(name=opt.model, num_classes=opt.n_cls, feat_dim=512)
    criterion1 = torch.nn.CrossEntropyLoss() 
    criterion2 = SupConLoss(temperature=0.01, device=device)

    model = model.to(device)
    criterion1 = criterion1.to(device)
    criterion2 = criterion2.to(device)

    return model, criterion1, criterion2

def train(train_loader, model, criterion, optimizer, opt, epoch, avg_cost, lambda_weight):
    """one epoch training"""
    model.train()
    criterion1, criterion2 = criterion

    save_embedding = []
    save_label = []

    if epoch <= 5:
        lambda_weight[:, epoch] = 1.0
    else:
        w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
        w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]
        lambda_weight[0, epoch] = np.exp(w_1/10)/(np.exp(w_1/10) + np.exp(w_2/10)) # handling overflow encountered in exp

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (samples, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if isinstance(samples, list): # handling case 6 with all transforms.
            samples = torch.cat((samples[0], samples[1]), dim=0) # [bsz*2, 1, 80, 21]
        else:
            samples = torch.cat((samples, samples), dim=0)
        samples = samples.to(device, dtype=torch.float)
        labels = labels.to(device)
        bsz = labels.shape[0]

        if torch.isnan(samples).any():
            print("line 186 NaN...")

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output, feat_rep = model(samples)

        class_logits, _ = torch.split(output, [bsz, bsz], dim=0)
        loss1 = criterion1(class_logits, labels) # for cross_entropy

        f1, f2 = torch.split(feat_rep, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [bsz, 2, len(embed)]
        loss2 = criterion2(features, labels) # for contrastive loss

        # save latent representation
        save_embedding.append(f1)
        save_label.append(labels)

        avg_cost[epoch, 0] = loss1.item()
        avg_cost[epoch, 1] = loss2.item()

        loss = lambda_weight[0, epoch]*loss1 + (1-lambda_weight[0, epoch])*loss2

        # update metric
        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(class_logits, labels, topk=(1, 1))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2) # be careful about clip, making it hard to train
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1))
        #     sys.stdout.flush()

    # torch.save([save_embedding, save_label], "./results/data/"+opt.case+"_train_embed_label.pt")
    return losses.avg, top1.avg

def load_samples(dataset):
    """
    Building a dictionary containing multiple samples per category
    """
    # TODO: replacing multiple if-else statements
    center_dict = {}
    cnt = [0, 0, 0, 0, 0, 0] # mark to stop data collection
    while sum(cnt) < 6:
        idx = random.randint(0, len(dataset)-1)
        sample, label = dataset[idx]
        if isinstance(sample, list): # handle TwoCropTransforms.
            sample = sample[0]
        if label == 0:
            if '0' in center_dict.keys():
                # collect 1000 samples for each key to save some time
                if len(center_dict['0'])>1000:
                    cnt[0]=1
                    continue
                center_dict['0'].append(sample)
            else:
                center_dict['0'] = [sample]
        elif label == 1:
            if '1' in center_dict.keys():
                if len(center_dict['1'])>1000:
                    cnt[1]=1
                    continue
                center_dict['1'].append(sample)
            else:
                center_dict['1'] = [sample]        
        elif label == 2:
            if '2' in center_dict.keys():
                if len(center_dict['2'])>1000:
                    cnt[2]=1
                    continue
                center_dict['2'].append(sample)
            else:
                center_dict['2'] = [sample] 
        elif label == 3:
            if '3' in center_dict.keys():
                if len(center_dict['3'])>1000:
                    cnt[3]=1
                    continue
                center_dict['3'].append(sample)
            else:
                center_dict['3'] = [sample]
        elif label == 4:
            if '4' in center_dict.keys():
                if len(center_dict['4'])>1000:
                    cnt[4]=1
                    continue
                center_dict['4'].append(sample)
            else:
                center_dict['4'] = [sample]
        elif label == 5:
            if '5' in center_dict.keys():
                if len(center_dict['5'])>1000:
                    cnt[5]=1
                    continue
                center_dict['5'].append(sample)
            else:
                center_dict['5'] = [sample]
        if sum(cnt) == 6:
            break
    return center_dict

def data_statistic(embed_all):
    """
    MAD: median absolute deviation test
    """
    MAD_list = []
    centroid_list = []
    med_sample2cent_i_list = []
    for embed_i in embed_all:
        embed_i = torch.stack(embed_i)
        centroid_i = torch.mean(embed_i, 0)
        sample2cent_i = []
        for kk in range(len(embed_i)):
            sample2cent_i.append(torch.linalg.norm(embed_i[kk] - centroid_i))

        sample2cent_i = torch.stack(sample2cent_i)
        med_sample2cent_i = torch.median(sample2cent_i)
        MAD_i = torch.median(torch.abs(sample2cent_i-med_sample2cent_i))
        MAD_list.append(MAD_i)
        centroid_list.append(centroid_i)
        med_sample2cent_i_list.append(med_sample2cent_i)
    return MAD_list, centroid_list, med_sample2cent_i_list

def get_feat_dict(sample_dict, model):
    """
    Transforming samples in sample_dict to latent representations
    """
    model.eval()
    feature_dict = {}
    feature_dict_mean = {}

    with torch.no_grad():
        for key in sorted(sample_dict.keys()): # '0', '1',...
            feature_dict[key]=[]
            feature_dict_mean[key]=[]
            for data in sample_dict[key]:
                data = data.to(device)
                data = data[None, :, :]
                _, feat_rep = model(data)
                feature_dict[key].append(feat_rep)
            feature_dict_mean[key] = torch.stack(feature_dict[key]).mean(dim=0)
    return feature_dict, feature_dict_mean

def Kmin(feature_dict, feature_dict_mean, sample, model, use_mad=False):
    """
    Getting the label of "sample" using distance-based method, filtering inconsistent test samples
    """
    model.eval()
    with torch.no_grad():
        sample = sample.to(device)
        sample = sample[None, :, :]
        _, feat_rep_base = model(sample)

        if use_mad == True:
            embed_all = list(feature_dict.values())
            MAD_list, centroid_list, med_sample2cent_i_list = data_statistic(embed_all)
            dist_list = []
            for i, centroid_i in enumerate(centroid_list):
                tmp_dist = torch.linalg.norm(feat_rep_base-centroid_i)
                tmp_dist = torch.abs(tmp_dist - med_sample2cent_i_list[i])/MAD_list[i]
                dist_list.append(tmp_dist.cpu())
            if np.min(dist_list)> 3: # threshold for MAD to filter out inconsistent test samples
                return -1
        min_distance = float('inf')
        label_id = "0"
        for key in feature_dict_mean.keys():
            tmp_dist = torch.cdist(feat_rep_base, feature_dict_mean[key])
            if tmp_dist < min_distance:
                min_distance = tmp_dist
                label_id = key
    return int(label_id)

def Kmin_adaption(sample_dict, val_loader, model, use_mad=False):
    """
    Getting pseudo labels using Kmin function
    """
    feature_dict, feature_dict_mean = get_feat_dict(sample_dict, model)

    sample_list = []
    label_list = []
    pseudo_label_list = []

    for idx, (samples, labels) in enumerate(val_loader):
        if isinstance(samples, list): # handle case 6 with all transforms
            firt_samples = samples[0]
        else:
            firt_samples = samples
        pseudo_labels = []

        tmp_sample_list = []
        tmp_label_list = []
        for idx in range(len(firt_samples)):
            label_id = Kmin(feature_dict, feature_dict_mean, firt_samples[idx], model, use_mad)
            if label_id == -1:
                continue
            pseudo_labels.append(label_id)
            tmp_sample_list.append(firt_samples[idx])
            tmp_label_list.append(labels[idx])
        sample_list.extend(tmp_sample_list)
        label_list.extend(tmp_label_list)
        pseudo_label_list.extend(pseudo_labels)

    adaptation_acc = sum(1 for x, y in zip(label_list, pseudo_label_list) if x==y)/float(len(label_list))
    print("Cluster Domain Adaptation Accuracy: ", adaptation_acc*100)

    # acc_index = [i for i, (x, y) in enumerate(zip(label_list, pseudo_label_list)) if x == y]
    # sample_list = [sample_list[i] for i in acc_index]
    # pseudo_label_list = [pseudo_label_list[i] for i in acc_index]

    return sample_list, pseudo_label_list

def evaluation(val_loader, model, criterion, opt):
    model.eval()  # this mode affects the model performance

    criterion1, _ = criterion

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    y_true = []
    y_pred = []
    y_prob = []

    save_embedding = []
    save_label = []

    with torch.no_grad():
        end = time.time()
        for idx, (samples, labels) in enumerate(val_loader):
            if isinstance(samples, list): # handling case 6 with all transforms.
                samples = torch.cat((samples[0], samples[1]), dim=0) # [bsz*2, 1, 80, 21]
            else:
                samples = torch.cat((samples, samples), dim=0)
            samples = samples.to(device, dtype=torch.float)
            labels = labels.to(device)
            bsz = labels.shape[0]

            # compute loss
            output, feat_rep = model(samples)
            class_logits, _ = torch.split(output, [bsz, bsz], dim=0)
            loss = criterion1(class_logits, labels) # only care about the classification loss

            # save latent representation
            f1, _ = torch.split(feat_rep, [bsz, bsz], dim=0)
            save_embedding.append(f1)
            save_label.append(labels)
        
            _, pred = class_logits.topk(1, 1, True, True)
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
            prob_tmp = torch.nn.Sigmoid()(class_logits).cpu()
            prob_tmp = prob_tmp.tolist()
            y_prob.extend(prob_tmp)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, _ = accuracy(class_logits, labels, topk=(1, 1))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if top1.avg > 85: # only save results when accuraacy is greater than 85%
            y_pred = list(itertools.chain(*y_pred))
            final_result={'y_true':y_true, 'y_pred': y_pred, 'y_prob': y_prob}
            with open('./results/case'+str(opt.case)+'_final_result.json', 'w') as fp:
                json.dump(final_result, fp)
        # torch.save([save_embedding, save_label], "./results/data/"+opt.case+"_test_embed_label.pt")
    return losses.avg, top1.avg


def main(opt):
    best_acc = 0
    model_path = os.path.join(opt.save_folder, "case_"+opt.case+"_model.pt")
    model_ckpt = os.path.join(opt.save_folder, "ckpt_epoch_300.pth")

    # build data loader, idx is for case 1 testing
    train_loader, val_loader, train_dataset = set_loader(opt, idx = opt.ppl_idx)
    print("dataset loader is ready.")

    # build model and criterion
    model, criterion1, criterion2 = set_model(opt)
    criterion = (criterion1, criterion2)
    print("model is ready.")

    # build optimizer
    optimizer = set_optimizer(opt, model)

    avg_cost = np.zeros([opt.epochs+1, 2], dtype=np.float32)
    lambda_weight = np.ones([1, opt.epochs+2])

    if os.path.exists(model_ckpt):
        model = load_model(model, model_ckpt, device)
        model = model.to(device)
        print("loaded "+model_ckpt)

    if opt.train_mode == True:
        for i in range(opt.round):
            print("Entering the {}-th round...".format(i))

            if i > -1: # for debug control
                # training routine
                for epoch in range(opt.epochs+1):
                    cur_epoch = epoch+opt.epochs*i
                    adjust_learning_rate(opt, optimizer, cur_epoch)

                    # train for one epoch
                    # time1 = time.time()
                    train_loss, train_acc = train(train_loader, model, criterion, optimizer, opt, epoch, avg_cost, lambda_weight)
                    # time2 = time.time()
                    # print('epoch {}: total time {:.2f}'.format(epoch, time2 - time1))

                    # evaluation
                    val_loss, val_acc = evaluation(val_loader, model, criterion, opt)

                    print("Epoch %d: train_acc: %.2f, val_acc: %.2f, train_loss: %f, val_loss: %f" \
                    %(epoch, train_acc, val_acc, train_loss, val_loss))
                
                    if val_acc > best_acc:
                        best_acc = val_acc

                    if cur_epoch > 10 and cur_epoch % opt.save_freq == 0:
                        save_file = os.path.join(
                            opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=cur_epoch))
                        save_model(model, optimizer, opt, cur_epoch, save_file)
                torch.save(model, model_path)

            print("Entering domain adaption stage: ")
            if os.path.exists(model_path): # loading previous trained model is helpful
                model = torch.load(model_path, map_location=device)
                print("loaded "+model_path)

            # build class center samples
            sample_dict = load_samples(train_dataset)
            print("center dictionary is ready.")

            sample_list, pseduo_label_list = Kmin_adaption(sample_dict, val_loader, model, use_mad=opt.use_mad)

            #==== here is to simulate the training with pseudo labels generated from clustering

            # do the transformation to avoid using the original samples
            xdomain_datset = XDomainDataset(sample_list, pseduo_label_list, transform=TwoCropTransform(WeightAverage()))

            train_dataset = ConcatDataset([train_dataset, xdomain_datset])
            
            train_class_vector=[]
            for _, label in train_dataset:
                train_class_vector.append(label)
            train_class_vector = torch.from_numpy(np.array(train_class_vector))
            train_sampler = StratifiedSampler(class_vector=train_class_vector, batch_size=opt.batch_size)

            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, \
                                                    num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
            # build optimizer -- in case changes
            opt.optimizer_name = 'sgd'
            opt.learning_rate = 1e-3 # high learning_rate got Nan
            opt.epochs = 5
            optimizer = set_optimizer(opt, model)
    else:
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=device)
        val_loss, val_acc = evaluation(val_loader, model, criterion, opt)
        print("Saving results into json file.")
        

if __name__ == '__main__':
    opt = parse_option()
    main(opt)