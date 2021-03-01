#!usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

import torchvision.transforms.functional as FF
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from collections import OrderedDict

import torch.backends.cudnn as cudnn
import re
cudnn.benchmark = True

from net import sphere20

"""
Desc: test process using LFW
Date: 2019/05/13
Author: Jesse
Contact: majie1@sensetime.com

"""


# using 10 folds to separate train set and test set
def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n // n_folds:(i + 1) * n // n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    print(np.asarray(folds).shape)
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    accuracy = 1.0 * np.count_nonzero(y_predict == y_true) / len(y_true)

    return accuracy


# find the best threshold and accuracy
def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold, best_acc


def eval(model_path=None):
    predicts = []
    # if sphere20.is_cuda:
    model = sphere20()

    check_points = torch.load(model_path)
    # print('check_points.items = ', check_points.items())

    # RuntimeError: Error(s) in loading state_dict for sphere20:
    # Missing key(s) in state_dict: "conv1_1.weight", "conv1_1.bias", "conv1_2.weight", "conv1_3.weight", "conv2_1.weight",
    # "conv2_1.bias", "conv2_2.weight", "conv2_3.weight", "conv3_1.weight", "conv3_1.bias", "conv3_2.weight", "conv3_3.weight",
    # "conv4_1.weight", "conv4_1.bias", "conv4_2.weight", "conv4_3.weight", "fc4.weight", "fc4.bias".
    # Unexpected key(s) in state_dict: "module.conv1_1.weight", "module.conv1_1.bias", "module.conv1_2.weight",
    # "module.conv1_3.weight", "module.conv2_1.weight", "module.conv2_1.bias",
    # "module.conv2_2.weight", "module.conv2_3.weight", "module.conv3_1.weight",
    # "module.conv3_1.bias", "module.conv3_2.weight", "module.conv3_3.weight", "module.conv4_1.weight", "module.conv4_1.bi
    # as", "module.conv4_2.weight", "module.conv4_3.weight", "module.fc4.weight", "module.fc4.bias".

    # You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it without DataParallel.
    # You can either add a nn.DataParallel temporarily in your network for loading purposes,
    # or you can load the weights file, create a new ordered dict without the module prefix, and load it back.

    new_state_dict = OrderedDict()
    for k, v in check_points.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()

    root = '/mnt/lustre/share/platform/dataset/lfw-112x112-mxnet/'
    with open('test_pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0, 1.0]
    ])

    for i in range(len(pairs_lines)):
        # p = pairs_lines[i].replace('\n', '').split('\t')
        p = ' '.join(pairs_lines[i].split()).split(' ')

        if len(p) == 3:
            sameflag = 1  # two pics of same person
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if len(p) == 4:   # jesse 1 emma 2 not the same person
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        else:
            print('p = ', p)
        img1 = Image.open(root + name1).convert('RGB')
        img2 = Image.open(root + name2).convert('RGB')

        img1, img1_, img2, img2_ = transform(img1), transform(FF.hflip(img1)), transform(img2), transform(FF.hflip(img2))

        # unsqueeze(0) 将size[n] -> [1, n]; volatile=True, 取消自动求导
        with torch.no_grad():
            img1, img1_ = Variable(img1.unsqueeze(0).cuda(), volatile=True), Variable(img1_.unsqueeze(0))
            img2, img2_ = Variable(img2.unsqueeze(0).cuda(), volatile=True), Variable(img2_.unsqueeze(0))

        f1 = torch.cat((model(img1), model(img1_), 1)).data[0]
        f2 = torch.cat((model(img2), model(img2_), 1)).data[0]

        # cos distance
        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)

        # predict
        predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))
    print('predicts = ', predicts)
    accuracy = []
    thd = []
    folds = KFold()
    thresholds = np.arrange(-1.0, 1.0, 0.005)

    predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))

    for idx, (train, test) in enumerate(folds):
        best_thresh, _ = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)

    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy), predicts



if __name__ == '__main__':
    # KFold()
    eval('checkpoints/CosFace_30_checkpoint.pth')
    pass














