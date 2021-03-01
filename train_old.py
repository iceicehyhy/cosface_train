#!usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import argparse
import time
import  pdb
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# torch 包
import torch
import torch.utils.data
import torch.optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn
cudnn.benchmarks = True

from dataset import ImageList
from net import sphere20
from layer import MarginCosineProduct
import LResnet50

DATASET_ROOT = '/export/home/iceicehyhy/CASIA-maxpy-clean/'

"""
Desc:
    complete the training process
Dataset:
    CASIA-Webface 112 * 112
Date:
    2019/05/09
Author:
    Jesse
Contact:
    majie1@sensetime.com
"""

# training parameter
parser = argparse.ArgumentParser('Pure implementation of CosFace by Pytorch')
parser.add_argument('--root-path', type=str, default= DATASET_ROOT , help='the training set root path')
parser.add_argument('--image-list', type=str, default='/export/home/iceicehyhy/CosFace_Adaptive_S_1/Casia_Webface_112x112_train_list.txt', help='the file and its path of image list')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--num-class', type=int, default=10575, help='number of people(class)')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--log-interval', type=float, default=20)
parser.add_argument('--step-size', type=int, default=[16000, 24000])
parser.add_argument('--save-path', type=str, default='checkpoints')
parser.add_argument('--no-cuda', type=bool, default=False)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = True
args.num_class = len(os.listdir(args.root_path))

def train(train_loader, model, MCP, criterion, optimizer, epoch):
    # train
    model.train()  # update all params
    print_with_time('Epoch {} start training'.format(epoch))
    # get current time
    time_curr = time.time()
    # show loss
    loss_display = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        # iteration number
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        # adjust lr
        adjust_learning_rate(optimizer, iteration, args.step_size)
        # use cuda or not
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # compute output
        output = model(data)
        # pdb.set_trace()
        output = MCP(output, target).requires_grad_()   # allomem for gradients

        # print('------------------')
        # print('output of MCP = ', output.shape)
        # print('target = ', target.shape)
        # print('---------------------------')
        loss = criterion(output, target)

        loss_display += loss.detach().item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        # back propagation
        # try:
        #     print('loss = ', loss)
        # except RuntimeError:
        #     pdb.set_trace()
        #     print('loss = ', loss)
        loss.backward()
        # update parameter
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(MCP.m, MCP.s)
            # INFO = ' lambda: {:.4f}'.format(MCP.lamb)
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display, time_used, args.log_interval) + INFO
            )
            time_curr = time.time()
            loss_display = 0.0


def print_with_time(string):
    print(time.strftime('%Y-%M-%d %H:%M:%S', time.localtime()) + string)


def main():
    if torch.cuda.device_count() > 1:
        print('available gpus is ', torch.cuda.device_count(), torch.cuda.get_device_name())
    else:
        print("only one GPU found !!!")
    #model = sphere20()
    model = LResnet50.LResNet50E_IR(is_gray = False)
    print (model)
    model = torch.nn.DataParallel(model, device_ids = [1],output_device=0).cuda()   # enable mutiple-gpu training

    # print(model)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # model.save(args.save_path + '/CosFace_0_checkpoint.pth')

    print('save checkpoint finished!')

    # upload training dataset
    train_loader = torch.utils.data.DataLoader(
        ImageList(
            root=args.root_path,
            fileList=args.image_list,

            # processing images
            transform=transforms.Compose([
                # hflip PIL 图像 at 0.5 probability
                transforms.RandomHorizontalFlip(),
                # transform a PIL image（H*W*C）in [0, 255] to torch.Tensor(H*W*C) in [0.0, 0.1]
                transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
                # use mean and standard deviation to normalize data
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 0.1] -> [-1.0, 1.0]
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True
    )

    # print the length of train dataset
    print('length of train dataset: {}'.format(str(len(train_loader.dataset))))
    # print the class number of train dataset
    print('Number of Classes: {}'.format(str(args.num_class)))

    # --------------------------------loss function and optimizer-------------------------------
    # core implementation of Cos face, using cuda
    scale = math.sqrt(2) * math.log(args.num_class - 1)
    MCP = MarginCosineProduct(512, args.num_class, s=scale).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([
        {'params': model.parameters()}, {'params': MCP.parameters()}],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, MCP, criterion, optimizer, epoch)
        torch.save(model.state_dict(), os.path.join(args.save_path, 'CosFace_' + str(epoch) + '_checkpoint.pth'))
        torch.save(MCP.state_dict(), os.path.join(args.save_path, 'MCP_' + str(epoch) + '_checkpoint.pth'))

    print('Finished Training')


# function of adjusting lr
def adjust_learning_rate(optimizer, iteration, step_size):
    """
    set lr to the initial LR decayed by 10% for each step size
    :param optimizer:
    :param iteration:
    :param step_size:
    :return:
    """
    if iteration in step_size:
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))

        #  managing parameters using param_groups in optimizer， param_group
        #  which including parameter group, corresponding lr, momentum etc.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    main()





















