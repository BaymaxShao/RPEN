import time
import csv
import os
import datetime
import torch
import torch.optim
import torch.utils.data
from dataloader import *
from dataloader_opt import *
from loss import *
from utils import *

import torch.backends.cudnn as cudnn
import models
import torch.nn as nn
from configs import Trainer
import numpy as np
import matplotlib.pyplot as plt
from path import Path
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)
args = Trainer().parse()


def main():
    root = args.data
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = '/checkpoints_'+timestamp
    print('=== everything will be saved to {} ==='.format(save_path))
    cudnn.deterministic = True
    cudnn.benchmark = True

    print('>>> searching for samples <<<')
    train_set = SeqData(
        root,
        istrain=True,
        img_size=224,
        train_list='train_file.txt',
        test_list='test_file.txt',
        step=4
    )

    print('=== {} samples found in {} train objects ==='.format(len(train_set), len(train_set.objs)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print('=== data loaded ===')

    model = models.ab6().to(device)

    print('=== Model Created ===')

    loss_func = QuatLoss().to(device)

    if args.pretrained_pose:
        print("=> using pre-trained weights")
        weights = torch.load(args.pretrained_pose)
        model.load_state_dict(weights['state_dict'], strict=False)

    model = torch.nn.DataParallel(model, device_ids=[0])

    optim_params = [
        {'params': model.parameters(), 'lr': args.lr},
        {'params': loss_func.parameters(), 'lr': args.lr}
    ]

    optimizer = torch.optim.Adam(optim_params, lr=args.lr, betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    train_losses = []
    os.mkdir(str(args.save_path) + save_path)
    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, optimizer, loss_func)
        train_losses.append(train_loss.item())

        print('=== epoch {}/{} done, loss: {:.3f} ==='.format(epoch+1, args.epochs, train_loss.item()))

        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        }

        torch.save(model_state, str(args.save_path)+save_path+'/{}_{:.3f}_{}'.format('ckpt', train_loss.item(), str(epoch)+'.pth.tar'))
        scheduler.step()
        print('=== epoch {}: model has been saved ==='.format(epoch+1))

    f = open(str(args.save_path)+save_path+'/loss_{}.txt'.format(timestamp), 'w')
    for loss in train_losses:
        f.write(str(loss) + '\n')
    f.close()


def train(train_loader, model, optimizer, loss_func):
    model.train()
    losses = []
    for i, (img1, img2, ref_pose, tar_pose, rel_pose_gt) in enumerate(tqdm(train_loader)):
        img1 = img1.to(device)
        img2 = img2.to(device)

        rel_pose_gt = rel_pose_gt.to(device)
        rel_pose_pred = model(img1, img2)
        loss = loss_func(rel_pose_pred, rel_pose_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)

    return sum(losses)/len(losses)

if __name__ == '__main__':
    main()









