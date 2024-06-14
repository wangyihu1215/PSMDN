import argparse
import os

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import triple_transforms
from nets import DGNLNet

from dataset_loader import ImageFolder
from misc import AvgMeter

cudnn.benchmark = True


def train(config):
    # datasets load
    train_set = ImageFolder(config.root, transform=transform, target_transform=transform,
                            triple_transform=triple_transform, is_train=True)
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              num_workers=config.num_workers, shuffle=True)

    # net load
    net = DGNLNet().cuda().train()

    # loss load
    criterion_img = nn.L1Loss()
    criterion_depth = nn.L1Loss()

    # optimizer load
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * config.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': config.lr, 'weight_decay': config.weight_decay}
    ])

    # checkpoint folder
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    log_path = os.path.join(config.checkpoint, config.log_name)

    # train
    curr_iter = 0

    while True:
        train_loss_record = AvgMeter()
        train_net_loss_record = AvgMeter()
        train_depth_loss_record = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * config.lr * (1 - float(curr_iter) / config.iter_num
                                                               ) ** config.lr_decay
            optimizer.param_groups[1]['lr'] = config.lr * (1 - float(curr_iter) / config.iter_num
                                                           ) ** config.lr_decay

            original, haze, depth = data

            batch_size = config.batch_size

            original = original.cuda()
            haze = haze.cuda()
            depth = depth.cuda()

            depth = torch.exp(-0.1 * depth)

            optimizer.zero_grad()

            result, depth_pred = net(haze)

            loss_net = criterion_img(result, original)
            loss_depth = criterion_depth(depth_pred, depth)

            loss = loss_net + loss_depth

            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            train_net_loss_record.update(loss_net.data, batch_size)
            train_depth_loss_record.update(loss_depth.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [lr %.13f], [loss_net %.5f], [loss_depth %.5f]' % \
                  (curr_iter, train_loss_record.avg, optimizer.param_groups[1]['lr'],
                   train_net_loss_record.avg, train_depth_loss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % config.snapshot_epoch == 0:
                torch.save(net.state_dict(), os.path.join(config.checkpoint, ('%d.pth' % (curr_iter + 1))))
                torch.save(optimizer.state_dict(),
                           os.path.join(config.checkpoint, ('%d_optim.pth' % (curr_iter + 1))))

            if curr_iter > config.iter_num:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='F:/Datasets/Train/')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--log_name', type=str, default='test.txt')
    parser.add_argument('--iter_num', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--img_size', type=list, default=[256, 512], help='resized img size [h, w]')
    parser.add_argument('--snapshot_epoch', type=int, default=1000)

    config = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # to_pil = transforms.ToPILImage()

    triple_transform = triple_transforms.Compose([
        triple_transforms.Resize((config.img_size[0], config.img_size[1])),
        # triple_transforms.RandomCrop(args['crop_size']),
        triple_transforms.RandomHorizontallyFlip()
    ])

    train(config)
