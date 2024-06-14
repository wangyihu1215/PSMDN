import sys

import torch
import torch.optim
import os
import argparse
import dataloader
import model
import Myloss
import warnings
from utils import plot_curve

warnings.filterwarnings("ignore")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DCE_net = model.enhance_net_nopool().cuda()

    DCE_net.apply(weights_init)
    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False,
                                               num_workers=config.num_workers, pin_memory=True)

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()

    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    L_vs = Myloss.L_vs(4)
    L_hue = Myloss.L_hue(4)
    L_con = Myloss.L_con(4)
    L_value = Myloss.L_value(4)

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()

    l_plot = []

    for epoch in range(config.num_epochs):
        for iteration, image_haze in enumerate(train_loader):

            image_haze = image_haze.cuda()

            image_dehaze_1, image_dehaze, A = DCE_net(image_haze)

            loss_TV = 200 * L_TV(A)

            loss_spa = torch.mean(L_spa(image_dehaze, image_haze))

            loss_col = 5 * torch.mean(L_color(image_dehaze))

            loss_exp = 2 * torch.mean(L_exp(image_dehaze))

            loss_vs = torch.mean(L_vs(image_dehaze)) * 2

            loss_hue = L_hue(image_haze, image_dehaze) * 1.5

            loss_con, b, r_b, g_b, b_b = L_con(image_dehaze)

            loss_con = torch.mean(loss_con) * 0.2

            loss_value = torch.mean(L_value(image_haze, image_dehaze))*100

            # best_loss
            loss = loss_TV + loss_spa + loss_exp + loss_vs + loss_col + loss_hue

            if loss < 0:
                print("ERROR!!!!!",
                      f'loss_tot : {loss}',
                      f'loss_con : {loss_con}',
                      f'background: {b}',
                      f'r_b: {r_b}',
                      f'g_b: {g_b}',
                      f'b_b: {b_b}',
                      sep='\n')
                sys.exit()

            # print(f'loss_vs  : {loss_vs}',
            #       f'loss_exp : {loss_exp}',
            #       f'loss_col : {loss_col}',
            #       f'loss_spa : {loss_spa}',
            #       f'Loss_TV  : {loss_TV}',
            #       f'loss_hue : {loss_hue}',
            #       f'loss_con : {loss_con}',
            #       f'loss_val : {loss_value}',
            #       f'loss_tot : {loss}',
            #       sep='\n')
            # exit()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print(f"Epoch : {epoch},",  " Loss at iteration", iteration + 1, ":", loss.item())
                l_plot.append(loss.cpu().detach().numpy())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "Dehaze_newdataset" + str(epoch) + '.pth')

    plot_curve(l_plot, 'training loss test')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="F:/Datasets/Train/haze/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=7)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
