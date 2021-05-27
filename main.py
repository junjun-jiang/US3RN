from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from model import S3RNet
from data import get_patch_training_set, get_test_set
from torch.autograd import Variable
from psnr import MPSNR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from utils import util, build_code_arch

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Defocus Deblur: Path to option ymal file.')
train_args = parser.parse_args()

opt, resume_state = build_code_arch.build_resume_state(train_args)
opt, logger, tb_logger = build_code_arch.build_logger(opt)

for phase, dataset_opt in opt['dataset'].items():
    if phase == 'train':
        train_dataset = get_patch_training_set(dataset_opt,opt['network_G'])
        training_data_loader = DataLoader(
            train_dataset, batch_size=dataset_opt['batch_size'], shuffle=True,
            num_workers=dataset_opt['workers'], pin_memory=True)
        logger.info('Number of train images: {:,d}'.format(len(train_dataset)))
    elif phase == 'val':
        val_dataset = get_test_set(dataset_opt,opt['network_G'])
        testing_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=dataset_opt['workers'], pin_memory=True)
        logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_dataset)))
assert training_data_loader is not None


# Training settings
# parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
# parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
# parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
# parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
# parser.add_argument('--ChDim', type=int, default=31, help='output channel number')
# parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
# parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. Default=0.01')
# parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
# parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
# parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
# opt = parser.parse_args()

# print(opt)

# if opt.cuda and not torch.cuda.is_available():
#     raise Exception("No GPU found, please run without --cuda")


# def set_random_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# set_random_seed(opt.seed)

# print('===> Loading datasets')
# train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
# test_set = get_test_set(opt.upscale_factor)
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, pin_memory=True)
#
# print('===> Building model')


model_opt = opt['network_G']
input_channel = model_opt['input_channel']
output_channel = model_opt['output_channel']
upscale_factor = model_opt['upscale_factor']

model = S3RNet(input_channel, output_channel, upscale_factor).cuda()
print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
model = torch.nn.DataParallel(model).cuda()


# optimizer = optim.Adam(model.parameters(), lr=opt.lr)
# scheduler = MultiStepLR(optimizer, milestones=[10,30,60,120], gamma=0.5)
optimizer = optim.Adam(model.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']),
                 lr=opt['train']['lr'])
scheduler = MultiStepLR(optimizer=optimizer,milestones=opt['train']['lr_steps'],
                                     gamma=opt['train']['lr_gamma'])

# if opt.nEpochs != 0:
#     load_dict = torch.load(opt.save_folder+"_epoch_{}.pth".format(opt.nEpochs))
#     opt.lr = load_dict['lr']
#     epoch = load_dict['epoch']
#     model.load_state_dict(load_dict['param'])
#     optimizer.load_state_dict(load_dict['adam'])
if resume_state:
    logger.info('Resuming training from epoch: {}.'.format(
        resume_state['epoch']))
    start_epoch = resume_state['epoch']
    optimizer.load_state_dict(resume_state['optimizers'])
    scheduler.load_state_dict(resume_state['schedulers'])
    model.load_state_dict(resume_state['state_dict'])
else:
    start_epoch = 0

criterion = nn.L1Loss()


# current_time = datetime.now().strftime('%b%d_%H-%M-%S')
# CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
# tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0

def train(epoch, optimizer, scheduler):
    epoch_loss = 0
    global current_step

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # with torch.autograd.set_detect_anomaly(True):
        W, Y, Z, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
        optimizer.zero_grad()
        W = Variable(W).float()
        Y = Variable(Y).float()
        Z = Variable(Z).float()
        X = Variable(X).float()
        HX, HY, HZ, listX, listY, listZ = model(W)

        loss = criterion(HX, X) + alpha*criterion(HY, Y) + alpha*criterion(HZ, Z)
        for i in range(len(listX) - 1):
            loss = loss + opt['train']['alpha1'] * criterion(X, listX[i]) + \
                   0.5 * opt['train']['alpha2'] * criterion(Y, listY[i]) + \
                   0.5 * opt['train']['alpha3'] * criterion(Z, listZ[i])
        epoch_loss += loss.item()

        tb_logger.add_scalar('total_loss', loss.item(), current_step)
        current_step += 1

        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 100 == 0:

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)

def test():
    avg_psnr = 0
    avg_time = 0
    model.eval()
    with torch.no_grad():
        for batch in testing_data_loader:
            W, X = batch[0].cuda(), batch[1].cuda()
            W = Variable(W).float()
            X = Variable(X).float()
            start_time = time.time()

            HX, HY, HZ, listX, listY, listZ = model(W)
            end_time = time.time()

            X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()
            psnr = MPSNR(HX,X)
            im_name = batch[2][0]
            print(im_name)
            print(end_time - start_time)
            avg_time += end_time - start_time
            (path, filename) = os.path.split(im_name)
            io.savemat(opt.outputpath + filename, {'HX': HX})
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. time: {:.4f} s".format(avg_time / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):
    if epoch % opt['logger']['save_checkpoint_freq'] == 0:
        logger.info('Saving models and training states.')
        save_filename = '{}_{}.pth'.format(epoch, 'models')
        save_path = os.path.join(opt['path']['models'], save_filename)
        state_dict = model.state_dict()
        save_checkpoint = {'state_dict': state_dict,
                           'optimizers': optimizer.state_dict(),
                           'schedulers': scheduler.state_dict(),
                           'epoch': epoch}
        torch.save(save_checkpoint, save_path)
        torch.cuda.empty_cache()

# def checkpoint(epoch):
#
#     model_out_path = opt.save_folder+"_epoch_{}.pth".format(epoch)
#     if epoch % 5 == 0:
#         save_dict = dict(
#             lr = optimizer.state_dict()['param_groups'][0]['lr'],
#             param = model.state_dict(),
#             adam = optimizer.state_dict(),
#             epoch = epoch
#         )
#         torch.save(save_dict, model_out_path)
#
#         print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(start_epoch, opt['epoch']):
        avg_loss = train(epoch, optimizer, scheduler)
        checkpoint(epoch)
        avg_psnr = test()
        tb_logger.add_scalar('psnr', avg_psnr, epoch)

# if opt.mode == 1:
#     for epoch in range(opt.nEpochs + 1, 201):
#         avg_loss = train(epoch, optimizer, scheduler)
#         checkpoint(epoch)
#         avg_psnr = test()
#         tb_logger.add_scalar('psnr', avg_psnr, epoch)
#
# else:
#     test()
