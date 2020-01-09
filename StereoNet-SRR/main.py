import argparse
import math
import sys
import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models.StereoNet import StereoNet

SCALE_NUM = 3

default_param = {
    'scale num': SCALE_NUM,
    'max disparity': 192,
    'epochs': 100,
    'batch size': 2,
    'load model': 'model_save_flyingthings3d_flyingthins3dTEST_b2_diff_scale_3/checkpoint_14.pth',
    'save model':
    f"./model_save_SceneFlow_flyingthins3dTEST_b2_diff_scale_{SCALE_NUM}_load_optimizer_state",
    'data path': '/home/caitao/Dataset/sceneflow/',
}
description_experiment = {
    'experiment details log file': 'experiment_details.txt',
    'train data':
    'SceneFlow',
    'test data':
    'flyingthings3d-TEST',
    'model':
    f"StereoNet using difference to generate cost volume with scale is equal to {SCALE_NUM}",
}
__normalize = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}

parser = argparse.ArgumentParser(description=description_experiment['model'])

parser.add_argument(
    '--maxdisp',
    type=int,
    default=default_param['max disparity'],
    help='maxium disparity')
# scale of low-solution image feature to input image
parser.add_argument(
    '--scalenum',
    type=int,
    default=default_param['scale num'],
    help='scale num')
parser.add_argument(
    '--datapath', default=default_param['data path'], help='datapath')
parser.add_argument(
    '--epochs',
    type=int,
    default=default_param['epochs'],
    help='number of epochs to train')
parser.add_argument(
    '--batchsize',
    type=int,
    default=default_param['batch size'],
    help='batch size')
parser.add_argument(
    '--loadmodel', default=default_param['load model'], help='load model')
parser.add_argument(
    '--savemodel', default=default_param['save model'], help='save model')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='enables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, \
    test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath,
        only_train_on_flyingthings3d=False)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(
        all_left_img,
        all_right_img,
        all_left_disp,
        True,
        normalize=__normalize),
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=min(4, args.batchsize),
    drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(
        test_left_img,
        test_right_img,
        test_left_disp,
        False,
        normalize=__normalize),
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=min(4, args.batchsize),
    drop_last=False)

model = StereoNet(args.scalenum, args.scalenum, args.maxdisp)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=0.001)

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    print('Load model %s success...' % (args.loadmodel),
          'Load optimizer success...')

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true < args.maxdisp) & (disp_true >= 0)
    mask.detach_()
    # ----
    optimizer.zero_grad()

    prediction_list = model(imgL, imgR)
    length = len(prediction_list)
    for i in range(length):
        if prediction_list[i].size()[-2:] != mask.size()[-2:]:
            assert i != length - 1
            prediction_list[i] *= (
                mask.size()[-1] / prediction_list[i].size()[-1])
            prediction_list[i] = torch.squeeze(
                F.interpolate(
                    torch.unsqueeze(prediction_list[i], dim=1),
                    size=mask.size()[-2:],
                    mode='bilinear',
                    align_corners=False),
                dim=1)

    count = len(torch.nonzero(mask))
    loss = torch.sum(
        torch.sqrt(
            torch.pow(disp_true[mask] - prediction_list[-1][mask], 2) + 4) /
        2 - 1) / count
    for i in range(length - 1):
        loss += torch.sum(
            torch.sqrt(
                torch.pow(disp_true[mask] - prediction_list[i][mask], 2) + 4) /
            2 - 1) / count

    loss = loss / length
    if loss.item() > 50:
        vis_loss = 50
    else:
        vis_loss = loss.item()

    output1 = prediction_list[-3]
    output2 = prediction_list[-2]
    output3 = prediction_list[-1]

    loss.backward()
    optimizer.step()

    if math.isnan(loss.item()):
        torch.save({
            'output1': output1,
            'output2': output2,
            'output3': output3
        }, 'exception.pth')
        sys.exit(-1)

    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    mask = (disp_true < args.maxdisp) & (disp_true >= 0)
    # ----
    prediction_list = model(imgL, imgR)
    output = prediction_list[-1]
    if len(disp_true[mask]) == 0:
        return 0
    else:
        # end-point-error
        return (torch.mean(
            torch.abs(output[mask] - disp_true[mask]))).item()


def save_default_and_experiment_description():
    save_param = {
        'scale num': args.scalenum,
        'max disparity': args.maxdisp,
        'epochs': args.epochs,
        'batch size': args.batchsize,
        'load model': args.loadmodel,
        'save model': args.savemodel,
        'data path': args.datapath,
        'normalize': __normalize
    }
    with open(os.path.join(
            args.savemodel,
            description_experiment['experiment details log file']),
            'w') as fout:
        fout.write(str(save_param))
        fout.write('\n')
        fout.write(str(description_experiment))


def main():
    save_default_and_experiment_description()
    start_full_time = time.time()
    train_loss_log = []
    eval_loss_log = []
    for epoch in range(args.epochs):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0
        # training
        for batch_idx, (imgL_crop, imgR_crop,
                        disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print(
                'Epoch (%d / %d) Iter (%d / %d) training loss = %.3f , time = %.2f'
                % (epoch, args.epochs, batch_idx, len(TrainImgLoader), loss,
                   time.time() - start_time))
            total_train_loss += loss
            with open(
                    os.path.join(args.savemodel, 'train loss every batch.txt'),
                    'a') as log_train_loss:
                log_train_loss.write(str([epoch, batch_idx, loss]))
                log_train_loss.write('\n')

        total_train_loss = total_train_loss / (batch_idx + 1)
        print(
            'Epoch %d total training loss = %.3f' % (epoch, total_train_loss))
        train_loss_log.append(total_train_loss)

        # SAVE
        savefilename = os.path.join(args.savemodel,
                                    'checkpoint_' + str(epoch) + '.pth')
        save_dir = os.path.dirname(savefilename)
        save_dir = save_dir if save_dir != '' else './'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': total_train_loss,
        }, savefilename)

        # evaluation
        with torch.no_grad():
            total_eval_loss = 0
            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                start_time = time.time()
                eval_loss = test(imgL, imgR, disp_L)
                print('Model %d Iter (%d / %d) eval loss = %.3f, time = %.2f' %
                      (epoch, batch_idx, len(TestImgLoader), eval_loss,
                       time.time() - start_time))
                total_eval_loss += eval_loss
                if (batch_idx + 1) * args.batchsize >= 400:
                    break

            total_eval_loss = total_eval_loss / (batch_idx + 1)
            print('total eval loss = %.3f' % (total_eval_loss))
            eval_loss_log.append(total_eval_loss)

        # print previous train and eval loss after finishing each epoch
        #
        # snack
        print('train loss log for previous epoch: ', train_loss_log)
        print('eval loss log for previous epoch: ', eval_loss_log)
        with open(
                os.path.join(
                    args.savemodel,
                    'train_and_eval_loss_log_every_epoch_tmp_98741.txt'),
                'a') as f:
            f.write(str([epoch, total_train_loss, total_eval_loss]))
            f.write('\n')

    # save train loss and eval loss after every training
    log_loss_data = {
        'train_loss_list': train_loss_log,
        'eval_loss_list': eval_loss_log
    }
    torch.save(
        log_loss_data,
        os.path.join(args.savemodel,
                     'map_list_train_and_eval_loss_after_every_epoch.pth'))
    # log the training time
    all_consume_time = (time.time() - start_full_time) / 3600
    print('full training time = %.2f HR' % (all_consume_time))
    with open(os.path.join(
            args.savemodel,
            description_experiment['experiment details log file']),
            'a') as f:
        f.write(f'full training time: {all_consume_time} hours')
        f.write('\n')


if __name__ == '__main__':
    if os.path.exists(args.savemodel):
        sys.exit(-1)
    os.makedirs(args.savemodel)
    main()
