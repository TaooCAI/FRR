import argparse
import math
import time
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

from dataloader import listflowfile as lt
from dataloader import SceneFlowLoader_demo as DA
from models.StereoNet import StereoNet

default_param = {
    'scale num':
    4,
    'max disparity':
    192,
    'batch size':
    2,
    'load model':
    'model_save_flyingthings3d_flyingthins3dTEST_b2_diff_scale_4/checkpoint_14.pth',
    'data path':
    '/home/caitao/Dataset/sceneflow/',
}
description_experiment = {
    'train data':
    'flyingthins3d',
    'test data':
    'flyingthings3d-TEST',
    'model':
    'StereoNet using difference to generate cost volume with scale is equal to four',
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
    '--batchsize',
    type=int,
    default=default_param['batch size'],
    help='batch size')
parser.add_argument(
    '--loadmodel', default=default_param['load model'], help='load model')
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

_, _, _, test_left_img, test_right_img, test_left_disp = lt.dataloader(
    args.datapath)

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


def save_prediction_for_demo(prediction_list, save_dir, left_path, disp_path,
                             batch_idx):
    save_dir = os.path.join(save_dir, batch_idx)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(save_dir, ' is existed!')
        import sys
        sys.exit(-1)

    first = os.path.join(save_dir, '1')
    if not os.path.exists(first):
        os.mkdir(first)
    second = os.path.join(save_dir, '2')
    if not os.path.exists(second):
        os.mkdir(second)

    shutil.copyfile(left_path[0],
                    os.path.join(first, os.path.basename(left_path[0])))
    shutil.copyfile(left_path[1],
                    os.path.join(second, os.path.basename(left_path[1])))

    first_disp = torch.from_numpy(np.load(disp_path[0]))
    tmp = torch.where(first_disp.data < 0, torch.zeros_like(first_disp.data),
                      first_disp.data)
    tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)
    tmp /= 255
    first_disp = transforms.ToPILImage()(torch.unsqueeze(tmp, dim=0).cpu())
    first_disp.save(os.path.join(first, f'gt.png'))

    second_disp = torch.from_numpy(np.load(disp_path[1]))
    tmp = torch.where(second_disp.data < 0, torch.zeros_like(second_disp.data),
                      second_disp.data)
    tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)
    tmp /= 255

    second_disp = transforms.ToPILImage()(torch.unsqueeze(tmp, dim=0).cpu())
    second_disp.save(os.path.join(second, f'gt.png'))

    length = len(prediction_list)
    for k, pred in enumerate(prediction_list):

        tmp = torch.where(pred.data < 0, torch.zeros_like(pred.data),
                          pred.data)
        tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)

        tmp = tmp / 255

        prediction_img = transforms.ToPILImage()(torch.unsqueeze(
            tmp[0], dim=0).cpu())
        prediction_img.save(
            os.path.join(first, f'prediction_1_{math.pow(2, length-k-1)}.png'))

        prediction_img = transforms.ToPILImage()(torch.unsqueeze(
            tmp[1], dim=0).cpu())
        prediction_img.save(
            os.path.join(second,
                         f'prediction_1_{math.pow(2, length-k-1)}.png'))


def test(imgL, imgR, disp_true, save_dir, left_path, disp_path, batch_idx):
    with torch.no_grad():
        # model.eval()
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)
        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

        # ---------
        mask = (disp_true < args.maxdisp) & (disp_true >= 0)
        # ----

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

        save_prediction_for_demo(prediction_list, save_dir, left_path,
                                 disp_path, str(batch_idx))
        output = prediction_list[-1]

        if len(disp_true[mask]) == 0:
            return 0
        else:
            # end-point-error
            return (torch.mean(
                torch.abs(output[mask] - disp_true[mask]))).item()


def test_model(model_path, save_dir, batch_count):

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['state_dict'])
    # ------------- TEST -----------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L, left_path,
                    disp_path) in enumerate(TestImgLoader):
        if batch_idx >= batch_count:
            break
        start_time = time.time()
        test_loss = test(imgL, imgR, disp_L, save_dir, left_path, disp_path,
                         batch_idx)
        print('Iter (%d / %d) test loss = %.3f, time = %.2f' %
              (batch_idx, len(TestImgLoader), test_loss,
               time.time() - start_time))
        total_test_loss += test_loss

    print('total test loss = %.3f' % (total_test_loss / batch_idx))


if __name__ == "__main__":
    test_model(args.loadmodel,
               './demo_flyingthings3d_flyingthins3dTEST_b2_diff_scale_4', 10)
