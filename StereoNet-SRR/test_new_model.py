import argparse
import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import visdom

from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models.StereoNet import StereoNet

default_param = {
    'scale num': 3,
    'max disparity': 192,
    'batch size': 2,
    'load model':
    'model_save_flyingthings3d_flyingthins3dTEST_b2_diff_scale_4',
    'data path': '/home/caitao/Dataset/sceneflow/',
}
description_experiment = {
    'train data':
    'flyingthins3d',
    'test data':
    'flyingthings3d-TEST',
    'model':
    'StereoNet using difference to generate cost volume with scale is equal to three',
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


def test(imgL, imgR, disp_true):
    # model.eval()
    with torch.no_grad():
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


def test_all_model(model_dir, name_prefix, count):
    for i in range(count):
        savefilename = os.path.join(model_dir, f'test_{i}.pth')
        if os.path.exists(savefilename):
            continue
        model_name = os.path.join(model_dir, name_prefix + str(i) + '.pth')
        while not os.path.exists(model_name):
            time.sleep(30 * 60)
        state_dict = torch.load(model_name)
        model.load_state_dict(state_dict['model_state_dict'])
        # ------------- TEST ------------------------------------------------------------
        total_test_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            start_time = time.time()
            test_loss = test(imgL, imgR, disp_L)
            print('Model %d Iter (%d / %d) test loss = %.3f, time = %.2f' %
                  (i, batch_idx, len(TestImgLoader), test_loss,
                   time.time() - start_time))
            total_test_loss += test_loss

        total_test_loss /= len(TestImgLoader)
        print('total test loss = %.3f batch_idx = %d' % (total_test_loss,
                                                         batch_idx))
        torch.save({
            'test_loss': total_test_loss,
        }, savefilename)


if __name__ == "__main__":
    test_all_model(args.loadmodel, 'checkpoint_', 15)
