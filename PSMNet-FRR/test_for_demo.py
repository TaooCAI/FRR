import argparse
import time
import os
import shutil

import visdom
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from dataloader import listflowfile as lt
from dataloader import SceneFlowLoader_demo as DA
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument(
    '--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--model', default='stereoSRR', help='select model')
parser.add_argument(
    '--datapath', default='/home/caitao/Dataset/sceneflow/', help='datapath')
parser.add_argument(
    '--loadmodel',
    default=
    './model_save_g32_retrain_from_scratch_using_flyingthings3D_TEST/checkpoint_18.pth',
    help='load model')
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
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'stereoSRR':
    model = stereoSRR(args.maxdisp)
else:
    print('no model')
print(f'Use model {args.model}')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()


def save_prediction_for_demo(prediction, save_dir, left_path, disp_path,
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

    # copy original left image
    shutil.copyfile(left_path[0],
                    os.path.join(first, os.path.basename(left_path[0])))
    shutil.copyfile(left_path[1],
                    os.path.join(second, os.path.basename(left_path[1])))
    # save the disparity map
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

    # save the prediction
    tmp = torch.where(prediction.data < 0, torch.zeros_like(prediction.data),
                      prediction.data)
    tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)
    tmp = tmp / 255
    prediction_img = transforms.ToPILImage()(torch.unsqueeze(tmp[0],
                                                             dim=0).cpu())
    prediction_img.save(os.path.join(first, 'prediction.png'))

    prediction_img = transforms.ToPILImage()(torch.unsqueeze(tmp[1],
                                                             dim=0).cpu())
    prediction_img.save(os.path.join(second, 'prediction.png'))


def test(imgL, imgR, disp_true, save_dir, left_path, disp_path, batch_idx):
    model.eval()
    with torch.no_grad():
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
        # ---------
        mask = (disp_true < args.maxdisp) & (disp_true >= 0)
        # ----
        output3 = model(imgL, imgR)
        output = torch.squeeze(output3.data.cpu(), 1)[:, 4:, :]
        save_prediction_for_demo(output, save_dir, left_path, disp_path,
                                 str(batch_idx))
        if len(disp_true[mask]) == 0:
            return 0
        else:
            # end-point-error
            return (torch.mean(
                torch.abs(output[mask] - disp_true[mask]))).item()


def test_model(model_path, save_dir, batch_count):

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model_state_dict'])
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
               './demo_g32_retrain_from_scratch_using_flyingthings3D_TEST', 10)
