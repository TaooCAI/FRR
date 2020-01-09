import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
import time
import os
import visdom

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument(
    '--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--model', default='stereoSRR', help='select model')
parser.add_argument(
    '--datapath', default='/home/caitao/Dataset/sceneflow/', help='datapath')
parser.add_argument(
    '--loadmodel',
    default='./model_save_g32_retrain_from_scratch_using_flyingthings3D_TEST',
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


def test(imgL, imgR, disp_true):
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

        print(
            'total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
        torch.save({
            'test_loss': total_test_loss / len(TestImgLoader),
        }, savefilename)


if __name__ == "__main__":
    test_all_model(args.loadmodel, 'checkpoint_', 20)
