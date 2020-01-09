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
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument(
    '--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--model', default='stackhourglass', help='select model')
parser.add_argument(
    '--datapath', default='/home/caitao/Dataset/sceneflow/', help='datapath')
parser.add_argument(
    '--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--batchsize', type=int, default=2, help='batch size')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument(
    '--savemodel',
    default='./model_save_g32_retrain_from_scratch_using_flyingthings3D_TEST',
    help='save model')
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

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
    args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=min(4, args.batchsize),
    drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=min(4, args.batchsize),
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
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    print('Load model %s success......' % (args.loadmodel))

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))


def train(imgL, imgR, disp_L, x_pos, loss_window, vis, image_groundtruth,
          image_output1, image_output2, image_output3):
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

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(
            output1[mask], disp_true[mask],
            size_average=True) + 0.7 * F.smooth_l1_loss(
                output2[mask], disp_true[mask],
                size_average=True) + F.smooth_l1_loss(
                    output3[mask], disp_true[mask], size_average=True)
    elif args.model == 'basic':
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3, 1)
        loss = F.smooth_l1_loss(
            output3[mask], disp_true[mask], size_average=True)
    elif args.model == 'stereoSRR':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(
            output1[mask], disp_true[mask],
            reduction='elementwise_mean') + 0.7 * F.smooth_l1_loss(
                output2[mask], disp_true[mask], reduction='elementwise_mean'
            ) + F.smooth_l1_loss(
                output3[mask], disp_true[mask], reduction='elementwise_mean')

        vis.line(
            X=torch.ones((1, )).cpu() * x_pos,
            Y=torch.Tensor([loss.item()]).cpu(),
            win=loss_window,
            update='append')

        tmp = torch.where(output1.data[0] < 0,
                          torch.zeros_like(output1.data[0]), output1.data[0])
        tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)
        vis.image(
            tmp.cpu(), win=image_output1, opts=dict(title='output1_psm_g32'))

        tmp = torch.where(output2.data[0] < 0,
                          torch.zeros_like(output2.data[0]), output2.data[0])
        tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)
        vis.image(
            tmp.cpu(), win=image_output2, opts=dict(title='output2_psm_g32'))

        tmp = torch.where(output3.data[0] < 0,
                          torch.zeros_like(output3.data[0]), output3.data[0])
        tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)
        vis.image(
            tmp.cpu(), win=image_output3, opts=dict(title='output3_psm_g32'))

        tmp = torch.where(disp_true.data[0] < 0,
                          torch.zeros_like(disp_true.data[0]),
                          disp_true.data[0])
        tmp = torch.where(tmp > 255, torch.full_like(tmp, 255), tmp)
        vis.image(
            tmp.cpu(), win=image_groundtruth, opts=dict(title='gt_psm_g32'))

    loss.backward()
    optimizer.step()

    return loss.item()


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


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    x_pos = 0
    vis = visdom.Visdom(port=9999)
    loss_window = vis.line(
        X=torch.zeros((1, )).cpu(),
        Y=torch.zeros((1, )).cpu(),
        opts=dict(
            xlabel='batches',
            ylabel='loss',
            title='Trainingloss_psm_g32',
            legend=['loss']))
    A = torch.randn([544, 960])
    A = (A - torch.min(A)) / torch.max(A)
    image_groundtruth = vis.image(
        A.cpu(), opts=dict(title='groundtruth_psm_g32'))
    image_output1 = vis.image(A.cpu(), opts=dict(title='output1_psm_g32'))
    image_output2 = vis.image(A.cpu(), opts=dict(title='output2_psm_g32'))
    image_output3 = vis.image(A.cpu(), opts=dict(title='output3_psm_g32'))

    train_loss_log = []
    eval_loss_log = []
    start_full_time = time.time()
    for epoch in range(args.epochs):
        total_train_loss = 0
        # adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop,
                        disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L, x_pos, loss_window,
                         vis, image_groundtruth, image_output1, image_output2,
                         image_output3)
            x_pos += 1
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

        total_train_loss = total_train_loss / len(TrainImgLoader)
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
                if batch_idx * args.batchsize > 400:
                    break
                start_time = time.time()
                eval_loss = test(imgL, imgR, disp_L)
                print('Model %d Iter (%d / %d) eval loss = %.3f, time = %.2f' %
                      (epoch, batch_idx, len(TestImgLoader), eval_loss,
                       time.time() - start_time))
                total_eval_loss += eval_loss

            total_eval_loss = total_eval_loss / batch_idx
            print('total eval loss = %.3f' % (total_eval_loss))
            eval_loss_log.append(total_eval_loss)

        # print previous train and eval loss after finishing each epoch
        #
        # snack
        print('train loss log for previous epoch: ', train_loss_log)
        print('eval loss log for previous epoch: ', eval_loss_log)
        with open(
                os.path.join(
                    args.model,
                    'train_and_eval_loss_log_every_epoch_tmp_98741.txt'),
                'a') as f:
            f.write(str([epoch, total_train_loss, total_eval_loss]))
            f.write('\n')

    # save train loss and eval loss after every epoch
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
    with open(os.path.join(args.model, 'training cost time.txt'), 'a') as f:
        f.write(f'full training time: {all_consume_time} hours')
        f.write('\n')


if __name__ == '__main__':
    if os.path.exists(args.savemodel):
        raise Exception(f'{args.savemodel} existed.')
    os.makedirs(args.savemodel)
    main()
