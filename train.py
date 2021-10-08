from datetime import datetime

import setproctitle
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation,label_edge_prediction
import smoothness
from tools import *
from lscloss import *


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=14, help='training batch size (origin 30)')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
parser.add_argument('--edge_loss_epoch', type=int, default=2, help='weight for edge loss')
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Generator(channel=opt.feat_channel)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)


# image_root = '../data/img/'
# gt_root = '../data/gt/'

# image_root = '/home/jingzhang/jing_files/scribble_data/img/'
# gt_root = '/home/jingzhang/jing_files/scribble_data/gt/'
# mask_root = '/home/jingzhang/jing_files/scribble_data/mask/'
# grayimg_root = '/home/jingzhang/jing_files/scribble_data/gray/'

image_root = './dataset/img/train_img/' #3040 train 不用变
# gt_root = './dataset/instance_bbox_and_scribbles/' #bbox+scribble
gt_root = './dataset/grabcut_mb_plus/' #new grabcut
# gt_root = './dataset/scribble_data/gt/train_img/' #纯gt
# gt_root = './dataset/bbox_rect_gt/train_img/' #bbox seg label
mask_root = './dataset/scribble_data/mask/train_img/' #scrrible (foreground+background) 不用变
grayimg_root = './dataset/gray/train_img/' #3040 gray 不用变

train_loader = get_loader(image_root, gt_root, mask_root, grayimg_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

scheduler = lr_scheduler.StepLR(generator_optimizer, step_size=20, gamma=0.9)
CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  # multi-scale training
smooth_loss = smoothness.smoothness_loss(size_average=True)
loss_lsc = LocalSaliencyCoherence().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
weight_lsc = 0.3

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def visualize_prediction_init(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_init.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_prediction_ref(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_edge(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_edge.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_original_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

setproctitle.setproctitle('Training 3040')
print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    scheduler.step()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, masks, grays = pack
            images = Variable(images)
            gts = Variable(gts)
            masks = Variable(masks)
            grays = Variable(grays)
            images = images.cuda()
            gts = gts.cuda()
            masks = masks.cuda()
            grays = grays.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            sal1, sal2 = generator.forward(images)
            img_size = images.size(2) * images.size(3) * images.size(0)
            ratio = img_size / torch.sum(masks)

            # images_scale = F.interpolate(images, scale_factor=0.6, mode='bilinear', align_corners=True)
            # sal1_scale, sal2_scale = generator(images_scale)
            # sal1_s = F.interpolate(sal1, scale_factor=0.6, mode='bilinear', align_corners=True)
            # sal2_s = F.interpolate(sal2, scale_factor=0.6, mode='bilinear', align_corners=True)
            # loss_ssc1 = SaliencyStructureConsistency(torch.sigmoid(sal1_scale), torch.sigmoid(sal1_s), 0.85)
            # loss_ssc2 = SaliencyStructureConsistency(torch.sigmoid(sal2_scale), torch.sigmoid(sal2_s), 0.85)
            # loss_ssc_all = 0.5 * (loss_ssc1 + loss_ssc2)
            #
            # images_ = F.interpolate(images, scale_factor=0.6, mode='bilinear', align_corners=True)
            # sample = {'rgb': images_}
            # loss_lsc1 = \
            #     loss_lsc(torch.sigmoid(sal1_s), loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample,
            #              images_.shape[2], images_.shape[3])[
            #         'loss']
            # loss_lsc2 = \
            #     loss_lsc(torch.sigmoid(sal2_s), loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample,
            #              images_.shape[2], images_.shape[3])[
            #         'loss']
            # loss_lsc_all = 0.5 * (loss_lsc1 + loss_lsc2)

            sal1_prob = torch.sigmoid(sal1)
            sal1_prob = sal1_prob * masks
            sal2_prob = torch.sigmoid(sal2)
            sal2_prob = sal2_prob * masks
            smoothLoss_cur1 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal1), grays)
            smoothLoss_cur2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal2), grays)
            sal_loss1 = ratio * CE(sal1_prob, gts * masks) + smoothLoss_cur1
            sal_loss2 = ratio * CE(sal2_prob, gts * masks) + smoothLoss_cur2

            sal_loss = 0.5*(sal_loss1+sal_loss2)
            # sal_loss = 0.5 * (sal_loss1 + sal_loss2) + loss_ssc_all + weight_lsc * loss_lsc_all

            sal_loss.backward()
            generator_optimizer.step()

            visualize_prediction_init(torch.sigmoid(sal1))
            visualize_prediction_ref(torch.sigmoid(sal2))
            visualize_gt(gts)
            # visualize_original_img(images)

            if rate == 1:
                loss_record.update(sal_loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '.pth')