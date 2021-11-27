import argparse
import os

import setproctitle
import torch.nn.functional as F
from torch.autograd import Variable

import smoothness

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from datetime import datetime
from model.ResNet_models import Pred_endecoder
from data import get_loader
from utils import adjust_lr, AvgMeter
import cv2
import torchvision.transforms as transforms
from tools import *

setproctitle.setproctitle('crf loss!')
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--focal_lamda', type=int, default=1, help='lamda of focal loss')
parser.add_argument('--bnn_steps', type=int, default=6, help='BNN sampling iterations')
parser.add_argument('--lvm_steps', type=int, default=6, help='LVM sampling iterations')
parser.add_argument('--pred_steps', type=int, default=6, help='Predictive sampling iterations')
parser.add_argument('--smooth_loss_weight', type=float, default=0.5, help='weight of the smooth loss')
parser.add_argument('--alex_loss_weight', type=float, default=0.3, help='weight of the alex loss')
parser.add_argument('--reg_loss_weight', type=float, default=0.3, help='weight of the reg loss')
parser.add_argument('--sample_steps', type=int, default=6, help='Predictive sampling iterations')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Pred_endecoder(channel=opt.feat_channel)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

image_root = '../BaseModel/dataset/img/train_img/'
# gt_root = '../BaseModel/dataset/mb_grab_crf/'
gt_root = '../BaseModel/pre_gt_dataset/mb_grab_new/'
gray_root = '../BaseModel/dataset/gray/train_img/'

train_loader = get_loader(image_root, gt_root, gray_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  # multi-scale training
smooth_loss = smoothness.smoothness_loss(size_average=True)


def structure_loss(pred, mask, temperature):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1 - epsilon) * gts + epsilon / 2
        return new_gts

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    new_gts = generate_smoothed_gt(mask)

    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='none')  # MAYBE SIGMOID ON PRED
    wbce = (weit * wbce / temperature).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def my_structure_loss(pred, mask, sig2):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1 - epsilon) * gts + epsilon / 2
        return new_gts

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    new_gts = generate_smoothed_gt(mask)

    wbce = F.binary_cross_entropy_with_logits(pred / sig2, new_gts, reduction='none')  # MAYBE SIGMOID ON PRED
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def structure_loss1(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1 - epsilon) * gts + epsilon / 2
        return new_gts

    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def structure_loss_focal_loss(pred, mask, weight):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1 - epsilon) * gts + epsilon / 2
        return new_gts

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='none')
    wbce = (((1 - weight) ** opt.focal_lamda) * weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def visualize_gt(var_map):
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_pred4(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred4.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_pred43(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred43.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_pred432(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred432.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_pred(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_alea(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_alea.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_alea_entropy(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_alea_entropy.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_original_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.4850 / .229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk, :, :, :]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1, 2, 0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path + name, new_img)


def no_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()


def yes_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


print("Let's go!")
for epoch in range(1, (opt.epoch + 1)):
    # scheduler.step()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, grays = pack
            images = Variable(images)
            gts = Variable(gts)
            grays = Variable(grays)
            images = images.cuda()
            gts = gts.cuda()
            grays = grays.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            pred, alea = generator(images)

            sal_pred_list = torch.sigmoid(pred)
            entropy_all = -sal_pred_list * torch.log(sal_pred_list + 1e-8)
            with torch.no_grad():
                for iter in range(opt.sample_steps):
                    temp, _ = generator(images)
                    temp = torch.sigmoid(temp)
                    entropy = -temp * torch.log(temp + 1e-8)
                    entropy_all = entropy_all + entropy
            entropy_all = entropy_all / (opt.sample_steps + 1)
            s = alea
            sig2 = torch.exp(s)
            smoothLoss = opt.smooth_loss_weight * smooth_loss(torch.sigmoid(pred), grays)  # smooth loss

            normed_sig2 = (sig2 - sig2.min()) / (sig2.max() - sig2.min() + 1e-8) #alea
            normed_entropy = (entropy_all - entropy_all.min()) / (entropy_all.max() - entropy_all.min() + 1e-8) #感觉是pred

            alea_loss = opt.alex_loss_weight * CE(normed_sig2, normed_entropy.detach()) #alea and pred same, then epistemic 0
            tempreature = torch.exp(sig2)
            reg = opt.reg_loss_weight * torch.log(tempreature).sum(dim=1)  # 方差 求和*1/n
            # reg = opt.reg_loss_weight * torch.log(sig2).sum(dim=1)  # 方差 求和*1/n
            entropy_loss = -torch.sigmoid(pred) * torch.log(torch.sigmoid(pred) + 1e-8)
            entropy_loss = entropy_loss.sum(dim=1)
            entropy_loss = entropy_loss.mean()
            # loss_all = structure_loss(pred, gts, tempreature) + reg.mean()  + entropy_loss +smoothLoss+ alea_loss
            loss_all = structure_loss(pred, gts, tempreature) + reg.mean() + smoothLoss + alea_loss
            loss_all.backward()
            generator_optimizer.step()

            visualize_pred(torch.sigmoid(pred))
            visualize_alea(normed_sig2)
            visualize_alea_entropy(normed_entropy)
            visualize_gt(gts)

            if rate == 1:
                loss_record.update(loss_all.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_crf_11_7.pth')

###因为gt带噪声，且这部分噪声无法被清除掉，我们可以去通过引入不确定性去估算这个噪声，假设输出了一个图像，他因为要尽可能接近伪
##gt，所以也是带了噪声的，那么我们就可以在训练的过程中，去计算 衡量这个噪声，我们不断拟合一个趋势，这导致我们输出和伪gt之间的差距 其实就是data uncertatinty
# 感觉pred还是去近似伪gt去了，所以要真想还原 还是要除一下方差？