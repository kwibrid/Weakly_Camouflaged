import torch
print(torch.cuda.is_available())
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
from model.ResNet_models import Pred_endecoder
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2



from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')
opt = parser.parse_args()

dataset_path = '../BaseModel/cod_test_dataset/'
# save_path = './results/results_grab_mb/'
# save_path1 = './results/crf_11_6/'
# save_path2 = './results/crf_11_6_uncertainty/'

save_path1 = './results/crf_11_7/'
save_path2 = './results/crf_11_7_uncertainty/'

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load('./models/Model_50_crf_11_7.pth'))

generator.cuda()
generator.eval()

test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
# test_datasets = ['CAMO', 'CHAMELEON']
# test_datasets = ['COD10K', 'NC4K']

if not os.path.exists(save_path1):
    os.makedirs(save_path1)

for dataset in test_datasets:
    save_test_path1 = save_path1 + dataset + "/"
    save_test_path2 = save_path2 + dataset + "/"

    if not os.path.exists(save_test_path1):
        os.makedirs(save_test_path1)
    if not os.path.exists(save_test_path2):
        os.makedirs(save_test_path2)
    image_root = dataset_path+dataset+"/Imgs/"
    test_loader = test_dataset(image_root, opt.testsize)

    for i in tqdm(range(test_loader.size)):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        generator_pred, _ = generator.forward(image)

        res = generator_pred
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_test_path1 + name, res)

        # res_alea = torch.exp(generator_alea)
        # res_alea = F.upsample(res_alea, size=[WW, HH], mode='bilinear', align_corners=False)
        # res_alea = res_alea.sigmoid().data.cpu().numpy().squeeze()
        # res_alea = 255 * (res_alea - res_alea.min()) / (res_alea.max() - res_alea.min() + 1e-8)
        # cv2.imwrite(save_test_path2 + name, res_alea)


