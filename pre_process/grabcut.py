"""
main pre-processing procedure:
using original cv2.grabCut function, the HED boundaries in SDI paper still nedd to do later
  --TODO grabcut+ and resize the saved image
using region proposal method to generate label for training from input bounding box and image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for custom database, grabCut algorithm can break when the bounding box's area is too small
and for the big object, like bounding box is just the whole image, it also need fix algorithm
for normal onject, if thE IOU of the region which generated by grabcut and the bounding box < threshold, 
we still need to use the original boudning box as segments 

besides, the coordinate[xmin,ymin,xmax,ymax] may be float decimal, and some of them may be beyond image.shape 
it need fixed to correct int number.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
------------------------------------------------------------------------------------------------------------------
as for the TBbacillus medicinal database,  the outcome seems  poorly... maybe I should only set the pixel-0 to 0.
++++++++++++++++++++++++++++++++++++
+  0 GC_BGD          --background      +
+ 1 GC_FGD           -- foreground     +
+ 2 GC_PR_BGD   --probable background  +
+ 3 GC_PR_FGD   --probable foreground  +
++++++++++++++++++++++++++++++++++++
Based on:
 https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html

This code is used for grabcut segments of images from train_pairs file,
however, due to lack of segmentation groundtruth, we also grabcut
all images im the dataset.--see grabcut_dataset.py
Same operation also did in box_i.

TODO How about the ICCV2019 new paper:EGNet? Maybe it will help.
  link: https://arxiv.org/pdf/1908.08297v1.pdf
"""

import json
import os

import setproctitle
from tqdm import tqdm
import cv2
import numpy as np
import torch


def parse_json(bbox_root):
    bbox_list = []
    bbox_image_name_list = []
    with open(bbox_root) as f:
        data_dic = json.load(f)
    for image_name in data_dic:
        bbox_image_name_list.append(image_name)
        bbox = data_dic[image_name]['bbox']
        bbox_list.append(bbox)
    return bbox_list, bbox_image_name_list



def grabcut(bbox_list, bbox_image_name_list,save_path,device='cpu'):
    current_i = 0
    for bbox_image_index, bbox_image_name in enumerate(tqdm(bbox_image_name_list)):
        scribble_image_full_path = scribble_images_path+bbox_image_name+".jpg"
        ##########
        img = cv2.imread(scribble_image_full_path,-1)
        bboxes = bbox_list[bbox_image_index]
        # mask2 = np.zeros(img.shape[:2], np.uint8)
        mask2 = torch.zeros(img.shape[:2]).to(torch.uint8).to(device)
        for bbox in bboxes:
            mask = np.zeros(img.shape[:2], np.uint8)
            x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, (x,y,w,h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = torch.from_numpy(mask.copy()).to(device)
            mask2 += torch.where((mask == 2) | (mask == 0), 0, 1).to(torch.uint8)
        mask2 = torch.where(mask2 ==0, 0, 1).to(torch.uint8)

        blank_img = torch.ones_like(torch.from_numpy(img))*255
        blank_img = blank_img.to(device)
        img = blank_img * mask2[:, :, np.newaxis]
        img = img.to('cpu').numpy()
        cv2.imwrite(save_path+bbox_image_name +'.png',img)
        ########
        current_i+=1


if __name__ == '__main__':
    setproctitle.setproctitle('GrabCut 3040')
    scribble_images_path = '../dataset/img/train_img/'
    save_path = '../dataset/grabcut/'
    bbox_root = '../dataset/instance_bbox_annotation.json'
    if not os.path.exists(scribble_images_path):
        raise Exception('scribble path not existed!')
    if not os.path.isfile(bbox_root):
        raise Exception('bbox path not existed!')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    bbox_list, bbox_image_name_list = parse_json(bbox_root)
    # load_ann()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    grabcut(bbox_list, bbox_image_name_list,save_path,device=device)
