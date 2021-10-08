import sys
import os
import cv2
import numpy as np
import setproctitle
import torch
import json
# np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm


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


def grabcut(bbox_list, bbox_image_name_list, save_path,device='cpu'):
    for bbox_image_index, bbox_image_name in enumerate(tqdm(bbox_image_name_list)):
        train_images_full_path = train_images_path + bbox_image_name + ".jpg"
        scribble_images_full_path = scribble_images_path+bbox_image_name+ ".png"
        bbox_seg_images_full_path = bbox_seg_images_path + bbox_image_name + ".png"
        mb_plus_images_full_path = mb_plus_images_path +bbox_image_name+'_MB+.png'

        img = cv2.imread(train_images_full_path, -1)
        bboxes = bbox_list[bbox_image_index]
        # FROM SCRIBBLE
        ori_mask = cv2.imread(scribble_images_full_path,-1)  # MUST CONVERT
        ori_mask = ori_mask / 255
        ori_mask = ori_mask.astype('uint8')
        # FROM BBOX_SEG
        bbox_seg_mask = cv2.imread(bbox_seg_images_full_path,-1)  # MUST CONVERT
        bbox_seg_mask = bbox_seg_mask / 255
        bbox_seg_mask = bbox_seg_mask.astype('uint8')
        bbox_seg_mask = torch.tensor(bbox_seg_mask).to(device)
        #from mb_plus
        mb_plus_mask = cv2.imread(mb_plus_images_full_path,-1)
        (thresh, mb_plus_mask) = cv2.threshold(mb_plus_mask, 127, 255, cv2.THRESH_BINARY)
        mb_plus_mask = mb_plus_mask / 255
        mb_plus_mask = mb_plus_mask.astype('uint8')
        mb_plus_mask = torch.tensor(mb_plus_mask).to(device)

        mask1 = torch.zeros(img.shape[:2]).to(torch.uint8).to(device)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        for bbox in bboxes:
            mask = ori_mask.copy()
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.grabCut(img, mask, (x, y, w, h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = torch.from_numpy(mask.copy()).to(device)
            mask1 += torch.where((mask == 2) | (mask == 0), 0, 1).to(torch.uint8)

        # mask1 = mask1.cpu().numpy()
        ori_mask = torch.tensor(ori_mask).to(device)
        ori_mask = torch.where((bbox_seg_mask == 0), torch.zeros_like(bbox_seg_mask), ori_mask)  # if outside of bbox, then 0 (background)
        ori_mask = torch.where((bbox_seg_mask == 1) &(mb_plus_mask ==1) &(ori_mask == 0), torch.ones_like(bbox_seg_mask)*3,ori_mask)  # encourage pixel inside bbox to be 1, so 3
        ori_mask = torch.where((bbox_seg_mask == 1) & (mb_plus_mask == 0)& (ori_mask == 0), torch.ones_like(bbox_seg_mask)*2, ori_mask) #
        ori_mask = torch.where(((mask1 == 1) & (ori_mask == 0)), torch.ones_like(bbox_seg_mask)*3, ori_mask)  # try to keep original mask to be 1 or 3
        # TODO ADD MORE MASK AND LET MASK BE 3 (POSSIBLE FOREHEAD)

        ori_mask = ori_mask.cpu().numpy()
        mask, bgdModel, fgdModel = cv2.grabCut(img, ori_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask = torch.tensor(mask).to(device)
        mask = torch.where((mask == 2) | (mask == 0), 0, 1).to(torch.uint8)
        img2 = torch.where(mask[:, :, np.newaxis] == 1, 255, 0)

        img2 = img2.cpu().numpy()
        cv2.imwrite(save_path + bbox_image_name + '.png', img2)


if __name__ == '__main__':

    setproctitle.setproctitle('New GrabCut 3040')
    train_images_path = '../dataset/img/train_img/'
    scribble_images_path = '../dataset/instance_bbox_and_scribbles/'
    bbox_seg_images_path = '../dataset/bbox_rect_gt/train_img/'
    mb_plus_images_path = '../dataset/mb_plus/'

    save_path = '../dataset/grabcut_mb_plus/'
    bbox_root = '../dataset/instance_bbox_annotation.json'

    if not os.path.exists(train_images_path):
        raise Exception('scribble path not existed!')
    if not os.path.isfile(bbox_root):
        raise Exception('bbox path not existed!')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    bbox_list, bbox_image_name_list = parse_json(bbox_root)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    grabcut(bbox_list, bbox_image_name_list, save_path, device=device)

