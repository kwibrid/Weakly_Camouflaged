import sys

import cv2
import numpy as np
import torch
np.set_printoptions(threshold=sys.maxsize)

def grabcut(device='cpu'):
    img = cv2.imread("../dataset/img/train_img/COD10K-CAM-1-Aquatic-1-BatFish-8.jpg",-1)
    # bboxes = [[ 112,203,552,239]]
    # bboxes = [[311,64,58,73],[152,415,118,96]]
    bboxes = [[70,40,314,558]]
    # FROM SCRIBBLE
    ori_mask = cv2.imread("../dataset/instance_bbox_and_scribbles/COD10K-CAM-1-Aquatic-1-BatFish-8.png",-1) #MUST CONVERT
    ori_mask = ori_mask/255
    ori_mask = ori_mask.astype('uint8')
    #FROM BBOX_SEG
    bbox_seg_mask = cv2.imread("../dataset/bbox_rect_gt/train_img/COD10K-CAM-1-Aquatic-1-BatFish-8.png",-1)  # MUST CONVERT
    bbox_seg_mask = bbox_seg_mask / 255
    bbox_seg_mask = bbox_seg_mask.astype('uint8')

    mask1 = torch.zeros(img.shape[:2]).to(torch.uint8).to(device)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    for bbox in bboxes:
        mask = ori_mask.copy()
        x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
        cv2.grabCut(img, mask, (x, y, w, h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask = torch.from_numpy(mask.copy()).to(device)
        mask1 += torch.where((mask == 2) | (mask == 0), 0, 1).to(torch.uint8)

    mask1 = mask1.numpy()


    ori_mask = np.where((bbox_seg_mask == 0), 0, ori_mask)  #if outside of bbox, then 0
    ori_mask = np.where((bbox_seg_mask == 1)&(ori_mask==0), 3, ori_mask) #encourage pixel inside bbox to be 1, so 3
    ori_mask = np.where(((mask1 == 1) & (ori_mask == 0)), 1, ori_mask)  #try to keep original mask to be 1
    #TODO ADD MORE MASK AND LET MASK BE 3 (POSSIBLE FOREHEAD)

    mask, bgdModel, fgdModel = cv2.grabCut(img, ori_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img2 = np.where(mask[:, :, np.newaxis] ==1,255,0)

    cv2.imwrite('final_output.png',img2)


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    #     device = torch.device('cpu')
    # grabcut(device)
    grabcut()