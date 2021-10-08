import denseCRF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from itertools import product

def densecrf(I, P, param):
    """
    input parameters:
        I    : a numpy array of shape [H, W, C], where C should be 3.
               type of I should be np.uint8, and the values are in [0, 255]
        P    : a probability map of shape [H, W, L], where L is the number of classes
               type of P should be np.float32
        param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
    output parameters:
        out  : a numpy array of shape [H, W], where pixel values represent class indices. 
    """
    out = denseCRF.densecrf(I, P, param) 
    return out   


def demo_densecrf1(I_path,L_path,save_path):
    # bbox = [112,203,552,239]
    # x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

    img_name = I_path.split('/')[-1].split('.')[0]
    I  = Image.open(I_path)
    Iq = np.asarray(I)

    # sobelxy = cv2.Sobel(src=Iq, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # new_image1 = np.zeros_like(sobelxy)
    # new_image1[y:y + h, x:x + w,:] = sobelxy[y:y + h, x:x + w,:]
    # cv2.imwrite(save_path + img_name + "_sobel.png",new_image1)

    edges = cv2.Canny(image=Iq, threshold1=100, threshold2=200)
    # new_image2 = np.ones_like(edges)*255
    # new_image2[y:y + h, x:x + w] = edges[y:y + h, x:x + w]
    # cv2.imwrite(save_path + img_name + "_canny1.png", new_image2)
    cv2.imwrite(save_path + img_name + "_canny2.png", edges)

    # load initial labels, and convert it into an array 'prob' with shape [H, W, C]
    # where C is the number of labels
    # prob[h, w, c] means the probability of pixel at (h, w) belonging to class c.
    L  = Image.open(L_path).convert('RGB')
    # Lq = np.asarray(255-new_image2, np.float32) / 255
    # Lq = cv2.cvtColor(Lq, cv2.COLOR_GRAY2RGB)
    Lq = np.asarray(L, np.float32) / 255
    prob = Lq[:, :, :2]
    prob[:, :, 0] = 1.0 - prob[:, :, 0]

    w1    = 10.0  # weight of bilateral term
    alpha = 80    # spatial std
    beta  = 13    # rgb  std
    w2    = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it    = 5.0   # iteration

    # w1    = 30.0  # weight of bilateral term
    # alpha = 100    # spatial std 空间方差越大 分母越大 能量越小 空间距离影响越小,能让图片收缩
    # beta  = 10    # rgb  std    颜色方差越大 分母越大 能量越小  颜色差别影响越小,能让图片收缩
    # w2    = 50.0   # weight of spatial term
    # gamma = 50    # spatial std 空间方差越大 分母越大 能量越小 空间距离影响越小 能让图片收缩
    # it    = 5.0   # iteration

    param = (w1, alpha, beta, w2, gamma, it)
    lab = densecrf(Iq, prob, param)
    cv2.imwrite(save_path + img_name + ".png", lab * 255)

    # cv2.imwrite(save_path+img_name+".png",lab*255)

def edge_process(I_path,save_path):

    img_name = I_path.split('/')[-1].split('.')[0]
    I  = Image.open(I_path)
    Iq = np.asarray(I)

    Iq = cv2.cvtColor(Iq, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(save_path + img_name + ".png", Iq)

    Iq = cv2.GaussianBlur(Iq, (5, 5), cv2.BORDER_DEFAULT)
    sobelxy = cv2.Sobel(src=Iq, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    cv2.imwrite(save_path + img_name + "_sobel.png",sobelxy)

    edges = cv2.Canny(image=Iq, threshold1=100, threshold2=200)
    cv2.imwrite(save_path + img_name + "_canny.png", edges)


if __name__ == "__main__":
    I_path = '../dataset/img/train_img/'
    L_path = '../dataset/bbox_rect_gt/train_img/'
    # save_path = '../dataset/CRF/train_img/'
    save_path = '../dataset/CRF/'
    if not os.path.exists(I_path):
        raise Exception('RGB img path not existed!')
    if not os.path.exists(L_path):
        raise Exception('bbox seg path not existed!')
    if not os.path.exists(save_path):
        # if not os.path.exists(save_path.split('train_img')[0]):
        #     os.mkdir(save_path.split('train_img')[0])
        os.mkdir(save_path)

    for i_index,i in enumerate(os.listdir(I_path)):
        print(i_index)
        if i_index>0:
            break
        i_name = i.split('.')[0]
        i_jpg = i_name + ".jpg"
        i_png = i_name + ".png"
        I = I_path+i_jpg
        L = L_path+i_png
        # demo_densecrf1(I,L,save_path)
        edge_process(I,save_path)