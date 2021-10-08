import os
import cv2
#accroding to the 'ins_3040', which only contains instance seg gts, we would like the files in ins_3040 to be train imgs
#we would like to get the filenames of this train imgs and split the data with train dataset and test dataset
# run before: ins_3040 which can provide train imgs names
# return: 2 folders which have train imgs and test imgs

if __name__ == '__main__':
    train_images_path = './dataset/ins_3040/'
    current_img_path = './dataset/gt/'

    train_img_save_path = './dataset/gt/train_img/'
    test_img_save_path = './dataset/gt/test_img/'

    if not os.path.exists(train_img_save_path):
        os.mkdir(train_img_save_path)
    if not os.path.exists(test_img_save_path):
        os.mkdir(test_img_save_path)


    count = 0
    for i in os.listdir(current_img_path):
        print(count)
        if i.split('.')[0] + '.png' in os.listdir(train_images_path):
            img = cv2.imread(current_img_path + i, 1)
            # cv2.imwrite(train_img_save_path + i.split('.')[0] + '.jpg', img)
            cv2.imwrite(train_img_save_path + i.split('.')[0] + '.png', img)
        else:
            img = cv2.imread(current_img_path + i, 1)
            # cv2.imwrite(test_img_save_path + i.split('.')[0] + '.jpg', img)
            cv2.imwrite(test_img_save_path + i.split('.')[0] + '.png', img)
        count += 1
