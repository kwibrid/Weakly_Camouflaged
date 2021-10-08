import cv2
import os
import json
# read instance seg (./dataset/ins_3040/)
# to generate a json file in ./dataset/intance_bbox_annotation.json

#run before: 1. instance seg gt files, 
#return: bbox locjson

if __name__ == '__main__':
    bbox_gt_path = './dataset/ins_3040/'
    # img_path = './dataset/img/'
    save_path = './dataset/instance_bbox/'
    image_dict = {}
    count = 0

    for i in os.listdir(bbox_gt_path):
        print(count)
        i_name = i.split('.')[0]
        i_jpg = i_name + ".jpg"
        i_png = i_name + ".png"
        gt = cv2.imread(bbox_gt_path + i_png, 0)
        # img = cv2.imread(img_path + i_jpg, 1)
        bbox_list = []
        gt_pixel_dic = {}
        for x_index, x in enumerate(gt):
            for y_index, y in enumerate(x):
                if y != 0 and y not in gt_pixel_dic:
                    gt_pixel_dic[y] = [(x_index, y_index)]
                elif y != 0 and y in gt_pixel_dic:
                    gt_pixel_dic[y] += [(x_index, y_index)]

        for key in gt_pixel_dic:
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
            for (x, y) in gt_pixel_dic[key]:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

            w = max_y - min_y
            h = max_x - min_x
            # cv2.rectangle(img, (min_y, min_x), (min_y + w, min_x + h), (0, 255, 0), 2)
            bbox_list.append((min_y, min_x, w, h))

        bounding_dict = {}
        bounding_dict['image_path'] = save_path + i_png
        bounding_dict['bbox'] = bbox_list
        image_dict[i_name] = bounding_dict

        # cv2.imwrite(save_path + i_png, img)
        count += 1

    with open('./dataset/instance_bbox_annotation.json', 'a+', encoding='utf-8') as f:
        json.dump(image_dict, f, ensure_ascii=False, indent=4)
