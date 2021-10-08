import cv2
import os
import json
# read instance seg (./dataset/ins_3040/)
# to generate a json file in ./dataset/intance_bbox_annotation.json

#run before: 1. instance seg gt files,
#return: bbox locjson

from detectron2.structures import BoxMode

if __name__ == '__main__':
    bbox_gt_path = './dataset/ins_3040/'
    # img_path = './dataset/img/'
    save_path = './dataset/instance_bbox/'
    image_dict = []
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
            annotations_dict ={}
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
            annotations_dict['bbox']= [min_y, min_x, w, h]
            annotations_dict['bbox_mode'] = BoxMode.XYWH_ABS
            annotations_dict['category_id'] = 1
            bbox_list.append(annotations_dict)

        bounding_dict = {}
        bounding_dict['file_name'] = save_path + i_png
        bounding_dict['height'] = gt.shape[0]
        bounding_dict['width'] = gt.shape[1]
        bounding_dict['image_id'] = count
        bounding_dict['annotations'] = bbox_list
        image_dict.append(bounding_dict)

        # cv2.imwrite(save_path + i_png, img)
        count += 1

    with open('./dataset/detection2_bbox_annotation.json', 'a+', encoding='utf-8') as f:
        json.dump(image_dict, f, ensure_ascii=False, indent=4)
