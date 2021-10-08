import json
import os
import cv2
import numpy as np

#combine bbox info (./dataset/instance_bbox_annotation.json)
# and scribble info (./dataset/scribble_data/gt/)together, to generate a new gt in ./dataset/instance_bbox_and_scribbles/

#run before: 1. bbox json file, scrrible files
#return: new gt instance_bbox_and_scribbles

#get bbox list
def parse_json(bbox_root = './dataset/instance_bbox_annotation.json'):
    bbox_list = []
    bbox_image_name_list = []
    with open(bbox_root) as f:
        data_dic = json.load(f)
    for image_name in data_dic:
        bbox_image_name_list.append(image_name)
        bbox = data_dic[image_name]['bbox']
        bbox_list.append(bbox)
    return bbox_list,bbox_image_name_list

if __name__ == '__main__':
    scribble_images_path = './dataset/scribble_data/gt/'
    scribble_images = [scribble_images_path + f for f in os.listdir(scribble_images_path) if f.endswith('.png')]

    bbox_list,bbox_image_name_list = parse_json()
    current_i = 0
    for bbox_image_index,bbox_image_name in enumerate(bbox_image_name_list):
        scribble_image_full_path = scribble_images_path+bbox_image_name+".png"

        img = cv2.imread(scribble_image_full_path,0)
        new_image = np.zeros_like(img)

        bboxes = bbox_list[bbox_image_index]
        for bbox in bboxes:
            x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
            new_image[y:y+h,x:x+w] = img[y:y+h,x:x+w]
            # cv2.rectangle(new_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imwrite('./dataset/instance_bbox_and_scribbles/'+bbox_image_name +'.png',new_image)
        current_i+=1
        print(str(current_i)+bbox_image_name)
