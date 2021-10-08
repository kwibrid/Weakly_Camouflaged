import json
import random
import cv2
from detectron2.utils.visualizer import Visualizer

def camouflaged_dataset_function(bbox_root):
    with open(bbox_root) as f:
        bbox_list = json.load(f)
        return bbox_list


from detectron2.data import DatasetCatalog

DatasetCatalog.register("camouflaged_dataset", camouflaged_dataset_function)
# later, to access the data:
dataset_dicts = camouflaged_dataset_function('./dataset/detection2_bbox_annotation.json')
for d in random.sample(dataset_dicts, 3):
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('', out.get_image()[:, :, ::-1])
    cv2.waitKey()
