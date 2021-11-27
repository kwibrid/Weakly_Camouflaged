# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import json
import cv2
from tabulate import tabulate
from tqdm import tqdm
import time
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
#camo dataset
FM = Fmeasure() #b mean
WFM = WeightedFmeasure() #没用过
SM = Smeasure() #yes
EM = Emeasure() #mean
MAE = MAE() #lower better

# pred_root = '../results/results_grab_mb/'
# pred_root = '../results/results_gt/'
# pred_root = '../results/results_bbox_scri/'
pred_root = '../results/crf_11_7/'
mask_root = '../../BaseModel/cod_test_dataset/'
test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
# json_file_path = './baseline_results.json'
json_file_path = './results.json'
outter_results = {}

if os.path.isfile(json_file_path):
    with open(json_file_path) as f:
        outter_results = json.load(f)

inner_results = {}
for dataset in test_datasets:
    gt_img_root = mask_root+dataset+"/GT/"
    predd_img_root =pred_root+dataset+"/"
    if len(os.listdir(predd_img_root))!= len(os.listdir(gt_img_root)):
        raise ValueError
    image_list = [path.split(".")[0] for path in os.listdir(predd_img_root)]
    for image_name in tqdm(image_list):
        image_jpg,image_png = image_name+".jpg",image_name+".png"
        pred_image_path,mask_image_path = predd_img_root+image_png,gt_img_root+image_png

        # mask = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)
        # pred = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)
        try:
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            MAE.step(pred=pred, gt=mask)
        except:
            #gt image size != rgb image size
            pass
    fm = FM.get_results()["fm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "meanFm": fm["curve"].mean(),
        "meanEm": em["curve"].mean(),
        "MAE": mae,
    }
    results['time'] =time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    ######### update results in json file modify
    current_key = predd_img_root.split('/')[-3]
    inner_results[dataset] = results
    ### visualize results
    table = []
    table_result_path = './visualize_results.txt'

    for key in inner_results.keys():
        values_list = list(inner_results[key].values())
        table.append([key]+values_list)

    headers = ["dataset","Smeasure","meanFm","meanEm","MAE","time"]
    my_table = tabulate(table, headers, tablefmt="fancy_grid")
    print(my_table)

outter_results_key = pred_root.split('/')[-2]
outter_results[outter_results_key] = inner_results
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(outter_results, f, ensure_ascii=False, indent=4)