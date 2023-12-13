import os
import shutil
import time
import os
import json
import copy
import cv2
import numpy as np
import shutil
import random
from mmpretrain.structures import DataSample
import mmengine
# 4CSASV
pkllist1 = [
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/EVA-02/out_epoch90.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/EVA-02_add4CSASV_LR_pidnet_addmask/out_epoch70.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/EVA-02_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3/out_epoch100.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/RIFormerv2/out_epoch90.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/RIFormer_add4CSASV_LR_pidnet_addmask/out_epoch100.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/RIFormer_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3v2/out_epoch80.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/shufflenetv2/out_epoch100.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/shufflenetv2_add4CSASV_LR_pidnet_addmaskv2/epoch_70.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/shufflenetv2_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3/epoch_60.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/XCiTv2/out_epoch80.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/XCiT_add4CSASV_LR_pidnet_addmask/epoch_70.pkl",
                r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/XCiT_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3/epoch_100.pkl"
            ]
# picdir = r"/root/lzz1/Diseases_lzz/weight/classresultshow_mmpretrain/chooseshowpic0921"
# picdir = r"/root/lzz1/Diseases_lzz/weight/chooseforvisio_CHD"
picdir = r"/root/lzz1/Diseases_lzz/weight/chooseforvisio_CHD_4pic"
import csv

with open(picdir + "_resCHD.csv","w",encoding="utf-8",newline="") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow(["method","4C","CHD"])
    piclist = os.listdir(picdir)
    for nowpic in piclist:
        print(nowpic)
        writer.writerow([nowpic,"",""])
        gtflag = 0
        for nowpkl in pkllist1:
            print(nowpkl)
            nowmethod = nowpkl.split("/")[-2]
            for result in mmengine.load(nowpkl):
                nowname = result['img_path'].split("/")[-1]
                if nowname == nowpic:
                    # print("标签：",result['gt_label'])
                    if gtflag == 0:
                        if int(result['gt_label']) == 0:
                            writer.writerow(["GT",1,0])
                        else:
                            writer.writerow(["GT",0,1])
                        gtflag = 1 
                    writer.writerow([nowmethod,float(result['pred_score'][0]),float(result['pred_score'][1])])
                    print("预测：",result['pred_score'])

# for nowpkl in pkllist1:
#     print(nowpkl)
#     for result in mmengine.load(nowpkl):
#         nowname = result['img_path'].split("/")[-1]
#         if nowname in piclist:
#             print("nowname:",nowname)
#             # print("标签：",result['gt_score'])
#             print("标签：",result['gt_label'])
#             print("预测：",result['pred_score'])