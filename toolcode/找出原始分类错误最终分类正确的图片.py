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
# picdir = r"/root/lzz1/Diseases_lzz/weight/classresultshow_mmpretrain"
picdir = r"/root/lzz1/Diseases_lzz/weight/classresultshow_mmpretrain_val"
# org = [
#         "EVA-02_out_epoch90_resshow",
#         "RIFormerv2_out_epoch90_resshow",
#         "shufflenetv2_out_epoch100_resshow",
#         "XCiTv2_out_epoch80_resshow"
#        ]
# addmask = [
#         "EVA-02_add4CSASV_LR_pidnet_addmask_out_epoch70_resshow",
#         "RIFormer_add4CSASV_LR_pidnet_addmask_out_epoch100_resshow",
#         "shufflenetv2_add4CSASV_LR_pidnet_addmaskv2_epoch_70_resshow",
#         "XCiT_add4CSASV_LR_pidnet_addmask_epoch_70_resshow"
#        ]
# addmaskloss = [
#         "EVA-02_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3_out_epoch100_resshow",
#         "RIFormer_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3v2_out_epoch80_resshow",
#         "shufflenetv2_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3_epoch_60_resshow",
#         "XCiT_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3_epoch_100_resshow"
#        ]
org = [
        "EVA-02_outval_epoch_90_resshow",
        "RIFormerv2_outval_epoch_90_resshow",
        "shufflenetv2_outval_epoch_100_resshow",
        "XCiTv2_outval_epoch_80_resshow"
       ]
addmask = [
        "EVA-02_add4CSASV_LR_pidnet_addmask_outval_epoch_70_resshow",
        "RIFormer_add4CSASV_LR_pidnet_addmask_outval_epoch_100_resshow",
        "shufflenetv2_add4CSASV_LR_pidnet_addmaskv2_outval_epoch_70_resshow",
        "XCiT_add4CSASV_LR_pidnet_addmask_outval_epoch_70_resshow"
       ]
addmaskloss = [
        "EVA-02_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3_outval_epoch_100_resshow",
        "RIFormer_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3v2_outval_epoch_80_resshow",
        "shufflenetv2_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3_outval_epoch_60_resshow",
        "XCiT_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3_outval_epoch_100_resshow"
       ]
bestpic = []
betterpic = []
pathbetter = {}
for noworg,nowaddmask,nowaddmaskloss in zip(org,addmask,addmaskloss):
    print(noworg.split("_")[0])
    noworg = os.path.join(picdir,noworg)
    nowaddmask = os.path.join(picdir,nowaddmask)
    nowaddmaskloss = os.path.join(picdir,nowaddmaskloss)
    # 4C_4C
    # print("4C_4C")
    nowpiclist = []
    nowbetterpic = []
    dir_list = os.listdir(os.path.join(nowaddmaskloss,"4C_4C"))
    for nowpic in dir_list:
        if os.path.exists(os.path.join(noworg,"4C_SVSA",nowpic)) and os.path.exists(os.path.join(nowaddmask,"4C_SVSA",nowpic)):
            # print(nowpic)
            nowpiclist.append(nowpic)
        if os.path.exists(os.path.join(noworg,"4C_SVSA",nowpic)):
        #     print(nowpic)
            nowbetterpic.append(nowpic)
            pathbetter[nowpic] = "4C"
    # SVSA_SVSA
    # print("SVSA_SVSA")
    # dir_list = os.listdir(os.path.join(nowaddmaskloss,"SVSA_SVSA"))
    # for nowpic in dir_list:
    #     if os.path.exists(os.path.join(noworg,"SVSA_4C",nowpic)) and os.path.exists(os.path.join(nowaddmask,"SVSA_4C",nowpic)):
    #         # print(nowpic)
    #         nowpiclist.append(nowpic)
    #     if os.path.exists(os.path.join(noworg,"SVSA_4C",nowpic)):
    #     #     print(nowpic)
    #         nowbetterpic.append(nowpic)
    #         pathbetter[nowpic] = "SVSA"
    print(len(nowpiclist))
    bestpic.append(nowpiclist)
    betterpic.append(nowbetterpic)
# print(bestpic)
print(len(bestpic))
print(len(betterpic))

# resbest = list(set(bestpic[0]).intersection(bestpic[1],bestpic[2],bestpic[3])) # 求多个list的交集：a、b、c同时拥有的元素
resbest = list(set(bestpic[0]).intersection(bestpic[1],bestpic[2])) 
print(resbest)
print("#########################################")
# resbetter = list(set(betterpic[0]).intersection(betterpic[1],betterpic[2],betterpic[3])) # 求多个list的交集：a、b、c同时拥有的元素
# resbetter = list(set(betterpic[0]).intersection(betterpic[1],betterpic[2]))
resbetter = list(set(betterpic[0]).intersection(betterpic[2]))
print(resbetter)
print(len(resbetter))
rootpath = r"/root/lzz1/Diseases_lzz/data/classification/val"
savepath = r"/root/lzz1/Diseases_lzz/weight/classresultshow_mmpretrain_val/chooseshowpic0921_4c_val"
for i in resbetter:
    # i = i.replace(".png","")
    print(pathbetter[i])
    shutil.copy(os.path.join(rootpath,pathbetter[i],i.replace(".png","")),os.path.join(savepath,i.replace(".png","")))


