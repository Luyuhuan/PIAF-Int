#show_pkl.py
 
import pickle
import os
import json
import csv
from numpy import *
nowpath = r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/shufflenetv2_add4CSASV_LR_pidnet_addmask/20230616_152115/vis_data"
name = nowpath.split("/")[-1]
dir_list = os.listdir(nowpath)
nowcsv = []
for cur_file in dir_list:
    newpath = os.path.join(nowpath, cur_file)
    if os.path.isfile(newpath) :
        if cur_file == "scalars.json":
            f = open(newpath, "r", encoding='utf-8')
            content = f.read()
            allcontent = content.split("\n")
            allcontent = list(filter(None, allcontent))
            for i in allcontent:
                nowi = json.loads(i)
                if "accuracy/top1" in nowi:
                    print(nowi)
                    nowcsv.append(["accuracy/top1", nowi["accuracy/top1"], "data_time", nowi["data_time"], "time", nowi["time"], "step", nowi["step"]])
            # allcontent = [json.loads(i) for i in allcontent]
            # 遍历所有图片 找出heart的包围盒
        else:
            continue 
# sorted(nowcsv.keys())
# writecsv = list(nowcsv.values())
with open(os.path.join(nowpath,name + "_" + "allval.csv"), "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    # writecsv.append(name)
    csv_writer.writerows(nowcsv)
    f.close()
