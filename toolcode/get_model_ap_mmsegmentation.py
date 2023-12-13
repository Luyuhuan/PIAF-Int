#show_pkl.py
 
import pickle
import os
import json
import csv
from numpy import *
nowpath = r"/root/lzz1/Diseases_lzz/weight/mmsegmentation_diffcount/PIDNet_LR_50_SASV_50_addnewloss_changeW*3"
name = nowpath.split("/")[-1]
dir_list = os.listdir(nowpath)
nowcsv = {}
for cur_file in dir_list:
    newpath = os.path.join(nowpath, cur_file)
    if os.path.isfile(newpath) :
        if cur_file == "test.log":
            # nowcount = int(cur_file.split(".")[0].split("test_epoch")[1])
            nowcount = len(nowcsv)
            alltsr = []
            # Open file 
            fileHandler  =  open  (newpath, "r", encoding="utf-8")
            while  True:
                # Get next line from file
                line  =  fileHandler.readline()
                # If line is empty then end of file reached
                if  not  line  :
                    break;
                # print(line.strip())
                alltsr.append(line.strip())
                # Close Close    
            fileHandler.close()
            print(len(alltsr))
            allneed = {}
            #allstr 
            for i in range(-19,-2):
            # for i in range(-6,-2):
                nowstr = alltsr[i].split("|")
                nowstr = [x.strip() for x in nowstr if x.strip()!='']
                print(nowstr)
                allneed[nowstr[0]] = {"IoU": nowstr[1],"Acc": nowstr[2]}
            nowcsvlist = list(allneed.keys())
            # allmean
            allmean = alltsr[-1]
            allmean = allmean.split(" ")
            allmean = [x.strip() for x in allmean if x.strip()!='']
            mIoU_index = allmean.index("mIoU:")
            mAcc_index = allmean.index("mAcc:")
            allneed["All_mean"] = {"IoU": allmean[mIoU_index + 1],"Acc": allmean[mAcc_index + 1]}
            nowcsvlist = ["All_mean"] + nowcsvlist
            nowcsviou = ["IoU"] + [allneed[j]["IoU"] for j in nowcsvlist]
            nowcsvacc = ["Acc"] + [allneed[j]["Acc"] for j in nowcsvlist]
            nowcsvlist = [" "] + nowcsvlist
            nowcsv[nowcount] =  [nowcsvlist, nowcsviou, nowcsvacc]
        else:
            continue  

with open(os.path.join(nowpath,name + "_" + "test.csv"), "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(nowcsv[0])
    f.close()
