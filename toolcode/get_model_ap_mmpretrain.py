#show_pkl.py
 
import pickle
import os
import json
import csv
from numpy import *
nowpath = r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/shufflenetv2_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3v2"
name = nowpath.split("/")[-1]
dir_list = os.listdir(nowpath)
nowcsv = {}
for cur_file in dir_list:
    newpath = os.path.join(nowpath, cur_file)
    if os.path.isfile(newpath) :
        if cur_file[:10] == "test_epoch":
            nowcount = int(cur_file.split(".")[0].split("test_epoch")[1])
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
            # alltsr = alltsr[-10]
            alltsr = alltsr[-1]
            nowcsv[nowcount] =  [cur_file, alltsr]
        else:
            continue  
sorted(nowcsv.keys())
writecsv = list(nowcsv.values())
with open(os.path.join(nowpath,name + "_" + "alltest.csv"), "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    writecsv.append(name)
    csv_writer.writerows(writecsv)
    f.close()
