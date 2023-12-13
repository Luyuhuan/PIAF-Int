import os
import json
import copy
from posixpath import abspath
import cv2
import numpy as np

class_dict = {"脊柱":"SP", "肋骨":"RB", "左心房":"LA", "房间隔":"AS", "右心房":"RA", 
              "右心室":"RV", "左心室":"LV", "室间隔":"VS", "左室壁":"LVW", "右室壁":"RVW", 
              "降主动脉":"DA", "右肺":"RL", "左肺":"LL", "单心房":"SA", "单心室壁":"SVW", 
              "中间型心室":"IV", 
              "残余右心室":"RV", "右心室壁":"RVW","左心室壁":"LVW",}
class_dictvalues = ["SP", "RB", "LA", "AS", "RA", 
                    "RV", "LV", "VS", "LVW", "RVW", 
                    "DA", "RL", "LL", "SA", "SVW", 
                    "IV"]

Structure = {}
SectionCount = {}
allstr = []
allstr1 = {}
realname = {"心尖单心室单心房":"单心室单心房",
            "胸骨旁单心室单心房":"单心室单心房",
            "心底单心室单心房":"单心室单心房"}
def list_dir(file_dir):
    dir_list = os.listdir(file_dir)
    print("############################################################################################")
    print(file_dir)
    print("############################################################################################")
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path) and cur_file == "annotations.json":
            f = open(path, encoding='utf-8')
            frame = json.load(f)
            annotations = frame["annotations"]
            # 遍历所有图片 找出heart的包围盒
            for x in annotations:
                picpath = os.path.join(file_dir, x)
                if not os.path.exists(picpath.encode("gbk")):
                    print(picpath,' 不存在')
                    continue
                if 'annosets' in annotations[x]:
                    if "bodyPart" in annotations[x]['annosets'][0]:
                        cur_section = annotations[x]['annosets'][0]["bodyPart"]
                    elif "image_type" in annotations[x]['annosets'][0]:
                        cur_section = annotations[x]['annosets'][0]["image_type"]
                    else:
                        print("Now!!")
                    cur_standard = annotations[x]['annosets'][0]["standard"]
                    # cur_section = cur_section + cur_standard
                    if cur_section in realname:
                        cur_section = realname[cur_section]
                    if cur_section not in Structure:
                        Structure[cur_section] = {}
                        SectionCount[cur_section] = {}
                    if cur_standard not in Structure[cur_section]:
                        Structure[cur_section][cur_standard] = {}
                        SectionCount[cur_section][cur_standard] = 0
                    SectionCount[cur_section][cur_standard] += 1
                    for y in annotations[x]['annosets'][0]["annotations"]:
                        if y['name'] not in allstr1:
                            allstr1[y['name']] = 0
                        allstr1[y['name']] = allstr1[y['name']] + 1
                        if y["name"] not in allstr:
                            allstr.append(y["name"])
                        if y['name'] not in Structure[cur_section][cur_standard]:
                            Structure[cur_section][cur_standard][y['name']] = 0
                        Structure[cur_section][cur_standard][y['name']] += 1
                else:
                    print("annosets not exist!")
        if os.path.isdir(path):
            list_dir(path)

list_dir(r"/root/lzz1/Diseases_lzz/data/segmentation/4CSASV/train")
# list_dir(r"G:\四腔心疾病数据\正常分割数据\4C")
for cur_section in SectionCount:
    print("**********************")
    print(cur_section)
    print(SectionCount[cur_section])
indexcount = 0
for cur_section in Structure:
    # print("**********************")
    print(indexcount," ", cur_section)
    indexcount = indexcount + 1
    for cur_str in Structure[cur_section]:
        # print(cur_str, Structure[cur_section][cur_str])
        print(list(Structure[cur_section][cur_str].keys()))
        break
print(allstr)
allidlist = []
allstr1 = sorted(allstr1.items(),key = lambda x:x[1],reverse = True)
for i in allstr1:
    print(i)
    if i[0] not in class_dict:
        continue
    enid = class_dictvalues.index(class_dict[i[0]])
    nowidlist = [enid for i in range(i[1])]
    # print(nowidlist)
    allidlist = allidlist + nowidlist
# print(allidlist)
print(list(set(allidlist)))