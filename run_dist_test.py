import os
import shutil
import time
ptname = [
          "epoch_10.pth","epoch_20.pth","epoch_30.pth","epoch_40.pth","epoch_50.pth",
          "epoch_60.pth","epoch_70.pth","epoch_80.pth", "epoch_90.pth","epoch_100.pth"]

ptpath = r"/data2/lyh/Diseases_lzz/weight/mmpretrain_feature_SKEM_withoutloss/shufflenetv2_8gpuv2"
# configfile = r"/data2/lyh/Diseases_lzz/code/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py"
# configfile = r"/data2/lyh/Diseases_lzz/code/mmpretrain/configs/riformer/riformer-s12_8xb128_in1k.py"
configfile = r"/data2/lyh/Diseases_lzz/code/mmpretrain/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py"
# configfile  =r"/data2/lyh/Diseases_lzz/code/mmpretrain/configs/xcit/xcit-nano-12-p16_8xb128_in1k.py"
for nowpt in ptname:
    nownum = nowpt.split("_")[1].split(".")[0]
    nowpt = os.path.join(ptpath, nowpt)
    nowpkl = os.path.join(ptpath, "out_epoch" + nownum + ".pkl")
    nowlog = os.path.join(ptpath, "test_epoch" + nownum + ".log")
    os.system(f'CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./dist_test.sh {configfile} {nowpt} 4 --out {nowpkl} >{nowlog} 2>&1 &')
    time.sleep(50) 
