# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import MMSegInferencer
import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image
def lzzgetmodel_frommmseg(nowcuda):
    nowmodel = "/data2/lyh/Diseases_lzz/code/mmsegmentation/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py"
    # nowcheckpoint = "/data2/lyh/Diseases_lzz/weight/mmsegmentation/PIDNet_LR_addnewloss_v2_changeW*3/iter_17500.pth"
    nowcheckpoint = r"/data2/lyh/Diseases_lzz/weight/mmsegmentation/PIDNet_LR/iter_17500.pth"
    # nowmodel = "/data2/lyh/Diseases_lzz/code/mmsegmentation/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py"
    # nowcheckpoint = "/data2/lyh/Diseases_lzz/weight_CAMUS/mmsegmentation/PIDNet_LR/iter_6250.pth"
    nowdataset_name = "cityscapes"
    inferencer = MMSegInferencer(nowmodel,nowcheckpoint,nowdataset_name,device=nowcuda)
    return inferencer

def lzzgetfeat_frommmseg(inferencer,inputs):
    nowfeat = {}
    nowshowdir = r"/data2/lyh/Diseases_lzz/weight/mmsegmentation/PIDNet_LR/vis_data"
    lzz_feat = inferencer(inputs, show=False, out_dir=nowshowdir, opacity=0.5)["feature"]
    lzz_feat = [F.interpolate(i.unsqueeze(0),size=[224, 224],mode='bilinear',align_corners=False) for i in lzz_feat]
    # lzz_feat = [F.interpolate(i.unsqueeze(0),size=[336, 336],mode='bilinear',align_corners=False) for i in lzz_feat]
    #  保存特征图像可视化
    # lzzappend
    lzz_saveflag = 0
    if lzz_saveflag == 1:
        savepath = r"/root/lzz1/Diseases_lzz/Mask_4C_SVSA/lzz_feat0822"
        for nowpic,nowfeat in zip(inputs,lzz_feat):
            nowpic = nowpic.split("/")[-1].split(".")[0]
            newsavepath = os.path.join(savepath, nowpic)
            os.makedirs(newsavepath, exist_ok=True)
            count = 0
            for noweveryfeat in nowfeat[0]:
                save_image(noweveryfeat,  os.path.join(newsavepath, str(count)+ ".png"))
                count = count + 1
    # lzzappend
    return lzz_feat
    # for i,j in zip(inputs, lzz_feat):
    #     nowfeat[i] = j
    # return nowfeat

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('model', help='Config file')
    # parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    # parser.add_argument(
    #     '--out-dir', default='', help='Path to save result file')
    
    # parser.add_argument('--img', default = "/root/lzz1/Diseases_lzz/data_CAMUS/segmentation/leftImg8bit/test", help='Image file')
    # parser.add_argument('--img', default = "/root/lzz1/Diseases_lzz/data/classification/test/SVSA", help='Image file')
    # parser.add_argument('--img', default = "/root/lzz1/Diseases_lzz/mmsegmentation/data/segdata_4CSASV_LR/leftImg8bit/test", help='Image file')
    # parser.add_argument('--img', default = "/root/lzz1/Diseases_lzz/mmsegmentation/data/segdata_4CSASV_LR/leftImg8bit/val", help='Image file')
    parser.add_argument('--img', default = "/root/lzz1/Diseases_lzz/mmsegmentation/data/segdata_4CSASV_LR/leftImg8bit/valpic", help='Image file')
    parser.add_argument('--model', default= "/root/lzz1/Diseases_lzz/mmsegmentation/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py",help='Config file')
    # parser.add_argument('--checkpoint', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR_addnewloss_v2_changeW*3/iter_17500.pth", help='Checkpoint file')
    parser.add_argument('--checkpoint', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR/iter_17500.pth", help='Checkpoint file')
    # parser.add_argument('--out-dir', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR_addnewloss_v2_changeW*3/testshowoutv0825", help='Path to output file')
    # parser.add_argument('--out-dir', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR_addnewloss_v2_changeW*3/valshowoutv0825", help='Path to output file')
    # parser.add_argument('--out-dir', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR_addnewloss_v2_changeW*3/trainshowoutv0826/500_600", help='Path to output file')
    # parser.add_argument('--out-dir', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR/testshowoutv0825", help='Path to output file')
    # parser.add_argument('--out-dir', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR/valshowoutv0825", help='Path to output file')
    parser.add_argument('--out-dir', default="/root/lzz1/Diseases_lzz/weight/mmsegmentation/PIDNet_LR/showout/valshowoutv0825", help='Path to output file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to display the drawn image.')
    parser.add_argument(
        '--dataset-name',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    mmseg_inferencer = MMSegInferencer(
        args.model,
        args.checkpoint,
        dataset_name=args.dataset_name,
        device=args.device)

    # test a single image
    lzzfeat = mmseg_inferencer(
        args.img, show=args.show, out_dir=args.out_dir, opacity=args.opacity, batch_size=1)
    print(lzzfeat)
if __name__ == '__main__':
    main()
