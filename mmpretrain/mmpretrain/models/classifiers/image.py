# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier
# lzz append
lzz_feat_flag = 1
if lzz_feat_flag == 1:
    from mmsegmentation.demo.image_demo_with_inferencer_getfeature import lzzgetmodel_frommmseg,lzzgetfeat_frommmseg

lzz_mask_flag = 0
if lzz_mask_flag == 1:
    from mmsegmentation.demo.image_demo_with_inferencer_getmask import lzzgetmodel_frommmseg,lzzgetfeat_frommmseg


lzz_maskSAM_flag = 0
if lzz_maskSAM_flag == 1:
    import os
    from PIL import Image
    import cv2
    import torch.nn.functional as F
    import concurrent.futures
    from torch.nn.functional import interpolate
@MODELS.register_module()
class ImageClassifier(BaseClassifier):
    """Image classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.
            - probs (List[float], optional): The probability of every batch
              augmentation methods. If None, choose evenly. Defaults to None.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super(ImageClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head
        # lzz append
        lzz_feat_flag = 1
        if lzz_feat_flag == 1:
            self.now_lzz_model = None
        # lzz append
    # lzz append
    def load_and_process_images(self,nowpath,device):
        piclist = os.listdir(nowpath)
        tensor_list = []

        for pic_name in piclist:
            pic_path = os.path.join(nowpath, pic_name)
            image = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
            image_tensor = torch.tensor(image, dtype=torch.float32,device=device)
            tensor_list.append(image_tensor)

        combined_tensor = torch.stack(tensor_list)
        return combined_tensor
    # lzz append
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs, data_samples=data_samples)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # lzzappend 
            lzzsaveflag = 0
            if lzzsaveflag == 1:
                from torchvision.utils import save_image
                import os
                savepath = r"/root/lzz1/Diseases_lzz/weight/mmpretrain/trainsave/shufflenetv2_add4CSASV_LR_pidnet_addmask_addnewloss_v2_changeW*3/ZXIsF3TeDi9BoUv"
                for nowindex,nowseg in enumerate(inputs[0]):
                    save_image(nowseg, os.path.join(savepath,str(nowindex)+".png"))
                save_image(inputs[0], os.path.join(savepath,"Now.png"))
            # lzzappend 
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, stage='neck', data_samples =None):
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.
                - "pre_logits": The feature before the final classification
                  linear layer. Usually returns a tensor.

                Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple and the output of
            pre_logits is a tensor.

        Examples:
            1. Backbone output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_classifier(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')
        # lzz append
        lzz_feat_flag = 1
        if lzz_feat_flag == 1:
            now_lzz_input = [i.img_path for i in data_samples]
            # now_lzz_input = ["/root/lzz1/Diseases_lzz/mmsegmentation/data/segdata_4CSASV_LR/leftImg8bit/ZXIsF3TeDi9BoUv/ZXIsF3TeDi9BoUv.png"]
            if self.now_lzz_model is None:
                self.now_lzz_model = lzzgetmodel_frommmseg(inputs.device)
            lzzsegfeat = lzzgetfeat_frommmseg(self.now_lzz_model,now_lzz_input)
            # lzzsegfeat = torch.stack(lzzsegfeat).squeeze()
            lzzsegfeat = torch.stack(lzzsegfeat).squeeze(1)
            # cam可视化只有一张图
            if lzzsegfeat.shape[0] != inputs.shape[0]:
                lzzsegfeat = lzzsegfeat.unsqueeze(0)
            # inputs = torch.cat((lzzsegfeat,inputs),1)
            #  单独用mask送进net
            inputs = lzzsegfeat
        # lzz append
        
         # lzz append
        lzz_maskSAM_flag = 0
        if lzz_maskSAM_flag == 1:
            orgpath = r"/data2/lyh/Diseases_lzz/data/classification_4C_CHD"
            # nowmaskpath = r"/data2/lyh/Diseases_lzz/data/segment-anything_premask/generate_mask_Byself_vit_h"
            # nowmaskpath = r"/data2/lyh/Diseases_lzz/data/segment-anything_premask/generate_mask_fromMask_vit_h"
            nowmaskpath = r"/data2/lyh/Diseases_lzz/data/segment-anything_premask/generate_mask_fromBOX_vit_h"
            # nowmaskpath = r"/data2/lyh/Diseases_lzz/data/segment-anything_premask/generate_mask_fromBGFG_vit_h"
            now_lzz_input = [i.img_path for i in data_samples]
            # allmaskpath = [nowmaskpath for i in data_samples]
            allmaskpath = [i.replace(orgpath,nowmaskpath).rsplit(".",1)[0] for i in now_lzz_input]
            # 我写的
            # lzzsegfeat = []
            # for nowpath in allmaskpath:
            #     piclist = os.listdir(nowpath)
            #     tensor_list = []
            #     # 读取图片并将其转换为二进制0和1的张量
            #     for pic_name in piclist:
            #         pic_path = os.path.join(nowpath, pic_name)
            #         image = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
            #         image_tensor = torch.tensor(image, dtype=torch.float32).to(inputs.device)  # 将图像转换为张量并移动到CUDA设备
            #         tensor_list.append(image_tensor)
            #     # 使用torch.cat将张量连接在一起
            #     combined_tensor = torch.stack(tensor_list)
            #     lzzsegfeat.append(combined_tensor)
            # lzzsegfeat = [F.interpolate(i.unsqueeze(0),size=[224, 224],mode='bilinear',align_corners=False) for i in lzzsegfeat]
            # lzzsegfeat = torch.stack(lzzsegfeat).squeeze()
            # 我写的

            # 使用多线程异步加载图像
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                lzzsegfeat = list(executor.map(self.load_and_process_images, allmaskpath, [inputs.device] * len(allmaskpath)))

            # 对图像进行插值
            lzzsegfeat = [interpolate(i.unsqueeze(0), size=[224, 224], mode='bilinear', align_corners=False) for i in lzzsegfeat]
            # lzzsegfeat = [interpolate(i.unsqueeze(0), size=[336, 336], mode='bilinear', align_corners=False) for i in lzzsegfeat]
            lzzsegfeat = torch.stack(lzzsegfeat).squeeze()
            # print(lzzsegfeat.size(),"     ",inputs.size())
            if lzzsegfeat.shape[0] != inputs.shape[0]:
                lzzsegfeat = lzzsegfeat.unsqueeze(0)
            inputs = torch.cat((lzzsegfeat,inputs),1)
        # lzz append
        x = self.backbone(inputs)
    
        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs, data_samples = data_samples)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs, data_samples = data_samples)
        return self.head.predict(feats, data_samples, **kwargs)

    def get_layer_depth(self, param_name: str):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            Tuple[int, int]: The layer-wise depth and the max depth.
        """
        if hasattr(self.backbone, 'get_layer_depth'):
            return self.backbone.get_layer_depth(param_name, 'backbone.')
        else:
            raise NotImplementedError(
                f"The babckone {type(self.backbone)} doesn't "
                'support `get_layer_depth` by now.')
