# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        # lzzFGBG
        # classes=("BG","FG"),
        # palette=[[128, 64, 128], [244, 35, 232]])
        # lzzFGBG
        
        classes=("BG", "SP", "RB", "LA", "AS", "RA", 
                "RV", "LV", "VS", "LVW", "RVW", 
                "DA", "RL", "LL", "SA", "SVW", 
                "IV"),
        palette=[[119, 11, 32], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100]]
        )

    def __init__(self,
                #  img_suffix='_leftImg8bit.png',
                #  seg_map_suffix='_gtFine_labelTrainIds.png',
                 img_suffix='',
                 seg_map_suffix='',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
