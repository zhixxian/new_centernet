from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.pictor_ctdet import PICTOR_CTDetDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.pictor import PICTOR


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'pictor' : PICTOR
}

_sample_factory = {
  'exdet': EXDetDataset, # ExtremeNet
  'ctdet': CTDetDataset, # Object Detection
  'ddd': DddDataset, # 3D Bounding Box Detection
  'multi_pose': MultiPoseDataset, # Multi-Person Human Pose Estimation
  'pictor_ctdet' : PICTOR_CTDetDataset 
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
