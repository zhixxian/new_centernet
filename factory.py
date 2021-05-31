from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetDetector
from .sample.ctdet import CTDetDataset
from .dataset.coco import COCO

detector_factory = {
  'ctdet': CtdetDetector
}

train_factory = {
  'ctdet': CtdetTrainer
}

dataset_factory = {
  'coco': COCO
}

_sample_factory = {
  'ctdet': CTDetDataset # Object Detection
}

def get_dataset(dataset, task):
      class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset