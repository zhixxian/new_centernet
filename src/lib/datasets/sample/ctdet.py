from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
# from utils.image import flip, color_aug
# from utils.image import get_affine_transform, affine_transform
# from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
# from utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset): 
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox
    
  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

    # anns = detection, segmentation 등 태스크를 위해 bbox 좌표, segmentation mask 픽셀 등 필요한 정보들
    # num_objs = the number of anns, the maximum number of 128 
  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs) # the maximum number of 128 

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1] # 0 -> 행 / 1 -> 열

    # c -> calculate center point
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res: # validation중에 원본해상도 유지 = store_true
      input_h = (height | self.opt.pad) + 1 #opt.pad = 127
      input_w = (width | self.opt.pad) + 1 # height와 width의 값에 따라 다르지만 10으로 지정했을 때 128이 나옴
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w #input size 결정
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        # border값은 이미지 크기가 256을 초과하면 border이 128, 아니면 64
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        # zoom, pan(확대, 축소) 및 이동에 대한 요인에 따라 중심점위치와 side 결정
        sf = self.opt.scale # shift augmentation
        cf = self.opt.shift # scale augmentation
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip: # horizontal flip
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        
    # 파라미터 결정 후, input 값에 대해 affine transformation 수행
    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    # 데이터 향상
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    # down_ratio = 4
    # affine transformation을 위해 output 준비
    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes # num class = 80
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    # hm = heatmap
    # wh = target length and width
    # dense_wh = return directly to the long and wide map
    # reg = 다운샘플링한 오프셋값
    # ind = the position of the center point (h * W + w), one-dimensional representation
    # reg_mask = wheter there are key points under the fixed-length representation, up to 128 targets

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32) #[C, H, W]
    wh = np.zeros((self.max_objs, 2), dtype=np.float32) # [128, 2]
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32) # [2, H, W]
    reg = np.zeros((self.max_objs, 2), dtype=np.float32) # [128, 2]]
    ind = np.zeros((self.max_objs), dtype=np.int64) # [128]
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8) # [128]
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32) # [128, C*2]
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8) # [128, C*2]
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    # dataset 준비
    gt_det = []
    for k in range(num_objs): # 타겟이 얾마나 있는지에 따라 달라짐
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      # 출력 affine transformation                                                                             
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        # heat map의 radius(반경)이 타겟의 크기에 따라 결정
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius) # heat map과 ct_int, radius를 gaussian으로 그리기
        # 개체 수가 128개가 안되면, 나머지 위치는 0
        wh[k] = 1. * w, 1. * h # h = [128, 2]
        ind[k] = ct_int[1] * output_w + ct_int[0] # center point는 1차원 h*W+w
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k] # 클래스 사이의 length, width는 공유되지않음
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret