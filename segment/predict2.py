import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

#..... Tracker modules......
import skimage
from .sort_count import *
import numpy as np
#...........................


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode


class Infer_seg():
    def __init__(self, weights="yolov5s-seg.pt", imgz=(640,640)):
        # Load model
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, fp16=False)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgz, s=stride)
        
        self.model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup
        seen, windows, self.dt = 0, [], (Profile(), Profile(), Profile())
    
    def forward(self, im, conf_thres=0.25, iou_thres=0.45):
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.device).transpose(2,0).unsqueeze(0)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None] 
        with self.dt[1]:
            pred, out = self.model(im, visualize=False)
            proto = out[1]
        with self.dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=100, nm=32)
        
        det = pred[0]  # per image
        gn = torch.tensor(im.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        masks_in, bboxes = det[:, 6:], det[:, :4]
        # x1,y1,x2,y2,conf,detclass = det[:, :6]
        
        detection_bbox = det[:, :6]
        if len(det):
            masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

            # # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im.shape).round()

            # Mask plotting ----------------------------------------------------------------------------------------
            mcolors = [colors(int(cls), True) for cls in det[:, 5]]
            im_masks = plot_masks(im[0], masks, mcolors)  # image with masks shape(imh,imw,3)
        
            return detection_bbox, masks, len(det)
        else:
            return 0, 0, 0
    
    def im_mask_from_cls(self, detection_bbox,im_mask, cls=1):
        x1,y1,x2,y2,conf,detclass = detection_bbox[:,:6].T
        class_indices = torch.where(detclass == cls)[0]
        cls_mask = torch.clamp(im_mask[class_indices].sum(dim=0), 0,1)
        return cls_mask.detach().cpu().numpy()
    
    def im_mask_from_center_region(self, detection_bbox, im_mask, cls=1, center_tol=200):
        """
        Computes a mask for detections of class `cls` within `center_tol` pixels of the image center.
        
        Args:
            detection_bbox (Tensor): [N, 6] (x1, y1, x2, y2, conf, class)
            im_mask (Tensor): [N, H, W] (binary masks per detection)
            cls (int): Class ID to filter
            center_tol (int): Pixel tolerance around center
        
        Returns:
            Tensor: [H, W] binary mask (summed & clamped to [0, 1])
        """
        # Extract bbox data [N, 6] -> (x1, y1, x2, y2, conf, class)
        x1, y1, x2, y2, conf, detclass = detection_bbox[:, :6].T  # shapes: [N]
        H, W = im_mask.shape[1], im_mask.shape[2]
        
        # Compute center bounds (fixed region)
        center_x, center_y = W // 2, H // 2
        x_min, x_max = center_x - center_tol, center_x + center_tol
        y_min, y_max = center_y - center_tol, center_y + center_tol
        
        # Find detections of class `cls` whose centers are inside the central region
        box_centers_x = (x1 + x2) / 2  # [N]
        box_centers_y = (y1 + y2) / 2  # [N]
        
        is_cls = (detclass == cls)  # [N]
        is_in_center = (
            (box_centers_x >= x_min) & 
            (box_centers_x <= x_max) & 
            (box_centers_y >= y_min) & 
            (box_centers_y <= y_max))
        
        valid_indices = torch.where(is_cls & is_in_center)[0]
        
        if len(valid_indices) == 0:
            return torch.zeros_like(im_mask[0]), 0 , (0,0) # No matches
        elif len(valid_indices) > 1:
            valid_indices = valid_indices[torch.argmax(conf[valid_indices])].unsqueeze(0)
        else: # Valid indices is only 1
            pass
        # Sum all valid masks and clamp
        cls_mask = torch.clamp(im_mask[valid_indices].sum(dim=0), 0, 1)
        uv_center = (int(box_centers_x[valid_indices][0].detach().cpu().numpy().round()), int(box_centers_y[valid_indices][0].detach().cpu().numpy().round()))
        return cls_mask.detach().cpu().numpy(), len(valid_indices), \
            uv_center
