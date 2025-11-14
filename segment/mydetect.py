import cv2
import json
from typing import Optional
import argparse
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import itertools
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.general import (LOGGER, NUM_THREADS, Profile, check_img_size, check_yaml,
                           non_max_suppression, print_args, increment_path,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import process_mask_upsample, scale_masks
from utils.torch_utils import de_parallel, smart_inference_mode
from utils import threaded

def init_coco_json():
    coco_format = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "TopViewTree",
            "contributor": "Dis",
            "url": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "name": "Public Domain"
            }
        ],
        "categories": [
            # {
            #     "id": 0,
            #     "name": "trunk",
            #     "supercategory": "none"
            # },
            # {
            #     "id": 1,
            #     "name": "crown",
            #     "supercategory": "none"
            # }
        ],
        "images": [
        ],
        "annotations": [
        ]
    }
    return coco_format

import supervision as sv
def bbox_area(bbox):
    """
    Calculate area of a bounding box in COCO format
    bbox: [x, y, width, height]
    """
    if len(bbox) != 4:
        raise ValueError("Bbox must have 4 elements: [x, y, width, height]")
    
    x, y, width, height = bbox
    area = width * height
    return area

def add_coco_image(coco_data, image_id, file_name, width, height):
    coco_data["images"].append({
        "id": image_id,
        "license":1,
        "file_name": file_name,
        "width": width,
        "height": height,
        "date_captured": datetime.now().isoformat()
    })
def add_coco_category(coco_data, class_map:dict):
    if not isinstance(class_map, dict):
        raise TypeError(f"class_map must be of type [Dict], Parsed [{type(class_map)}]")
    for i, (class_id, class_name) in enumerate(class_map.items()):
        cate_dict = {
            'id': int(class_id),
            'name':str(class_name),
            'supercategory': "none"
        }
        coco_data["categories"].append(cate_dict)
    
def check_masks_not_zero_numpy(masks):
    # np.all(mask) evaluates to True if all elements are non-zero
    non_zero_indcs = non_zero_indices_array = np.where(masks != 0)[0]
    print(f"Non, {non_zero_indcs}")

def add_coco_annotation(predn, coco_data, path, class_map, pred_masks):
    """
    Save one JSON result {id: int(N_labels), "image_id": int(img_id), "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    """
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}

    def single_encode(mask):
        list_of_polygons = sv.mask_to_polygons(mask)
        return list_of_polygons
        

    image_name = os.path.basename(path)
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    N,W,H = pred_masks.shape
    # rles = []
    # for mask in pred_masks:
    #     rles.append(single_encode(mask))
    image_id = len(coco_data["images"])
    add_coco_image(coco_data, image_id, path.stem, width=W,height=H)
    for i, (p, b, m) in enumerate(zip(predn.tolist(), box.tolist(), pred_masks)):
        if not np.any(m):
            continue
        poly = single_encode(m)
        if len(poly)<1:
            continue
        bbox = [int(round(x, 3)) for x in b]
        label_id = len(coco_data["annotations"])
        segmentation = list(itertools.chain.from_iterable(poly[0].tolist()))
        annotation = {
            "id": label_id,
            "image_id": image_id,
            'category_id': class_map[int(p[5])],
            'bbox': bbox,
            'score': round(p[4], 5),
            'segmentation': segmentation,
            "area": bbox_area(bbox),
            "iscrowd" : 0
        }
        coco_data["annotations"].append(annotation)
   
import yaml
def load_yaml(file_path):
    """Load and parse a YAML file"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

@smart_inference_mode()
def run(
        source,
        yaml_conf,
        weights,  # model.pt 
        save_dir:Path,
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        save_json=False,  # save a COCO-JSON results file
        save_overlay=False,
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None
):
    # Init
    # Load Yaml
    yaml_conf = load_yaml(yaml_conf)
    
    # Create_dirs based on Directory you've given
    save_dir = increment_path(save_dir, exist_ok=True, mkdir=True)
    masks_dir = increment_path(save_dir/'masks', exist_ok=True, mkdir=True)
    overlays_dir = increment_path(save_dir/'overlays', exist_ok=True, mkdir=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=yaml_conf, fp16=False)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    # batch_size = model.batch_size if engine else 1 
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    num_masks = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
    

    # Configure
    model.eval()
    nc = int(yaml_conf['nc'])  # number of classes
    niou = torch.linspace(0.5, 0.95, 10, device=device).numel()


    # Dataloader
    if pt :
        ncm = model.model.nc
        assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                            f'classes). Pass correct combination of --weights and --data that are trained together.'
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    dataloader = create_dataloader(
        source, imgsz, batch_size, stride, single_cls=False, pad=0.0, rect=True, 
        workers=int(os.cpu_count()), overlap_mask=False, mask_downsample_ratio=1
        )[0]

    
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    dt = Profile(), Profile(), Profile()
    
    
    #################################################################
    ########## CREATE COCO #####################
    names = model.names if hasattr(model, 'names') else model.module.names
    coco_json = init_coco_json()
    add_coco_category(coco_data=coco_json, class_map=names)
    #################################################################
    
    
    stats = []
    seen = 0
    pbar = tqdm(dataloader, desc='Processing')  # progress bar
    for batch_i, (im, targets, paths, shapes, _) in enumerate(pbar):
        with dt[0]:
            im = im.to(device, non_blocking=True).float()
            im /= 255  # 0 - 255 to 0.0 - 1.0

        # Inference
        with dt[1]:
            out, train_out = model(im) 


        # NMS
        with dt[2]:
            out = non_max_suppression(out,
                                      conf_thres,
                                      iou_thres,
                                      labels=[],
                                      multi_label=True,
                                      agnostic=False,
                                      max_det=max_det,
                                      nm=num_masks)

        
        for img_idx, pred in enumerate(out):
            seen += 1
            labels = targets[targets[:, 0] == img_idx, 1:]
            n_labels, n_preds = labels.shape[0], pred.shape[0]  # number of labels, predictions
            img_path, shape = Path(paths[img_idx]), shapes[img_idx][0]
            correct_masks = torch.zeros(n_preds, niou, dtype=torch.bool, device=device)  # init
            correct_bboxes = torch.zeros(n_preds, niou, dtype=torch.bool, device=device)  # init
            
            # Skip If No Preds detected
            if n_preds == 0:
                if n_labels:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            # Pred Masks
            pred_masks = process_mask_upsample(train_out[1][img_idx], pred[:, 6:], pred[:, :4], shape=im[img_idx].shape[1:])
            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            
            # Predictions
            predn = pred.clone()
            scale_coords(im[img_idx].shape[1:], predn[:, :4], shape, shapes[img_idx][1]) 

            if save_json:
                pred_masks_scaled = scale_masks(
                    im[img_idx].shape[1:],pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[img_idx][1]
                    )
                add_coco_annotation(predn, coco_json, img_path, list(range(len(names))), pred_masks_scaled)
            
            if save_overlay:
                pred_masks_scaled = scale_masks(
                        im[img_idx].shape[1:],pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[img_idx][1]
                        )
                curr_img = F.interpolate(im[img_idx].unsqueeze(0), shape, mode='bilinear').squeeze(0)
                process_and_save_overlay(img_path, curr_img, pred_masks_scaled, overlays_dir)
            

            desired_classes = ['fullcrown',"trunk"]
            cls_idxes = [cls_idx for cls_idx, val in names.items() if val in desired_classes]

            pred_masks_scaled = scale_masks(
                im[img_idx].shape[1:],pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[img_idx][1])
            filter_masks_by_labels(img_path, masks_dir, pred_masks_scaled, predn, cls_idxes)

    # Save JSON
    if save_json:
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        pred_json_all = str(save_dir / f"{w}_predictions.json")
        with open(pred_json_all,"w")as f:
            json.dump(coco_json,f)

@threaded
def process_and_save_overlay(img_path, im, pred_masks, save_pth):
    image_name = Path(img_path).stem
    img_np = (im.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_np_overlay = img_np.copy()
    
    for i in range(pred_masks.shape[2]):
        mask = pred_masks[:, :, i]
        if np.any(mask > 0):  # If mask has detected areas
            # Generate random color for this instance
            color = np.random.randint(0, 255, 3).tolist()
            
            # Apply colored mask
            colored_mask = np.zeros_like(img_np_overlay)
            colored_mask[mask > 0] = color
            
            # Blend with original image
            cv2.addWeighted(img_np_overlay, 1.0, colored_mask, 0.5, 0, img_np_overlay)

    cv2.imwrite(f"{save_pth}/{image_name}_overlay.jpg", img_np_overlay)
    return

@threaded
def filter_masks_by_labels(img_path, save_pth, pred_masks, predn, keep_labels):
    # predn [x1, y1, x2, y2, conf, cls]
    H, W = pred_masks.shape[:2]
    device = predn.device
    
    # Filter by Cls id
    cls_ids = predn[:,5]
    keep_labels_tensor = torch.tensor(keep_labels, device=device)
    bool_filter_by_cls = torch.isin(cls_ids, keep_labels_tensor)
    
    # Filter by center distance
    center = torch.tensor([W/2, H/2], device=device)
    tolerance = torch.linalg.vector_norm(torch.tensor([W, H], device=device) * 0.05)
    bbox_centers = (predn[:, :2] + predn[:, 2:4]) / 2
    dist_from_center = torch.linalg.vector_norm((bbox_centers-center), dim=1)
    bool_filter_by_dist = (dist_from_center < tolerance)

    # Filter Masks by combining the bools
    keep_masks = (bool_filter_by_cls & bool_filter_by_dist).contiguous().cpu().numpy()
    filtered_masks = pred_masks[:, :, keep_masks]
    
    if filtered_masks is None or filtered_masks.shape[2] == 0:
        print("Did this get called?")
        return False
    
    binary_mask = np.any(filtered_masks > 0, axis=2).astype(np.uint8) * 255
    image_name = Path(img_path).stem
    cv2.imwrite(f"{save_pth}/{image_name}_mask.jpg", binary_mask)
    return True


if __name__ == "__main__":
    source_imgs = "/home/ds1804/Documents/dockers/yolov7-segmentation/data/test"
    yaml_config = "/home/ds1804/Documents/dockers/yolov7-segmentation/myConfigs/customData.yaml"
    save_dir = Path("/home/ds1804/Documents/dockers/yolov7-segmentation/runs/inference_out")
    weights = "/home/ds1804/Documents/dockers/yolov7-segmentation/runs/train-seg/yolov7-seg9/weights/best.pt"
    run(source_imgs, yaml_config, weights=weights, save_dir=save_dir)
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='dataset.yaml path')
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
#     parser.add_argument('--batch-size', type=int, default=32, help='batch size')
#     parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
#     parser.add_argument('--task', default='val', help='train, val, test, speed or study')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
#     parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--verbose', action='store_true', help='report mAP by class')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
#     parser.add_argument('--project', default=ROOT / 'runs/val-seg', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     opt.data = check_yaml(opt.data)  # check YAML
#     # opt.save_json |= opt.data.endswith('coco.yaml')
#     opt.save_txt |= opt.save_hybrid
#     print_args(vars(opt))
#     return opt


# def main(opt):
#     run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
