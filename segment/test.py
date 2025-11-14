import json
from datetime import datetime

def create_coco_json():
    coco_format = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "Your dataset description",
            "contributor": "Your name",
            "url": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Your License",
                "url": ""
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "your_class_name",
                "supercategory": "none"
            }
            # Add more categories as needed
        ],
        "images": [
            # Will be populated with image info
        ],
        "annotations": [
            # Will be populated with annotations
        ]
    }
    return coco_format
def add_image(coco_data, image_id, file_name, width, height):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "license": 1,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": ""
    }
    coco_data["images"].append(image_info)

def add_annotation(coco_data, annotation_id, image_id, category_id, bbox, segmentation, area):
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,  # [[x1, y1, x2, y2, ...]]
        "area": area,
        "bbox": bbox,  # [x, y, width, height]
        "iscrowd": 0
    }
    coco_data["annotations"].append(annotation)
    
def create_sample_coco_json():
    coco = create_coco_json()
    
    # Add images
    add_image(coco, 1, "image1.jpg", 640, 480)
    add_image(coco, 2, "image2.jpg", 800, 600)
    
    # Add annotations
    add_annotation(coco, 1, 1, 1, [100, 100, 50, 50], [[100, 100, 150, 100, 150, 150, 100, 150]], 2500)
    add_annotation(coco, 2, 1, 1, [200, 200, 60, 40], [[200, 200, 260, 200, 260, 240, 200, 240]], 2400)
    
    # Save to file
    with open("coco_annotations.json", "w") as f:
        json.dump(coco, f, indent=2)
    
    return coco