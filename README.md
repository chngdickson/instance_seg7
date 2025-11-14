- Download weights from [link](https://github.com/RizwanMunawar/yolov7-segmentation/releases/download/yolov7-segmentation/yolov7-seg.pt) and store in "yolov7-segmentation" directory.

- Run the code with mentioned command below.
```bash
conda create -n 
```

- Output file will be created in the working directory with name `yolov7-segmentation/runs/predict-seg/exp/original-video-name.mp4`</b>


### Custom Data Labelling

- Once you will complete labelling, you can then export the data and follow mentioned steps below to start training.

### Custom Training

- Move your (segmentation custom labelled data) inside "yolov7-segmentation\data" folder by following mentioned structure.



![ss](https://user-images.githubusercontent.com/62513924/190388927-62a3ee84-bad8-4f59-806f-1185acdc8acb.png)



- Go to the <b>data</b> folder, create a file with name <b>custom.yaml</b> and paste the mentioned code below inside that.

```yaml
train: "path to train folder"
val: "path to validation folder"
# number of classes
nc: 1
# class names
names: [ 'car']
```

- Download weights from the <a href= "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt">link</a> and move to <b>yolov7-segmentation</b> folder.
- Go to the terminal, and run mentioned command below to start training.
```bash
python3 segment/train.py \
--data myConfigs/customData.yaml \
--batch-size 16 \
--weights "myConfigs/models/yolov7x-seg.pt" \
--epochs 300 \
--name yolov7-seg \
--img 640 \
--hyp myConfigs/hyp.scratch-high.yaml
```

```bash
cd Documents/dockers/yolov7-segmentation/
conda activate yolov7
python3 segment/train.py --data myConfigs/customData.yaml \
                          --batch-size 16 \
                          --weights "myConfigs/yolov7-seg-best.pt"
                          --cfg myConfigs/yolov7-seg.yaml \
                          --epochs 10 \
                          --name yolov7-seg \
                          --img 640 \
                          --hyp myConfigs/hyp.scratch-high.yaml
```

### Label New dataset
```bash
python3 detect.py --weights path/to/your/best.pt --source path/to/your/unlabeled_images --save-txt --conf 0.25 --iou 0.45
```

### Custom Model Detection Command
```bash
python3 segment/predict.py --weights "runs/yolov7-seg/exp/weights/best.pt" --source "videopath.mp4"
```



### References
- https://github.com/WongKinYiu/yolov7/tree/u7/seg
- https://github.com/ultralytics/yolov5


**Some of my articles/research papers | computer vision awesome resources for learning | How do I appear to the world? 🚀**

[Ultralytics YOLO11: Object Detection and Instance Segmentation🤯](https://muhammadrizwanmunawar.medium.com/ultralytics-yolo11-object-detection-and-instance-segmentation-88ef0239a811) ![Published Date](https://img.shields.io/badge/published_Date-2024--10--27-brightgreen)
For more details, you can reach out to me on [Medium](https://muhammadrizwanmunawar.medium.com/) or can connect with me on [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/)