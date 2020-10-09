from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import torch
import cv2
import os

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


class VOCAnnotationTransform:
    def __init__(self):
        self.label_map = COCO_LABEL_MAP

    def __call__(self, target, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if "bbox" in obj:
                bbox = obj["bbox"]
                # 获得类别id
                label_idx = self.label_map[obj["category_id"]] - 1
                # 坐标归一化
                print("scale_before", bbox)
                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                print(final_box)
                res += [final_box]
            else:
                print("No bbox found for object", obj)
        return res


class COCODetection(Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=VOCAnnotationTransform):
        self.root = root  # 数据集根目录
        self.coco = COCO(ann_path)  # COCO类对象
        self.ids = list(self.coco.imgToAnns.keys())  # 获取所有的图片id号（包含标注信息）
        self.transform = transform  # 数据增强
        self.target_transform = target_transform  # 数据变换

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        im, gt, _, _ = self.pull_item(index)  # 返回图像及其对应的标注信息
        return im, gt

    def pull_item(self, index):
        img_id = self.ids[index]  # 对应索引的图片id
        ann_ids = self.coco.getAnnIds(imgIds=img_id)  # 基于图片id获得标注id
        target = self.coco.loadAnns(ann_ids)  # 加载标注信息
        # 对应id的图片名称
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        file_path = os.path.join(self.root, file_name)  # 图片具体路径
        # 读取图片
        img = cv2.imread(file_path)
        height, width, _ = img.shape

        if self.target_transform is not None and len(target):
            target = self.target_transform()(target, width, height)

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


if __name__ == "__main__":
    # 获取COCODetection对象
    datasets = COCODetection(root="../COCO2017/val2017",
                             ann_path="../COCO2017/annotations/instances_val2017.json")
    # 获取图片
    im = datasets[0][0].permute(1, 2, 0).numpy()
    # 图片信息
    w, h, _ = im.shape
    scale = np.array([h, w, h, w])
    # 标注框
    boxes = datasets[0][1]
    for i in range(len(boxes)):
        # 类别
        label = COCO_CLASSES[boxes[i][-1]]
        # 边界框并转换
        box = np.array(boxes[i][:4] * scale, np.int)
        print(box)
        # 显示内容
        cv2.putText(im, label, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=1)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=1)
    cv2.imshow("image", im)
    cv2.waitKey(0)
