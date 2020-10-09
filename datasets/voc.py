import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform:
    def __init__(self, class_to_ind=None, keep_difficult=False):
        # 将符号类别转换为数字类别
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []  # 用于存放结果
        for obj in target.iter('object'):  # 遍历图像中的目标
            difficult = int(obj.find('difficult').text) == 1
            if self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()  # 类别
            bbox = obj.find('bndbox')  # 边界框
            pts = ['xmin', 'ymin', 'xmax', 'ymax']  # 边界框坐标
            bndbox = []  # 存放边界框的坐标及对应类别
            for i, pt in enumerate(pts):  # 坐标标准化
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)  # 添加结果
            res += [bndbox]
        return res  # [[xmin,ymin,xmax,ymax,label_ind],...]


class VOCDetection(Dataset):
    def __init__(self, root, image_sets=None, transform=None,
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        if image_sets is None:
            image_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        self.root = root  # 数据集根目录
        self.transform = transform  # 数据变换
        self.target_transform = target_transform  # 数据增强
        self.name = dataset_name  # 数据集名称

        self.annopath = os.path.join('%s', 'Annotations', '%s.xml')  # 标注文件路径
        self.imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')  # 图像文件路径

        self.ids = list()
        for (year, name) in image_sets:
            root_path = os.path.join(self.root, 'VOC' + year)
            # self.ids=[['/VOC2007','000012'],['/VOC2007','000017'],...]
            for line in open(os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((root_path, line.strip()))

    def __getitem__(self, index):
        im, gt, _, _ = self.pull_item(index)  # 返回图像及其对应的标注信息
        return im, gt

    def __len__(self):
        return len(self.ids)  # 返回数据集数目

    def pull_item(self, index):
        img_id = self.ids[index]  # 根据index取对应元素
        target = ET.parse(self.annopath % img_id).getroot()
        img = cv2.imread(self.imgpath % img_id)

        height, width, _ = img.shape

        if self.target_transform is not None:  # 数据增强
            target = self.target_transform(target, width, height)

        if self.transform is not None:  # 数据变换
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]  # 通道变换为RGB
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        # 根据索引返回对应图像的OpenCV格式
        img_id = self.ids[index]
        return cv2.imread(self.imgpath % img_id, cv2.IMREAD_COLOR)


if __name__ == "__main__":
    # 获取VOCDetection对象
    datasets = VOCDetection(root="../VOCdevkit")
    # 获取图片
    im = datasets[0][0].permute(1, 2, 0).numpy()
    # 图片信息
    w, h, _ = im.shape
    scale = np.array([h, w, h, w])
    # 标注框
    boxes = datasets[0][1]
    for i in range(len(boxes)):
        # 类别
        label = VOC_CLASSES[boxes[i][-1]]
        # 边界框并转换
        box = np.array(boxes[i][:4] * scale, np.int)
        # 显示内容
        cv2.putText(im, label, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=1)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=1)
    cv2.imshow("image", im)
    cv2.waitKey(0)
