"""
    继承torch.utils.data的Dataset类
    重写父类__len__()  __getitem__()方法
    __getitem__(index) 返回图片、标注信息(真实框位置以及类别)
    image:PIL对象   
    targets:dict(tensor) 
"""

import torch
import json
import os
from lxml import etree
from PIL import Image
from torch.utils.data import Dataset
from transforms import Compose, To_tensor, RandomHorizontalSlip

class my_dataset(Dataset):
    # data_path: ImageSets/main/train.txt  xml_path: /Annotations  json_path: /my_classes.json  image_path: JPEGImages
    def __init__(self, data_path, xml_path, json_path, image_path, transforms=None):
        super(my_dataset, self).__init__()
        self.xml_list = []
        with open(data_path, 'r') as o:
            list = [i.strip() for i in o.readlines()]
        for i in list:
            with open(os.path.join(xml_path, i + ".xml"), 'r') as f:
                # string ->  <annotation> element 
                xml_ele = etree.fromstring(f.read())
                # 遍历element对象, 返回字典
                data = self.xml_to_dict(xml_ele)["annotation"]
                # 筛选没有目标信息的图片
                if "object" not in data:
                    continue
                self.xml_list.append(os.path.join(xml_path, i + ".xml"))
                with open(json_path, 'r') as w:
                    self.class_dict = json.load(w)
                self.transforms = transforms
                self.image_path = image_path

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, index):
        with open(self.xml_list[index], 'r') as f:
            xml_ele = etree.fromstring(f.read())
            data = self.xml_to_dict(xml_ele)["annotation"]
            # 获取图片信息、标注信息、类别信息
            image = Image.open(os.path.join(self.image_path, data["filename"]))
            # 标签
            labels = []
            # 先验框信息
            boxes = []
            # 
            targets = {}
            # 循环object标签内容: list
            for i in data["object"]:
                if float(i["bndbox"]["xmax"]) <= float(i["bndbox"]["xmin"]) or float(i["bndbox"]["ymax"]) <= float(i["bndbox"]["ymin"]):
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(self.xml_list[index]))
                    continue
                labels.append(self.class_dict[i["name"]])
                boxes.append([float(i["bndbox"]["xmin"]), float(i["bndbox"]["ymin"]), 
                                float(i["bndbox"]["xmax"]), float(i["bndbox"]["ymax"])])
            

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([index])

            targets["boxes"] = boxes
            targets["labels"] = labels
            targets["image_id"] = image_id

            if self.transforms != None:
                image, targets = self.transforms(image, targets)
            # if transforms 
            # image->torch.Size([3, 1080, 1920])
            # targets-> dict  e.g {'boxes': tensor([[127., 484., 155., 508.]]), 'labels': tensor([1])}
            return image, targets

    def xml_to_dict(self, xml_ele):
        if(len(xml_ele) == 0):
            return {xml_ele.tag: xml_ele.text}
        # child <=> xml_ele
        result = {}
        # child ----> <folder> <filename> <path> ...
        for child in xml_ele:
            # {"object": {"xmin": "", "xmax": "", ...}}
            # {"folder": ""}
            data = self.xml_to_dict(child)
            if child.tag != "object":
                # 字典中添加元素
                result[child.tag] = data[child.tag]
            else:
                # 字典内同名key会覆盖
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(data[child.tag])
        # {"annotation": {"folder": "", "path": "", "object": [{"xmin": "", "xmax": "", ...}, {"xmin": "", "xmax": "", ...}]}}
        return {xml_ele.tag: result}

    def get_width_height(self, index):
        with open(self.xml_list[index], 'r') as f:
            xml_ele = etree.fromstring(f.read())
            data = self.xml_to_dict(xml_ele)["annotation"]
            return data["size"]["width"], data["size"]["height"]

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def my_collate(batch):
        """
            zip用法
            my_list = [11,12,13]   my_tuple = (21,22,23)
            print([x for x in zip(my_list,my_tuple)])
            [(11, 21), (12, 22), (13, 23)]
        """
        # ((image:Tensor, ...), (targets:dict, ...))
        return tuple(zip(*batch))

if __name__ == "__main__":
    transforms = {
    "train": Compose([To_tensor(), RandomHorizontalSlip(0.5)]),
    "val": Compose([To_tensor()])
}
    my = my_dataset(data_path="D:/VSCode_item/image_classification/Faster_RCNN/VOCdevkit/VOC2007/ImageSets/main/train.txt",
    xml_path="D:/vehicle_data/VOCdevkit/VOC2007/Annotations", 
    json_path="D:/VSCode_item/image_classification/Faster_RCNN/my_classes.json",
    image_path="D:/vehicle_data/VOCdevkit/VOC2007/JPEGImages", 
    transforms=transforms["train"]) 

    image, targets = my[len(my)-1]
    print(image.shape)
    print(targets)