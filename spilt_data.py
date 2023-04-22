"""
    Annotations  JPEGImages  -> "D:/vehicle_data/VOCdevkit/VOC2007/.."  
    ImageSets/main -> Faster_RCNN文件夹下
    ImageSets/main文件夹下生成划分好的训练集、验证集、测试集(基于Annotations文件)
"""

import random
import argparse
import os

def main(opt):
    random.seed(0)
    # List[str]
    xml_list = sorted([i.split('.')[0] for i in os.listdir(opt.xml_path)])
    # 训练验证集随机序列 e.g [0, 1, 2, 3, 4 ,5 ,6 ,7 ,8 ,9] -> [1, 4, 6, 8, 9]
    trainval_range = random.sample(range(0, len(xml_list)), k=int(len(xml_list) * opt.trainval_test))
    # 验证集随机序列 e.g  [1, 4, 6, 8, 9] -> [1, 6, 8]
    train_range = random.sample(trainval_range, k=int(len(trainval_range) * opt.train_val))

    trainval_list = []
    train_list = []
    val_list = []
    test_list = []

    for index, file_name in enumerate(xml_list):
        if index in trainval_range:
            trainval_list.append(file_name)
            if index in train_range:
                train_list.append(file_name)
            else:
                val_list.append(file_name)
        else:
            test_list.append(file_name)

    with open(os.path.join(opt.main_path, "test.txt"), 'w') as tes:
        tes.write('\n'.join(test_list))
    with open(os.path.join(opt.main_path, "train.txt"), 'w') as tra:
        tra.write('\n'.join(train_list))
    with open(os.path.join(opt.main_path, "val.txt"), 'w') as val:
        val.write('\n'.join(val_list))
    with open(os.path.join(opt.main_path, "trainval.txt"), 'w') as trv:
        trv.write('\n'.join(trainval_list))

def parse_opt():
    parse = argparse.ArgumentParser()

    parse.add_argument("--train_val", default=0.9, type=int)
    parse.add_argument("--trainval_test", default=0.9, type=int)
    parse.add_argument("--xml_path", default="D:/vehicle_data/VOCdevkit/VOC2007/Annotations", type=str, help="Annotations/.xml")
    parse.add_argument("--main_path", default="D:/VSCode_item/image_classification/Faster_RCNN/VOCdevkit/VOC2007/ImageSets/main", type=str, help="main/train.txt")


    return parse.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)