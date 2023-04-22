"""
    操作my_dataset.py返回的image, targets
    image: PIL -> tensor
    targets: 水平翻转
"""
# compose
# PIL to tensor
# randomHorizontalSlip
import random
from torchvision.transforms.functional import to_tensor

class Compose(object):
    def __init__(self, class_list):
        self.class_list = class_list

    def __call__(self, image, targets):
        for i in self.class_list:
            image, targets = i(image, targets)
        return image, targets

class To_tensor(object):
    def __call__(self, image, targets):
        image = to_tensor(image)
        return image, targets

class RandomHorizontalSlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, targets):
        # 水平翻转，调整真实框位置
        # image: Tensor
        if random.random() < self.prob:
            width, height = image.size()[-2:]
            # 沿最后一个维度翻转  列->水平翻转
            image = image.flip(-1)
            # {'boxes': tensor([[925., 484., 953., 508.]]), 'labels': tensor([1])}
            # tensor 支持多维切片  list不支持
            targets["boxes"][:, [0, 2]] = width - targets["boxes"][:, [2, 0]]
        return image, targets