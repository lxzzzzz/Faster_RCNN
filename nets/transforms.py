"""
    对传入网络的
    image进行normalize和resize
    targets适应image改变

"""
import torch
import math
from torch import Tensor, nn
from typing import List, Tuple, Dict

class Transforms_Img_Tar(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        # type: (int, int, List, List) -> None
        super(Transforms_Img_Tar, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        # type: (Tensor) ->Tensor
        """
            标准化处理
        """
        # 统一数据类型
        dtype, device = image.dtype, image.device

        # scalar -> vector
        # mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device).unsqueeze(0)
        # std = torch.as_tensor(self.image_std, dtype=dtype, device=device).unsqueeze(0)
        
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # None默认在已有的维度前添加维度  在后面添加[:, None, None]: shape [3] -> [3, 1, 1]
        # 维度相同, shape不同
        # 如果不从scalar -> vector第一个维度上无法进行切片操作，会报错
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        # type: (Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]
        """
            image shape[channel, height, width]
        """
        height, width = image.shape[-2:]
        img_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(img_shape))
        max_size = float(torch.max(img_shape))
        scale = float(torch.tensor(self.min_size)) / min_size

        if max_size * scale > self.max_size:
            scale = float(torch.tensor(self.max_size)) / max_size

        # interpolate利用插值的方法缩放图片
        # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
        # bilinear只支持4D Tensor
        image = torch.nn.functional.interpolate(image[None, :, :, :], scale_factor=scale, 
                            mode="bilinear", recompute_scale_factor=True, align_corners=False)[0]
        # 验证模式不用处理target
        if target is None:
            return image, target
        # Tensor
        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        
        ratios_height = torch.tensor(image.shape[-2:][0], dtype=torch.float32, device=bbox.device) / torch.tensor(height, dtype=torch.float32, device=bbox.device)
        ratios_width = torch.tensor(image.shape[-2:][1], dtype=torch.float32, device=bbox.device) / torch.tensor(width, dtype=torch.float32, device=bbox.device)
        
        # Tensor.unbind(dim=x) 移除某一维度
        # torch.Size([2, 4]) -> (tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,]))
        xmin, ymin, xmax, ymax = bbox.unbind(1)
        xmin = xmin * ratios_width
        xmax = xmax * ratios_width
        ymin = ymin * ratios_height
        ymax = ymax * ratios_height
        # (tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,])) -> torch.Size([2, 4])
        # dim=1理解为堆叠获得的维度在维度1的位置
        bbox = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        target["boxes"] = bbox
        return image, target

    def max_c_h_w(self, list_shape):
        # type: (List[List[int]]) -> List[int]
        max_c_h_w = list_shape[0]
        for item in list_shape[1:]:
            for index, value in enumerate(item):
                max_c_h_w[index] = max(max_c_h_w[index], value)
        return max_c_h_w

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
            打包batch_size个图片(大小不同)
        """
        # list[list] -> list
        # 注意 这里list(Tensor)与[Tensor]的区别 
        max_c_h_w = self.max_c_h_w([list(image.shape) for image in images])
        # 将获得的最大h, w尺寸缩放到32的整数倍
        # 加速硬件计算
        stride = float(size_divisible)
        # / 浮点数除法  向上取整
        # 注意 * stride(float)后 转换int
        max_c_h_w[1] = int(math.ceil(float(max_c_h_w[1]) / stride) * stride)
        max_c_h_w[2] = int(math.ceil(float(max_c_h_w[2]) / stride) * stride)
        # [channel, height, width] ->[batch, ...]
        # 这里是给list添加元素
        batch_shape = max_c_h_w
        batch_shape.insert(0, len(images))
        # 创建全是0模板 list -> tensor
        batch_images = torch.zeros(batch_shape, dtype=images[0].dtype, device=images[0].device)
        # 将images复制到模板中
        for image, to_image in zip(images, batch_images):
            # 将图片复制到模板的左上角对其
            to_image[: image.shape[0], : image.shape[1], : image.shape[2]].copy_(image)
        return batch_images

    def forward(self, images, targets=None):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[Tensor, List[Tuple[int, int]], List[Dict[str, Tensor]]]
        """
            作用于DataLoader之后, batch_size个image
            注意这里的images
        """
        # tuple(tensor) -> list(tensor)
        list_images_size = []
        images = [image for image in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None and target is not None:
                targets[i] = target
        # 记录resize后, paddding前的图片尺寸
        image_sizes = [image.shape[-2:] for image in images]
        images = self.batch_images(images)
        for image_size in image_sizes:
            list_images_size.append((image_size[0], image_size[1]))
        return images, list_images_size, targets

    def postprocess(self, detections, list_images_size, original_image_sizes):
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]) -> None
        """
            将网络的预测结果映射到resize之前的图片上
        """
        if self.training:
            return detections
        for i, (j, k, l) in enumerate(zip(detections, list_images_size, original_image_sizes)):
            bbox = j["boxes"]
            # 错误3: ratios_x 计算中，除数与被除数设置反了
            ratios_height = torch.tensor(original_image_sizes[0][0], dtype=torch.float32, device=bbox.device) / torch.tensor(list_images_size[0][0], dtype=torch.float32, device=bbox.device)
            ratios_width = torch.tensor(original_image_sizes[0][1], dtype=torch.float32, device=bbox.device) / torch.tensor(list_images_size[0][1], dtype=torch.float32, device=bbox.device)
            
            # Tensor.unbind(dim=x) 移除某一维度
            # torch.Size([2, 4]) -> (tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,]))
            xmin, ymin, xmax, ymax = bbox.unbind(1)
            xmin = xmin * ratios_width
            xmax = xmax * ratios_width
            ymin = ymin * ratios_height
            ymax = ymax * ratios_height
            # (tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,]), tensor.Size([2,])) -> torch.Size([2, 4])
            # dim=1理解为堆叠获得的维度在维度1的位置
            bbox = torch.stack((xmin, ymin, xmax, ymax), dim=1)
            detections[i]["boxes"] = bbox
        return detections

        