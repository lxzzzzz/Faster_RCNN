from torch import Tensor
from typing import List, Tuple
import torch

def clip_boxes_to_image(proposals_per_image, list_images_size_per_image):
    # type: (Tensor, Tuple[int, int]) ->Tensor
    """
        调整超出边界的框, 将超出坐调整到边界位置
    """
    # 错误5: 切片, satck使用错误
    # 注意: 由于rpn中也会使用，与roi中维度不同，所以不能直接设置为定值
    dim = proposals_per_image.dim()

    # 注意: 这里xmin_xmax = proposals_per_image[: , 0::2]错误
    # ...表示前面所有维度
    xmin_xmax = proposals_per_image[..., 0::2]
    ymin_ymax = proposals_per_image[..., 1::2]
    height, width = list_images_size_per_image

    xmin_xmax = xmin_xmax.clamp(min=0, max=width)
    ymin_ymax = ymin_ymax.clamp(min=0, max=height)
    # 注意: 直接cat box坐标顺序改变 (xmin xmax ymin ymax)
    # torch.cat([xmin_xmax, ymin_ymax], dim=-1)
    # 注意: torch.stack([xmin_xmax, ymin_ymax], dim=2) 会导致坐标顺序错误
    # 直接设置为dim = 3错误
    return torch.stack([xmin_xmax, ymin_ymax], dim=dim).reshape(proposals_per_image.shape)

def remove_small_boxes(proposals_per_image, min_size):
    # type: (Tensor, int) -> Tensor
    """
        返回宽高都大于min_size的索引
    """
    width = proposals_per_image[:, 2] - proposals_per_image[:, 0]
    height = proposals_per_image[:, 3] - proposals_per_image[:, 1]
    # 宽高同时大于min_size
    keep = torch.logical_and(torch.ge(width, min_size), torch.ge(height, min_size))
    # ------------------------------------#
    # keep = torch.tensor([True, False, True]) 
    # torch.where(keep)  
    # (tensor([0, 2]),)
    # ------------------------------------#
    return torch.where(keep)[0]

def batched_nms(proposals_per_image, box_cls_per_image, levels_per_image, num_thresh):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """

    """
    if proposals_per_image.numel() == 0:
        # -------------------------- #
        # >>> a = torch.empty((0, 0))
        # >>> a.shape
        # >>> torch.Size([0, 0])
        # -------------------------- #
        return torch.empty((0,), dtype=torch.int64, device=proposals_per_image.device)
    # max返回Tensor中最大值, 也是Tensor 
    max_coordinate = proposals_per_image.max()
    offsets = levels_per_image.to(proposals_per_image) * (max_coordinate + 1)
    proposals_per_image_nms = proposals_per_image + offsets[:, None]
    keep = torch.ops.torchvision.nms(proposals_per_image_nms, box_cls_per_image, num_thresh)
    return keep

def box_iou(box1, box2):
    # type: (Tensor, Tensor) -> Tensor
    """
        Args: box1: (N * 4)  box2: (M * 4)
        return: (N * M)
    """
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    # N * M * 2
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    # 注意: rightbottom -> min
    # N * M * 2
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    # N * M * 2
    wh = (rb - lt).clamp(min=0)
    inner = wh[:, :, 0] * wh[:, :, 1]
    return inner / (box1_area[:, None] + box2_area - inner)
