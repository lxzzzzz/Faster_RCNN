import math
import torch
from typing import List, Tuple
from torch import Tensor

class BalancedPositiveNegativeSampler(object):
    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, labels):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        pos_index = []
        neg_index = []
        for label in labels:
            # 正样本的索引
            # 错误2: positive = torch.where(torch.eq(label, 1.0))[0]
            positive = torch.where(torch.ge(label, 1.0))[0]
            # 负样本的索引
            negative = torch.where(torch.eq(label, 0.0))[0]
            # 设置正负样本的数量
            # int * float -> float -> int
            set_positive = int(self.batch_size_per_image * self.positive_fraction)
            num_positive = min(positive.numel(), set_positive)
            set_negative = self.batch_size_per_image - num_positive
            num_negative = min(negative.numel(), set_negative)
            # ---------------------------------#
            # 根据正负样本的数量随机选择正负样本
            # randperm 将0~n-1（包括0和n-1）随机打乱后获得的数字序列
            # ---------------------------------#
            # 注意: tensors used as indices must be long, byte or bool tensors
            # 这里的perm1数据类型不能是float
            perm1 = torch.randperm(positive.numel(), device=label.device)[: num_positive]
            perm2 = torch.randperm(negative.numel(), device=label.device)[: num_negative]
            
            pos_index_per_image = positive[perm1]
            neg_index_per_image = negative[perm2]
            pos_index_per_image_mask = torch.zeros_like(label, dtype=torch.uint8, device=label.device)
            pos_index_per_image_mask[pos_index_per_image] = 1
            pos_index.append(pos_index_per_image_mask)

            neg_index_per_image_mask = torch.zeros_like(label, dtype=torch.uint8, device=label.device)
            neg_index_per_image_mask[neg_index_per_image] = 1
            neg_index.append(neg_index_per_image_mask)
        return pos_index, neg_index

class BoxCoder(object):
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        super(BoxCoder, self).__init__()
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode(self, box_regression, anchors):
        # type: (Tensor, List[Tensor]) -> Tensor
        """
            预测信息 + 先验框anchors信息 -> proposal anchors
        """
        # ---------------------------------#
        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        # ---------------------------------#
        all_anchors = torch.cat(anchors, dim=0)
        # (xmin, ymin, xmax, ymax) -> (x, y, w, h)
        width = all_anchors[:, 2] - all_anchors[:, 0]
        height = all_anchors[:, 3] - all_anchors[:, 1]
        x = all_anchors[:, 0] + 0.5 * width
        y = all_anchors[:, 1] + 0.5 * height
        # 调整信息
        wx, wy, ww, wh = self.weights
        dx = box_regression[:, 0::4] / wx
        dy = box_regression[:, 1::4] / wy
        dw = box_regression[:, 2::4] / ww
        dh = box_regression[:, 3::4] / wh
        # 限制dw和dh的最大输入值
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        # 调整后的框位置信息
        pred_x = dx * width[:, None] + x[:, None]
        pred_y = dy * height[:, None] + y[:, None]
        pred_w = torch.exp(dw) * width[:, None]
        pred_h = torch.exp(dh) * height[:, None]
        # (x, y, w, h) -> (xmin, ymin, xmax, ymax)
        xmin = pred_x - 0.5 * pred_w
        ymin = pred_y - 0.5 * pred_h
        xmax = pred_x + 0.5 * pred_w
        ymax = pred_y + 0.5 * pred_h
        # 合并
        # 错误4: 这里reshape(xmin.shape[0], -1)代替flatten(1)第一个维度为0会报错
        pred_box = torch.stack((xmin, ymin, xmax, ymax), dim=2).flatten(1)
        # 错误4: 防止pred_boxes为空时导致reshape报错
        if xmin.shape[0] > 0:
            return pred_box.reshape(xmin.shape[0], -1, 4)
        else:
            return pred_box

    def encode(self, matched_gt_boxes_list, anchors_list):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
            编码过程计算anchors上每个先验框坐标到对应gtbox坐标的回归参数
        """
        # 每张图片对应的先验框数量
        num_anchors_per_image = [len(matched_gt_boxes) for matched_gt_boxes in matched_gt_boxes_list]
        # 堆叠List中的每张图片
        matched_gt_boxes = torch.cat(matched_gt_boxes_list, dim=0)
        anchors = torch.cat(anchors_list, dim=0)
        # 获得中心坐标以及宽高dx, dy, dw, dh权重
        weights = torch.as_tensor(self.weights, dtype=anchors.dtype, device=matched_gt_boxes.device)
        wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]
        # (xmin, ymin, xmax, ymax) -> (x, y, w, h)
        gt_width = matched_gt_boxes[:, 2].unsqueeze(1) - matched_gt_boxes[:, 0].unsqueeze(1)
        gt_height = matched_gt_boxes[:, 3].unsqueeze(1) - matched_gt_boxes[:, 1].unsqueeze(1)
        gt_x = matched_gt_boxes[:, 0].unsqueeze(1) + 0.5 * gt_width
        gt_y = matched_gt_boxes[:, 1].unsqueeze(1) + 0.5 * gt_height

        anchors_width = anchors[:, 2].unsqueeze(1) - anchors[:, 0].unsqueeze(1)
        anchors_height = anchors[:, 3].unsqueeze(1) - anchors[:, 1].unsqueeze(1)
        anchors_x = anchors[:, 0].unsqueeze(1) + 0.5 * anchors_width
        anchors_y = anchors[:, 1].unsqueeze(1) + 0.5 * anchors_height
        # 计算回归参数
        # 注意: dx = wx * (gt_x - anchors_x) / anchors_x 分母此时为0, 导致结果inf无穷大
        dx = wx * (gt_x - anchors_x) / anchors_width
        dy = wy * (gt_y - anchors_y) / anchors_height
        dw = ww * torch.log(gt_width  / anchors_width)
        dh = wh * torch.log(gt_height / anchors_height)
        # [:, 1] ... -> [:, 4]
        return torch.cat([dx, dy, dw, dh], dim=1).split(num_anchors_per_image, dim=0)

class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def set_maxiou_positive(self, indexes, indexes_clone, iou_matrix):
        """
            作用: 解决当任何一个预测框与某一真实框的iou都小于0.7
            调整预测框, 计算与真实框最大iou预测框索引,  并调整预测框为正样本
        """
        values, _ = iou_matrix.max(dim=1)
        tuple_row_col = torch.where(torch.eq(iou_matrix, values[:, None]))[1]
        indexes[tuple_row_col] = indexes_clone[tuple_row_col]

    def __call__(self, iou_matrix):
        # type: (Tensor) -> None
        values, indexes = iou_matrix.max(dim=0)

        if self.allow_low_quality_matches:
            indexes_clone = indexes.clone()
        else:
            indexes_clone = None

        low_mask = values < self.low_threshold
        between_mask = (values >= self.low_threshold) & (values < self.high_threshold)

        indexes[low_mask] = self.BELOW_LOW_THRESHOLD
        indexes[between_mask] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_maxiou_positive(indexes, indexes_clone, iou_matrix)
        return indexes

def smooth_l1_loss(box_regression, regression_targets, beta=1 / 9, size_average=True):
    # type: (Tensor, Tensor, float, bool) -> None
    # 注意: abs(box_regression, regression_targets)错误
    n = torch.abs(box_regression - regression_targets)
    condition = torch.lt(n, beta)
    loss = torch.where(condition, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()