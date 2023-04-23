import torch
from torch import nn, Tensor
import torch.nn.functional as F 
from collections import OrderedDict
from typing import Tuple, List, Dict
from my_utils.codebox import BoxCoder, Matcher, BalancedPositiveNegativeSampler, smooth_l1_loss
from my_utils.box_util import clip_boxes_to_image, remove_small_boxes, batched_nms, box_iou


class RPN_Head(nn.Module):
    """
        对特征层进行3*3卷积
        features层的channle = 256
    """
    __annotations__ = {
        "in_channels": int,
        "num_anchors": int,
    }
    
    def __init__(self, in_channels, num_anchors):
        # type: (int, int, int) -> None
        super(RPN_Head, self).__init__()
        # 3*3滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 存在目标的概率 num_anchors表示每个特征层上每个位置存在框个数
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 先验框的调整参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        # 初始化权重、偏置参数
        for layer in self.children():
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        cls_prob_list = []
        bbox_pred_list = []
        for feature in features:
            # 注意: nn.Relu 与 F.relu
            output = F.relu(self.conv(feature))
            cls_prob_list.append(self.cls_logits(output))
            bbox_pred_list.append(self.bbox_pred(output))
        return cls_prob_list, bbox_pred_list

class AnchorsGenerator(nn.Module):
    """
        生成先验框
    """
    def __init__(self, sizes=(128, 256, 512), ratios=(0.5, 1.0, 2.0)):
        # type: (Tuple, Tuple) -> None
        super(AnchorsGenerator, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            # ((s,) for s in sizes)返回的不是tuple 与[(s,) for s in sizes]返回list不同
            sizes = tuple((s,) for s in sizes)
        if not isinstance(ratios[0], (list, tuple)):
            # ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), ...)
            ratios = (ratios,) * len(sizes)
        self.sizes = sizes
        self.ratios = ratios
        self.cell_anchors = None

    def generator_anchors(self, size, ratio, dtype=torch.float32, device = "cpu"):
        # type: (Tuple, Tuple, torch.dtype, torch.device) -> Tensor
        """
            作用: 生成先验框 -> Tensor torch.Size([3, 4])
            注意: 1. 图片对应的 height, width顺序先验框的位置信息对应的 width, height
            2. (xmin, ymin, xmax, ymax)以特征层左上角为原点
        """
        # 注意: torch.as_tensor(size, dtype, device)错误的原因
        size = torch.as_tensor(size, dtype=dtype, device=device)
        ratio = torch.as_tensor(ratio, dtype=dtype, device=device)
        h_ratio = torch.sqrt(ratio)
        w_ratio = 1.0 / h_ratio
        # troch.Size([3, 1]) * torch.Size([1, 1])-> torch.Size([3])
        w_size = (w_ratio[:, None] * size[None, :]).view(-1)
        h_size = (h_ratio[:, None] * size[None, :]).view(-1)
        # 4个torch.Size([3])堆叠 -> torch.Size([3, 4])
        # 以(0, 0)为中心
        base_anchors = torch.stack([-w_size, -h_size, w_size, h_size], dim=1) / 2
        # 四舍五入，还是float32
        return base_anchors.round()

    def map_anchors(self, features_shape, strides):
        # type: (List[torch.Size], List[Tuple[Tensor, Tensor]]) -> List[Tensor]
        """
            作用: 将每个特征层上的先验框位置信息映射回resize和padding后的原图
            返回: List[Tensor] Tensor对应(一张图片 -> 一个特征层框数量 H * W * 3)
        """
        original_anchors = []
        for feature_shape, stride, cell_anchor in zip(features_shape, strides, self.cell_anchors):
            feature_height, feature_width = feature_shape
            stride_height, stride_width = stride
            # 注意：新建变量需要定义device
            device = stride_height.device
            y_axe = torch.arange(0, feature_height, dtype=torch.float32, device=device) * stride_height
            x_axe = torch.arange(0, feature_width, dtype=torch.float32, device=device) * stride_width
            # y_distribute x_distribute -> torch.Size([feature_height, feature_width])
            # 原图上建立feature_height * feature_width个网格左上角对应的y, x坐标
            y_distribute, x_distribute = torch.meshgrid(y_axe, x_axe)
            # 原图上对应的坐标信息(xmin, ymin, xmax, ymax)
            # 合并为一个维度
            y_distribute = y_distribute.reshape(-1)
            # x -> (0, 4, 8, 16, ..., 1340, 0, 4, 8, ..., 1340, ...)
            # y -> (0, 0, 0, 0, ...,        0, 0, 0, 0, ...,    ...)
            x_distribute = x_distribute.reshape(-1)
            # torch.Size([feature_height * feature_width, 4]) 
            x_y_distribue = torch.stack([x_distribute, y_distribute, x_distribute, y_distribute], dim=1)
            # 特征图先验框信息 + 调整信息
            original_anchor = cell_anchor[None, :, :] + x_y_distribue[:, None, :]
            original_anchors.append(original_anchor.reshape(-1, 4))
        return original_anchors

    def forward(self, images, list_images_size, features):
        # type: (Tensor, List[Tuple], List[Tensor]) ->List[Tensor]
        """
            images ->resize, padding, batch后的图片
            list_images_size ->resize后padding前的图片尺寸
            return: batch张图片, 每张图片对应的所有特征层上所有先验框的和
        """
        # 获取特征层数据类型以及设备类型
        dtype, device = features[0].dtype, features[0].device
        # 计算每个特征层对应实际图像比例
        features_shape = [feature.shape[-2:] for feature in features]
        images_shape = images.shape[-2:]
        strides = [(torch.tensor(images_shape[0] // feature_shape[0], dtype=torch.int64, device=device), 
                    torch.tensor(images_shape[1] // feature_shape[1], dtype=torch.int64, device=device)                             
                    ) for feature_shape in features_shape]
        # 根据sizes, ratios生成先验框
        cell_anchors = []
        # 每个特征层对应一个size大小的先验框
        for size, ratio in zip(self.sizes, self.ratios):
            cell_anchors.append(self.generator_anchors(size, ratio, dtype, device))
        self.cell_anchors = cell_anchors
        # 将先验框信息(特征层上)映射到原图上
        original_anchors = self.map_anchors(features_shape, strides)
        # 遍历batch张图片，返回List[List[Tensor]]
        anchors = []
        for i in range(images.shape[0]):
            anchors.append(original_anchors)
        # 将每一张图片的所有特征层上的先验框信息合并
        # torch.stack会在原有的维度上增加一个维度, 只允许对相同大小、维度的张量进行拼接
        # troch.cat则会保持原有维度, 在指定维度上进行叠加.
        anchors = [torch.cat(anchor, dim=0) for anchor in anchors]
        return anchors


class RPN(nn.Module):
    """
        RPN框架
    """
    def __init__(self, head, anchors_generator, 
                rpn_pre_nms_top_n, rpn_post_nms_top_n, 
                rpn_fg_iou_thresh, rpn_bg_iou_thresh, 
                batch_size_per_image, positive_fraction,
                rpn_score_thresh, rpn_nms_thresh
                ):
        super(RPN, self).__init__()
        self.head = head
        self.anchors_generator = anchors_generator
        self.codebox = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.min_size = 1.0
        self.score_thresh = rpn_score_thresh
        self.nms_thresh = rpn_nms_thresh

        self.proposal_matcher = Matcher(
            rpn_fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            rpn_bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction)


    def pre_nms_top_n(self):
        if self.training:
            return self.rpn_pre_nms_top_n["train"]
        else:
            return self.rpn_pre_nms_top_n["test"]

    def post_nms_top_n(self):
        if self.training:
            return self.rpn_post_nms_top_n["train"]
        else:
            return self.rpn_post_nms_top_n["test"]

    def get_top_n_anchors_index(self, box_cls, num_anchors_per_features):
        # type: (Tensor, List[int]) -> Tensor
        """
            获得每个特征层上概率靠前的特征框索引
            注意: 由于box_cls将每个特征层上概率堆叠在一起, 
            所以下面基于每一个特征层计算的索引需要加上offset
        """
        index_list= []
        # 偏置 
        offset = 0 
        for obj in box_cls.split(num_anchors_per_features, dim=1):
            num_anchors = obj.shape[1]
            pre_nms_top_n = min(num_anchors, self.pre_nms_top_n())
            # 返回前top n的值和索引
            _, top_n_index = obj.topk(pre_nms_top_n, dim=1)
            index_list.append(top_n_index + offset)
            offset += num_anchors
        return torch.cat(index_list, dim=1)

    def filter_proposals(self, proposals, box_cls, list_images_size, num_anchors_per_features):
        # type: (Tensor, Tensor, List[Tuple], List) -> Tuple[List[Tensor], List[Tensor]]
        """
            过滤建议框
            Args:
                proposals: 预测得到的bbox坐标(num_images, num_anchors, 4)
                box_cls: 预测目标的概率 (num_images * num_anchors, 1)
                list_images_size: resize后padding前的图片大小
                num_anchors_per_features: 每一特征层预测框数量
            return: 筛选后每一张图片对应的建议框
            
        """
        # 改变box_cls形状
        box_cls = box_cls.reshape(len(list_images_size), -1)
        # levels负责记录分隔不同预测特征层上的anchors索引信息
        levels = [torch.full((n, ), index, dtype=torch.int64, device=proposals.device) for index, n in enumerate(num_anchors_per_features)]
        levels = torch.cat(levels, dim=0)[None, :].expand_as(box_cls)
        # ----------------------------------------------#
        # train模式前2000, test模式下前1000
        # 获得每个特征层上概率靠前的特征框索引
        # ----------------------------------------------#
        top_n_index = self.get_top_n_anchors_index(box_cls, num_anchors_per_features)
        image_range = torch.arange(len(list_images_size), device=proposals.device)
        # torch.Size([batch_size, 1])
        batch_idx = image_range[:, None]
        # ----------------------------------------------#
        # torch.Size([batch_size, top_n_index.shape[1]])
        # 得到筛选后的box_cls levels proposals
        # 切片操作？？
        # ----------------------------------------------#
        box_cls = box_cls[batch_idx, top_n_index]
        box_cls = torch.sigmoid(box_cls)
        levels = levels[batch_idx, top_n_index]
        proposals = proposals[batch_idx, top_n_index]
        # ----------------------------------------------#
        # 遍历每张图片调整超过图片边界目标框(调整到边界位置)、移除小目标、小概率框, nms处理
        # ----------------------------------------------#
        proposals_list = []
        box_cls_list = []
        for i, j, k, l in zip(proposals, box_cls, levels, list_images_size):
            # 调整超过图片边界目标框(调整到边界位置)
            i = clip_boxes_to_image(i, l)
            # 移除小目标
            # 返回宽高都大于min_size的索引
            keep = remove_small_boxes(i, self.min_size)
            i, j, k = i[keep], j[keep], k[keep]
            # 移除小概率
            keep = torch.where(torch.ge(j, self.score_thresh))[0]
            i, j, k = i[keep], j[keep], k[keep]
            # nms
            keep = batched_nms(i, j, k, self.nms_thresh)[: self.post_nms_top_n()]
            i, j = i[keep], j[keep]
            proposals_list.append(i)
            box_cls_list.append(j)
        return proposals_list, box_cls_list

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
            返回每张图片预测框对应的正负样本类型以及预测框对应的gtbox坐标信息
        """
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # --------------------------- #
            # 计算真实框与预测框的交并比
            # --------------------------- #
            gt_boxes = targets_per_image["boxes"]
            iou_matrix = box_iou(gt_boxes, anchors_per_image)
            # --------------------------- #
            # 将预测框划分正负忽略样本(正: >=0 负: -1 忽略: -2) 
            # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
            # --------------------------- #
            # 注意：????????????????????????????
            # ne_po_ig_index 每个预测框对应的gtbox索引
            # 返回每个预测框对应的真实gtbox坐标信息
            # --------------------------- #
            ne_po_ig_index = self.proposal_matcher(iou_matrix)
            # 注意: 这一步将忽略的样本以及负样本标志都置为0, 只是避免gt_boxes[]索引负数报错，对后面计算loss没有影响
            matched_gt_boxes_per_image = gt_boxes[ne_po_ig_index.clamp(min=0)]
            # 正样本
            positive_mask  = ne_po_ig_index >= 0
            labels_per_image = positive_mask.to(dtype=torch.float32)
            # 负样本
            negative_mask = ne_po_ig_index == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[negative_mask] = 0.0
            # 忽略样本
            ignore_mask = ne_po_ig_index == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_mask] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def compute_loss(self, box_cls, box_regression, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
            计算RPN损失
        """
        # -----------------------------------------#
        # 1. 选择部分正负样本进行训练
        # -----------------------------------------#
        sample_pos_labels, sample_neg_labels = self.fg_bg_sampler(labels)
        # 默认获取非0位置的索引
        # 获取所有图片正样本索引
        sample_pos_index = torch.where(torch.cat(sample_pos_labels, dim=0))[0]
        # 获取所有图片负样本索引
        sample_neg_index = torch.where(torch.cat(sample_neg_labels, dim=0))[0]
        # 堆叠
        sample_index = torch.cat((sample_pos_index, sample_neg_index), dim=0)
        # 降维 保持与其它量维度相同
        box_cls = box_cls.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        # -----------------------------------------#
        # 2. 计算边界框回归损失
        # -----------------------------------------#
        # **这里使用的是sample_pos_index而不是sample_index, 除数是正负样本的总数sample_index.numel()**
        box_loss= smooth_l1_loss(box_regression[sample_pos_index], 
                            regression_targets[sample_pos_index], size_average=False) / (sample_index.numel())

        # -----------------------------------------#
        # 3. 计算目标预测概率损失
        # -----------------------------------------#
        object_loss = F.binary_cross_entropy_with_logits(box_cls[sample_index], labels[sample_index])
        return object_loss, box_loss

    def forward(self, images, list_images_size, features, targets=None):
        # type: (Tensor, List[Tuple], Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], Dict[str, Tensor]]
        # -----------------------------------------#
        # 1. 调整特征图像存放类型 Dict[Tensor] -> List[Tensor]
        # [Torch.Size([2, 256, 192, 336]), ...]
        # -----------------------------------------#
        features = list(features.values())
        # -----------------------------------------#
        # 2. 获取3*3滑动窗口在各个特征层上得到的预测结果
        # 存在目标概率[Torch.Size([2, 3, 192, 336]), ...] 先验框回归参数[Torch.Size([2, 3*4, 192, 336]), ...]
        # -----------------------------------------#
        cls_prob_list, bbox_pred_list = self.head(features)
        num_anchors_per_features = [cls_prob[0].shape[0] * cls_prob[0].shape[1] * cls_prob[0].shape[2] for cls_prob in cls_prob_list]
        # -----------------------------------------#
        # 3. 调整cls_prob_list, bbox_pred_list形状
        # N: batch_size
        # A: anchors_num_per_position
        # C: classes_num or 4(bbox coordinate)
        # H: height
        # W: width
        # cls_prob_list [N, A*C, H, W] -> [N, -1, C] 这里C指的是classes_num
        # bbox_pred_list [N, A*4, H, W] -> [N, -1, C] 这里C指的是4
        # box_cls [-1, C]
        # box_regression [-1, C]
        # -----------------------------------------#
        # 遍历每个特征层
        for index, (cls_prob, bbox_pred) in enumerate(zip(cls_prob_list, bbox_pred_list)):
            
            N, AC, H, W = cls_prob.shape
            A4 = bbox_pred.shape[1]
            A = A4 // 4
            C = AC // A
            # 错误1: reshape(N, A, C, -1).permute(...)
            cls_prob = cls_prob.view(N, -1, C, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, C)
            bbox_pred = bbox_pred.view(N, -1, C, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
            cls_prob_list[index] = cls_prob
            bbox_pred_list[index] = bbox_pred
        box_cls = torch.cat(cls_prob_list, dim=1).reshape(-1, C)
        box_regression = torch.cat(bbox_pred_list, dim=1).reshape(-1, 4)
        # -----------------------------------------#
        # 4. batch张图片, 每张图片对应的所有特征层上所有先验框的位置信息和
        # -----------------------------------------#
        anchors = self.anchors_generator(images, list_images_size, features)
        # -----------------------------------------#
        # 5. 预测信息 + anchors信息 生成proposal anchors
        # -----------------------------------------#
        proposals = self.codebox.decode(box_regression.detach(), anchors).reshape(len(anchors), -1, 4)
        # -----------------------------------------#
        # 6. 过滤得到 filter proposal anchors
        # -----------------------------------------#
        proposals_list, box_cls_list = self.filter_proposals(proposals, box_cls.detach(), 
                                            list_images_size, num_anchors_per_features)
        # -----------------------------------------#
        # 7. 计算RPNLoss
        # 只有在训练模式下才计算RPN损失
        # -----------------------------------------#
        rpn_losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 计算每个特征层上的每个位置对应的(先验框->最匹配的gtbox)回归参数 
            regression_targets = self.codebox.encode(matched_gt_boxes, anchors)
            # 计算损失
            object_loss, box_loss = self.compute_loss(box_cls, box_regression, labels, regression_targets)
            rpn_losses = {
                "rpn_object_loss": object_loss, 
                "rpn_box_loss": box_loss
            }
        return proposals_list, rpn_losses