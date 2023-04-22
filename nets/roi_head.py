import torch
from torch import nn, Tensor
from typing import List, Tuple, Dict
from my_utils.box_util import box_iou
import torch.nn.functional as F
from my_utils.codebox import Matcher, BalancedPositiveNegativeSampler, BoxCoder, smooth_l1_loss
from my_utils.box_util import clip_boxes_to_image, remove_small_boxes, batched_nms, box_iou

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class ROIHeads(nn.Module):
    def __init__(self, fg_iou_thresh, bg_iou_thresh, 
                    roi_batch_size_per_image, roi_positive_fraction, 
                    box_roi_pool, box_head, box_predictor, 
                    score_thresh, nms_thresh, detection_per_img):
        super(ROIHeads, self).__init__()
        self.proposal_matcher = Matcher(
            fg_iou_thresh, bg_iou_thresh,
            allow_low_quality_matches=False
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            roi_batch_size_per_image, roi_positive_fraction)

        self.codebox = BoxCoder(weights=(10., 10., 5., 5.))

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head 
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

    def assign_targets_to_anchors(self, proposals_list, gt_boxes, gt_labels):
        """
            将proposals划分为正, 负, 忽略样本
            返回matched_gt_labels_list对应 -> (负样本标签->0 丢弃的样本->-1 正样本1 2 3...)
            clamped_ne_po_ig_index_list对应clamped后
        """
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_gt_labels_list = []
        clamped_ne_po_ig_index_list = []
        for proposals, gt_box, gt_label in zip(proposals_list, gt_boxes, gt_labels):
            # box_iou的参数顺序错误 会报错
            iou_matrix = box_iou(gt_box, proposals)
            # < low_threshold 置为 BELOW_LOW_THRESHOLD = -1;
            # >low_threshold & < hight_threshold 置为BETWEEN_THRESHOLDS = -2
            # 其它的在 0 ~ (gtbox个数-1)之间
            ne_po_ig_index = self.proposal_matcher(iou_matrix)
            # --------------------------------#
            # 负样本标签->0 丢弃的样本->-1
            # --------------------------------#
            # 避免检索负数报错
            # 每个proposal对应的gtbox索引, clamp后的
            clamped_ne_po_ig_index = ne_po_ig_index.clamp(min=0)
            # 每个proposal对应的gtbox的label信息, 暂时的
            matched_gt_labels_per_image = gt_label[clamped_ne_po_ig_index]
            # 负样本对应标签置为0 背景对应标签置为-1 正样本对应json文件中 1 2 3 ...
            matched_gt_labels_per_image[ne_po_ig_index == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0
            matched_gt_labels_per_image[ne_po_ig_index == self.proposal_matcher.BETWEEN_THRESHOLDS] = -1
            
            matched_gt_labels_list.append(matched_gt_labels_per_image)
            clamped_ne_po_ig_index_list.append(clamped_ne_po_ig_index)
        return matched_gt_labels_list, clamped_ne_po_ig_index_list

    def select_training_samples(self, proposals_list, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        gt_boxes = [dic["boxes"] for dic in targets]
        gt_labels = [dic["labels"] for dic in targets]
        # 堆叠每张图片的proposals以及gtbox
        # ??????????????????????????
        # 由于正样本数量较少，此时将gtbox添加到proposals中增加正样本量
        proposals_list = [torch.cat((proposals, gt_box), dim=0) for proposals, gt_box in zip(proposals_list, gt_boxes)]
        # 对每张图片上的proposals分配gtbox、正负样本
        # 与rpn中的assign_targets_to_anchors()作用相同
        labels_list, index_list = self.assign_targets_to_anchors(proposals_list, gt_boxes, gt_labels)
        # 选取部分正负样本, 并获取每张图片上的正负样本位置索引
        sample_list = []
        sample_pos_labels, sample_neg_labels = self.fg_bg_sampler(labels_list)
        for pos, neg in zip(sample_pos_labels, sample_neg_labels):
            # 一张图片上的正负样本位置索引
            sample_list.append(torch.where(pos | neg)[0])
        # 遍历每张图片
        matched_gt_boxes = []
        for i in range(len(sample_list)):
            sample_list_per_image = sample_list[i]
            labels_list[i] = labels_list[i][sample_list_per_image]
            
            proposals_list[i] = proposals_list[i][sample_list_per_image]
            # 注意: 这里两层索引理解
            gt_boxes_per_image = gt_boxes[i]
            index_list[i] = index_list[i][sample_list_per_image]
            # 获得每张图片上选择的正负样本对应的gtbox信息
            matched_gt_boxes.append(gt_boxes_per_image[index_list[i]])
        
        regression_targets = self.codebox.encode(matched_gt_boxes, proposals_list)
        return proposals_list, labels_list, regression_targets


    def forward(self, features, proposals_list, list_images_size, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], List[Dict[str, Tensor]]) -> None
        """
        # --------------------------------#
        # 训练模式下, 需要划分样本并计算loss:
        # 对rpn网络筛选得到的proposals, 以及targets操作 
        # -> 划分正负样本
        # -> 确定正负样本对应的gtbox
        # -> 计算正负样本对应gtbox的regression参数
        # --------------------------------#
        """
        fastrcnn_losses = {}
        result = []
        # 训练模式
        if self.training:
            proposals_list, labels_list, regression_targets = self.select_training_samples(proposals_list, targets)
        else:
            labels_list = None
            regression_targets = None
        
        box_features = self.box_roi_pool(features, proposals_list, list_images_size)
        box_features = self.box_head(box_features)
        cls_prob, bbox_pred = self.box_predictor(box_features)

        if self.training:  
            classification_loss, box_loss = self.fastrcnn_loss(cls_prob, bbox_pred, labels_list, regression_targets)
            fastrcnn_losses= {
                "fastrcnn_classifier_loss": classification_loss,
                "fastrcnn_box_reg_loss": box_loss
            }
        # 预测模式
        else:
            boxes_all, scores_all, labels_all = self.postprocess_detections(cls_prob, bbox_pred, proposals_list, list_images_size)
            for i in range(len(proposals_list)):
                result.append(
                    {
                        "boxes": boxes_all[i],
                        "scores": scores_all[i],
                        "labels": labels_all[i]
                    }
                )
        return result, fastrcnn_losses

    def fastrcnn_loss(self, cls_prob, bbox_pred, labels_list, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Arguments:
            cls_prob : 预测类别概率信息，shape=[num_anchors, num_classes]
            bbox_pred : 预测边目标界框回归信息
            labels_list : 真实类别信息
            regression_targets : 真实目标边界框信息
        """
        labels = torch.cat(labels_list, dim=0) 
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(cls_prob, labels)
        # 调整bbox_pred最后一个维度与regression_targets对应
        # bbox_pred -> (num_proposals, num_classes, 4) 
        bbox_pred = bbox_pred.reshape(bbox_pred.shape[0], -1, 4)
        # 注意: 这里gt(>) 不是 ge(>=)
        pos_index = torch.where(torch.gt(labels, 0))[0]
        pos_labels = labels[pos_index]
        box_loss = smooth_l1_loss(bbox_pred[pos_index, pos_labels], regression_targets[pos_index], 
                                size_average=False) / (labels.numel())
        return classification_loss, box_loss

    def postprocess_detections(self, cls_prob, bbox_pred, proposals_list, list_images_size):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
            Args:
                cls_prob: 网络预测类别概率信息 (num_images * proposals_per_image, 1)
                bbox_pred: 网络预测的边界框回归参数 (num_images * proposals_per_image, 8)
                proposals_list: rpn输出的proposal 
                list_images_size: 打包成batch前的每张图像的宽高
        """
        # 回归参数 + rpn筛选后的建议框
        num_proposals_per_image = [proposals.shape[0] for proposals in proposals_list]
        # 注意: (4000, 2, 4)
        proposals = self.codebox.decode(bbox_pred, proposals_list)
        # (4000, 2)
        pred_scores = F.softmax(cls_prob, -1)

        pred_boxes_list = proposals.split(num_proposals_per_image, dim=0)
        pred_scores_list = pred_scores.split(num_proposals_per_image, dim=0)

        boxes_all = []
        scores_all = []
        labels_all = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, list_images_size):
            boxes = clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(cls_prob.shape[-1], dtype=torch.int64, device=cls_prob.device).view(1, -1).expand_as(scores)
            # 剔除背景 0
            # ???
            # 去除第二个维度中索引0(背景)的预测信息
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 移除低概率目标
            keep = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # 移除小目标 
            keep = remove_small_boxes(boxes, min_size=1)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # 对不同的类别同时做nms处理
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # 获取scores排在前topk个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            boxes_all.append(boxes)
            scores_all.append(scores)
            labels_all.append(labels)
        return boxes_all, scores_all, labels_all
