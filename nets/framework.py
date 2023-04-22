from torch import nn
from nets.transforms import Transforms_Img_Tar
from nets.rpn import RPN_Head, AnchorsGenerator, RPN
from nets.roi_head import ROIHeads, TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

class FasterRCNN(nn.Module):
    def __init__(self,
                backbone=None, 
                # 
                min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], 
                # 
                in_channels=256, num_anchors=3,
                sizes = ((32,), (64,), (128,), (256,), (512,)), ratios = ((0.5, 1.0, 2.0),) * 5, 
                rpn_pre_nms_top_n=dict(train = 2000, test = 1000), 
                rpn_post_nms_top_n=dict(train = 2000, test = 1000), 
                rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, batch_size_per_image=256, positive_fraction=0.5,
                rpn_score_thresh=0.0, rpn_nms_thresh=0.7,
                # 
                featmap_names=['0', '1', '2', '3'], output_size=[7, 7], sampling_ratio=2,
                representation_size=1024, num_classes=2,
                fg_iou_thresh=0.5, bg_iou_thresh=0.5, 
                roi_batch_size_per_image=512, roi_positive_fraction=0.25, 
                box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100
                ):
        super(FasterRCNN, self).__init__()
        # 注意这里的self对象属性都会作为网络层加入model中
        # 
        self.backbone = backbone
        # 
        self.transform = Transforms_Img_Tar(min_size, max_size, image_mean, image_std)
        # 
        rpn_head = RPN_Head(in_channels, num_anchors)
        anchorsGenerator = AnchorsGenerator(sizes, ratios)
        self.rpn = RPN(rpn_head, anchorsGenerator, rpn_pre_nms_top_n, rpn_post_nms_top_n, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh, batch_size_per_image, positive_fraction, rpn_score_thresh, rpn_nms_thresh)
        # 
        box_roi_pool =MultiScaleRoIAlign(featmap_names,  # 在哪些特征层进行roi pooling
                                            output_size,sampling_ratio) 
        out_channels = self.backbone.out_channels
        resolution = box_roi_pool.output_size[0]
        box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes)
        self.roi_heads = ROIHeads(fg_iou_thresh, bg_iou_thresh, 
                        roi_batch_size_per_image, roi_positive_fraction, 
                        box_roi_pool, box_head, box_predictor, 
                        box_score_thresh, box_nms_thresh, box_detections_per_img)
        

    def forward(self, images, targets=None):
        # 训练模式下必须传入targets参数
        if self.training and targets == None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [(image.shape[-2:][0], image.shape[-2:][1]) for image in images]
        
        images, list_images_size, targets = self.transform(images, targets)
    
        features = self.backbone(images)
        proposals_list, rpn_losses = self.rpn(images, list_images_size, features, targets)
        detections, detector_losses  = self.roi_heads(features, proposals_list, list_images_size, targets)
        # 验证模式下
        detections = self.transform.postprocess(detections, list_images_size, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(rpn_losses)

        if self.training:
            return losses
        return detections