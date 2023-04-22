import torch
from tqdm import tqdm
from my_dataset import my_dataset
from transforms import Compose, To_tensor, RandomHorizontalSlip
from torch.utils.data import DataLoader
from nets.framework import FasterRCNN
from torch.optim import SGD
from backbones.resnet50_fpn_model import resnet50_fpn_backbone
from nets.roi_head import FastRCNNPredictor
# from torchvision.transforms.functional import to_pil_image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transforms = {
    "train": Compose([To_tensor(), RandomHorizontalSlip(0)]),
    "val": Compose([To_tensor()])
}

train_dataset = my_dataset(data_path="D:/VSCode_item/image_detection/Faster_RCNN/VOCdevkit/VOC2007/ImageSets/main/train.txt",
    xml_path="D:/vehicle_data/VOCdevkit/VOC2007/Annotations", 
    json_path="D:/VSCode_item/image_detection/Faster_RCNN/my_classes.json",
    image_path="D:/vehicle_data/VOCdevkit/VOC2007/JPEGImages", transforms=transforms["train"])

train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, pin_memory=True, collate_fn=train_dataset.my_collate)

backbone = resnet50_fpn_backbone(pretrain_path="./backbones/resnet50.pth",
                                     norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
model = FasterRCNN(backbone=backbone, num_classes=91)
weights_dict = torch.load("./backbones/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)
# 换predictor 
num_classes = 2
# 注意: 这里model.box_predictor错误的
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.to(device)
print(model.rpn.head.conv.weight.shape)

parameters = [para for para in model.parameters() if para.requires_grad]
optimizer = SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1e-4)
Epoch=2
for epoch in range(Epoch):
    pbar = tqdm(total=len(train_dataset.xml_list) // 4, desc=f'Epoch {epoch + 1}/{Epoch}', mininterval=0.3)
    for data in train_data_loader:
        images, targets = data
        
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        pbar.update()
    pbar.close()

    torch.save(model.state_dict(), "./faster_rcnn_{}.pth".format(epoch))

