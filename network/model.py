import torch 
import torch.nn as nn
from utils import get_backbone

class Normalise(nn.Module):

    def __init__(self, power=2):
        super(Normalise, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class PreTrainNet(nn.Module):
    def __init__(self, cfg):
        super(PreTrainNet, self).__init__()
        trunk = cfg.MODEL.TRUNK
        fc_dim = cfg.MODEL.PRETRAINING.FC_DIM 

        self.backbone, backbone_channels = get_backbone(trunk)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_contrastive = nn.Linear(backbone_channels, fc_dim)
        self.l2norm = Normalise(2)

        self.fc_rotation_1 = nn.Linear(backbone_channels, 200)
        self.bn_rotation_1 = nn.BatchNorm1d(num_features = 200)
        self.fc_rotation_2 = nn.Linear(200, 200)
        self.bn_rotation_2 = nn.BatchNorm1d(num_features = 200)
        self.fc_rotation_3 = nn.Linear(200,4)



    def forward(self, x, rotation = False):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if rotation == False:
            x = self.fc_contrastive(x)
            x = self.l2norm(x)
            return x 
        elif rotation == True:
            x = self.fc_rotation_1(x)
            x = self.backbone.relu(x)
            x = self.bn_rotation_1(x)
            
            x = self.fc_rotation_2(x)
            x = self.backbone.relu(x)
            x = self.bn_rotation_2(x)

            x = self.fc_rotation_3(x)
            return x 
        else:
            raise ValueError('Rotation is either a Boolean True or False')

class FineTuneNet(nn.Module):
    def __init__(self, cfg):
        super(FineTuneNet, self).__init__()
        trunk = cfg.MODEL.TRUNK
        num_classes = cfg.MODEL.FINETUNING.NUM_CLASSES 
        assert num_classes > 0 and isinstance(num_classes, int), 'Please give a positive integer for the number of classes in the finetuning stage'
        p_dropout = cfg.MODEL.FINETUNING.DROPOUT

        self.backbone, backbone_channels = get_backbone(trunk)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p = p_dropout)
        self.fc = nn.Linear(backbone_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
