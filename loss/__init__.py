import torch
import torch.nn as nn 
from loss.NCECriterion import NCECriterion
from loss.NCEAverage import *


def get_loss(cfg, n_data):
    if cfg.LOSS.PRETRAINING.MINING == False:
        average = NCEAverageKinetics(
            feat_dim = cfg.MODEL.PRETRAINING.FC_DIM,
            n_data = n_data,
            K = cfg.LOSS.PRETRAINING.NCE.NEGATIVES,
            T = cfg.LOSS.PRETRAINING.NCE.TEMPERATURE,
            momentum = cfg.LOSS.PRETRAINING.NCE.MOMENTUM
        )
    else:
        average = NCEAverageKineticsMining(
            feat_dim = cfg.MODEL.PRETRAINING.FC_DIM,
            n_data = n_data,
            K = cfg.LOSS.PRETRAINING.NCE.NEGATIVES,
            T = cfg.LOSS.PRETRAINING.NCE.TEMPERATURE,
            max_hard_negatives_percentage = cfg.LOSS.PRETRAINING.MINING.MAX_HARD_NEGATIVES_PERCENTAGE
        )
    criterion = NCECriterion(n_data)

    TCELoss = NCELoss(average, criterion)
    RotationLoss = nn.CrossEntropyLoss()

    return TCELoss, RotationLoss

class NCELoss(nn.Module):
    def __init__(self, Average, Criterion):
        super(NCELoss, self).__init__()
        self.Average = Average 
        self.Criterion = Criterion 

    def forward(self, inputs, threshold):
        NCE_Average = self.Average(inputs, threshold)
        NCE_loss = self.Criterion(NCE_Average)
        return NCE_loss 