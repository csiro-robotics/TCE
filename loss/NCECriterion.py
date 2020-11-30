import torch
from torch import nn

eps = 1e-7


class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):

        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz 


        return loss
        
class NCEAvgCriterion(nn.Module):
    def __init__(self, n_data):
        super(NCEAvgCriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):

        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).mean(0)) / bsz 


        return loss

class NCEMaxCriterion(nn.Module):
    def __init__(self, n_data):
        super(NCEMaxCriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):

        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        log_D0_max, log_D0_indexes = log_D0.view(-1, 1).max(0)

        loss = - (log_D1.sum(0) + log_D0_max) / bsz 


        return loss
        
class ContrastCriterion(nn.Module):
    def __init__(self, n_data):
        super(ContrastCriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):

        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.avg(torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps))).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz 


        return loss