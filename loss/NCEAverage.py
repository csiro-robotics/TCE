import math 
import torch 
import torch.nn as nn

class Normalise(nn.Module):

    def __init__(self, power=2):
        super(Normalise, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class NCEAverageKinetics(nn.Module):

    def __init__(self, feat_dim, n_data, K, T=0.07,  momentum=0.5):
        super(NCEAverageKinetics, self).__init__()
        self.nLem  = n_data
        self.unigrams = torch.ones(self.nLem)
        self.K = int(K)
        self.feat_dim = int(feat_dim)
        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))

        stdv = 1. / math.sqrt(feat_dim / 3)
        self.register_buffer('memory_bank', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.l2norm = Normalise(2)


    @torch.no_grad()
    def update_memorybank(self, features, index):
        with torch.no_grad():
            self.memory_bank.index_copy_(0, index.view(-1), features)


    def contrast(self, anchor_feature, pair_feature, negatives):
        Z_c = self.params[2].item()
        T = self.params[1].item()
        K = int(self.params[0].item())

        batchSize = anchor_feature.size(0)
        # ===== Retrieve negatives from memory bank, reshape, insert positive into 0th index =====
        weight_c = torch.index_select(self.memory_bank, 0, negatives.view(-1)).detach()
        weight_c = weight_c.view(batchSize, K + 1, self.feat_dim)
        weight_c[:,0] = pair_feature

        # ===== BMM between positive and negative features + positive pairing =====
        out_c = torch.bmm(weight_c, anchor_feature.view(batchSize, self.feat_dim, 1))
        out_c = torch.exp(torch.div(out_c, T))

        if Z_c < 0:
            self.params[2] = out_c.mean() * self.memory_bank.size(0)
            Z_c = self.params[2].clone().detach().item()
            print("normalization constant Z_c is set to {:.1f}".format(Z_c))

        out_c = torch.div(out_c, Z_c).contiguous()
        
        return out_c

    
    def forward(self, inputs, _ = None):

        anchor_feature = inputs['anchor_feature']
        pair_feature= inputs['pair_feature']
        negatives = inputs['negatives']
        membank_idx = inputs['membank_idx']

        out_c = self.contrast(anchor_feature, pair_feature, negatives)
        self.update_memorybank(anchor_feature, membank_idx)
   
        
        return out_c

class NCEAverageKineticsMining(nn.Module):
    def __init__(self, feat_dim, n_data, K, max_hard_negatives_percentage = 1, T=0.07):
        super(NCEAverageKineticsMining, self).__init__()
        self.nLem                   = n_data
        self.unigrams               = torch.ones(self.nLem)
        self.K                      = int(K)
        self.feat_dim              = int(feat_dim)
        self.max_negs               = int(self.K * max_hard_negatives_percentage)

        self.register_buffer('params', torch.tensor([K, T, -1]))

        stdv = 1. / math.sqrt(feat_dim / 3)
        self.register_buffer('memory_bank', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.l2norm = Normalise(2)

    def update_memorybank(self, features, index):
        self.memory_bank.index_copy_(0, index.view(-1), features)
        return None


    def contrast(self, anchor_feature, pair_feature, negatives):
        Z_c = self.params[2].item()
        T = self.params[1].item()
        K = int(self.params[0].item())
        batchSize = anchor_feature.size(0)

        # ===== Retrieve negatives from memory bank, reshape, insert positive into 0th index =====
        weight_c = torch.index_select(self.memory_bank, 0, negatives.view(-1)).detach()
        weight_c = weight_c.view(batchSize, K + 1, self.feat_dim)
        weight_c[:,0] = pair_feature

        # ===== BMM between positive and negative features + positive pairing =====
        out_c = torch.bmm(weight_c, anchor_feature.view(batchSize, self.feat_dim, 1))
        out_c = torch.exp(torch.div(out_c, T))

        if Z_c < 0:
            self.params[2] = out_c.mean() * self.memory_bank.size(0)
            Z_c = self.params[2].clone().detach().item()
            print("normalization constant Z_c is set to {:.1f}".format(Z_c))

        out_c = torch.div(out_c, Z_c).contiguous()
        
        return out_c

    def mine_negative_examples(self, anchor_feature, threshold, membank_idx):
        with torch.no_grad():
            bs                          = anchor_feature.size(0)
            cosine_scores               = torch.matmul(anchor_feature, self.memory_bank.view(self.feat_dim, -1))
            cosine_mask = cosine_scores >= threshold
            cosine_scores[cosine_mask] = -2
            
            scores, indices             = cosine_scores.topk(k=self.K)
            negative_indices            = torch.zeros(bs, self.K + 1).cuda()

            for rowid in range(bs):
                # Get all the hard examples, determine how many random examples are required
                row_topk                    = scores[rowid] # Top K values, sorted
                row_indices                 = indices[rowid] # Top K indices, associated
                row_idx                     = membank_idx[rowid]
                hard_examples                    = min((row_topk != -2).sum().item(), self.max_negs)
                rand_examples                    = self.K - hard_examples

                if hard_examples >= self.K:
                    # Enough hard examples to use for all negatives samples
                    negative_indices[rowid, 1:]    = row_indices
                elif hard_examples == 0:
                    probs                   = torch.ones(self.nLem)
                    probs[row_idx]          = 0
                    rand_indices            = torch.multinomial(probs, self.K, replacement = True)
                    negative_indices[rowid, 1:]    = rand_indices
                else:
                    # Mix of hard and random examples
                    negative_indices[rowid, 1:hard_examples + 1] = row_indices[:hard_examples]
                    probs                   = torch.ones(self.nLem)
                    probs[row_indices[:hard_examples]] = 0 # Don't sample hard negatives a second time
                    probs[row_idx]          = 0
                    rand_indices            = torch.multinomial(probs, self.K - hard_examples, replacement = True)
                    
                    negative_indices[rowid, hard_examples + 1:]    = rand_indices
            
            return negative_indices.long().cuda()

    def forward(self, inputs, threshold):

        anchor_feature = inputs['anchor_feature']
        pair_feature= inputs['pair_feature']
        membank_idx = inputs['membank_idx']
        
        negatives = self.mine_negative_examples(anchor_feature, threshold, membank_idx)
        out_c = self.contrast(anchor_feature, pair_feature, negatives)
        self.update_memorybank(anchor_feature, membank_idx)
   
        
        return out_c
