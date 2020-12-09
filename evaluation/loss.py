import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=15.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Find the pairwise distance or eucledian distance of two output feature vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        # perform contrastive loss calculation with the distance
        # print(euclidean_distance)
        # print(torch.clamp(self.margin - euclidean_distance, min=0.0))
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class HyLoss(nn.Module):
    def __init__(self, lbd):
        super(HyLoss, self).__init__()
        self.lbd = lbd
        self.CE = FocalLoss()
        self.margin = ContrastiveLoss()

    def forward(self, output1, output2, pred, target):
        """
        :param output1:
        :param output2:
        :param pred: should be raw logits for bce since it does softmax itself!!!
        :param target:
        :return:
        """
        # focal loss
        F_loss = self.CE(output1, output2, pred, target)
        # cmp loss
        target = target.float()
        y = 2 * target - 1
        MG_loss = self.margin(output1, output2, y.float())
        return self.lbd * F_loss + (1 - self.lbd) * MG_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, output1, output2, pred, target):
        # self.alpha = (target == 0).sum().item()/target.shape[0]
        target = target.float()
        # print(pred[1])
        BCE_loss = self.bce(pred[1], target)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        # alpha_tensor = self.alpha * target + (1 - self.alpha) * (1 - target)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        # alpha_tensor *
        return F_loss.mean()


class GHM(nn.Module):
    """GHM Classification Loss.
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(self, bins=10, momentum=0.9):
        super(GHM, self).__init__()
        self.bins = bins
        self.border = torch.arange(bins + 1).float().cuda() / bins
        self.border[-1] += 1e-6
        self.mmt = momentum
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()

    def forward(self, pred, target):
        target = target.float()
        GHM_weights = torch.zeros_like(pred)
        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)
        tot = max(target.shape[0], 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            g_in_bin = (g >= self.border[i]) & (g < self.border[i+1])
            num_in_bin = g_in_bin.sum().item()
            if num_in_bin > 0:
                if self.mmt > 0:
                    self.acc_sum[i] = self.mmt * self.acc_sum[i] + (1 - self.mmt) * num_in_bin
                    GHM_weights[g_in_bin] = tot / self.acc_sum[i]
                else:
                    GHM_weights[g_in_bin] = tot / num_in_bin
                n += 1
        if n > 0:
            GHM_weights = GHM_weights / n
        # alpha = (target == 0).sum().item() / target.shape[0]
        # alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        weights = GHM_weights
        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / tot
        return loss


class marginal_log_likelihood(nn.Module):
    """
    following E2E coref paper
    -- this likelihood is for positive samples
    -- prediction here is scores
    -- there exists the dumb node(score=0), but it's for ranking algorithm.
    -- we here is for pair-wise binary classification, it's not a proper loss
    """
    def __init__(self, reduction='mean'):
        super(marginal_log_likelihood, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        batch_loss = torch.tensor([0.]).to(device)
        input_now, tgt_now = prediction, target.float()
        blk_num = int(math.sqrt(input_now.shape[0]))
        chunked_ipt_now = [chunked.view(1, blk_num) for chunked in torch.chunk(input_now, blk_num, dim=0)]
        chunked_tgt_now = [chunked.view(1, blk_num) for chunked in torch.chunk(tgt_now, blk_num, dim=0)]
        ipt_scores, tgt_scores = torch.cat(chunked_ipt_now), torch.cat(chunked_tgt_now)
        # calculate loss
        gold_scores = ipt_scores + torch.log(tgt_scores)
        marginalized_gold_scores = gold_scores.logsumexp(dim=1)
        log_norm = ipt_scores.logsumexp(dim=1)
        smp_loss = torch.sum(log_norm - marginalized_gold_scores)
        if self.reduction == 'mean':
            smp_loss /= blk_num
        batch_loss += smp_loss
        return batch_loss
