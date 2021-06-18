import torch.nn as nn
import torch.nn.functional as F


class JsdCrossEntropy(nn.Module):
    def __init__(self):
        super(JsdCrossEntropy, self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction="batchmean")

        return (0.5 * loss)
