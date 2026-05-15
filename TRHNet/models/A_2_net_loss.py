#import msadapter.pytorch as torch
#import msadapter.pytorch.nn as nn
import torch
import torch.nn as nn
#import mindspore as ms

class EndToEnd_A_2_net_Loss(nn.Module):
    def __init__(self, code_length, num_classes, margin=1):
        super(EndToEnd_A_2_net_Loss, self).__init__()
        self.code_length = code_length
        self.MSE_loss = nn.MSELoss()
        self.margin = margin

        # 为每个类别生成固定的 {-1, 1} 代理哈希中心，不参与梯度更新
        hash_centers = torch.empty(num_classes, code_length).bernoulli_().mul_(2).sub_(1)
        self.register_buffer('hash_centers', hash_centers)

    def forward(self, hash_features, targets):
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        targets = targets.long()

        # 代理哈希损失：拉近网络输出与对应类别的 {-1, 1} 代理中心
        batch_centers = self.hash_centers[targets]
        hash_loss = self.MSE_loss(hash_features, batch_centers) * self.margin
        return hash_loss
