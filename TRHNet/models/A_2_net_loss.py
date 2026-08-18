import torch
import torch.nn as nn


class EndToEnd_A_2_net_Loss(nn.Module):
    # 【改动 1】：在初始化中引入 num_classes，用于生成代理中心
    def __init__(self, code_length, batch_size, num_classes, margin=1, finetune=False):
        super(EndToEnd_A_2_net_Loss, self).__init__()
        self.code_length = code_length
        self.MSE_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.margin = margin
        self.finetune = finetune

        # 【改动 2】：为每个类别随机生成固定的、值域在 {-1, 1} 的代理哈希中心 (Proxy Centers)
        # register_buffer 确保它会和模型一起保存在设备上，但不参与梯度更新
        hash_centers = torch.sign(torch.rand(num_classes, code_length) - 0.5)
        self.register_buffer('hash_centers', hash_centers)

    # 【改动 3】：清理前向传播参数。去掉 B, S, omega, deep_S，加入当前的标签 targets
    def forward(self, F, targets, dret, all_f, inputs):

        # 1. 保留：去相关损失 (Decorrelation Loss) - 约束特征位独立
        I = torch.eye(self.code_length, device=F.device)
        decorrelation_loss = torch.nn.functional.mse_loss(F.t() @ F, self.batch_size * I,
                                                          reduction='mean') / self.batch_size / self.code_length * self.margin

        # 2. 保留：GAN 损失 (GAN Loss) - 约束生成对抗器
        if self.finetune:
            gan_loss1 = torch.nn.functional.mse_loss(inputs['_img'], inputs['img']) * 0.1
            gan_loss2 = torch.nn.functional.mse_loss(inputs['_img2'], inputs['img']) * 0.1
            gan_loss = gan_loss1 + gan_loss2
        else:
            gan_loss = torch.nn.functional.mse_loss(inputs['_img'], inputs['img']) * 0.01

        # 3. 保留：重构损失 (Reconstruction Loss) - 约束特征重构
        if self.finetune:
            reconstruction_loss = self.MSE_loss(dret, all_f)
        else:
            with torch.no_grad():
                reconstruction_loss = self.MSE_loss(dret, all_f)

        # 4. 【核心替换】：代理哈希损失 (Proxy Hash Loss)
        # 直接拉近网络输出的连续特征 F 与它对应类别的 {-1, 1} 离散中心点
        batch_centers = self.hash_centers[targets]
        hash_loss = self.MSE_loss(F, batch_centers) * self.margin

        # 5. 总损失计算
        if self.finetune:
            loss = hash_loss + reconstruction_loss + decorrelation_loss + gan_loss
        else:
            loss = hash_loss + decorrelation_loss + gan_loss

        return loss, hash_loss, reconstruction_loss, decorrelation_loss, gan_loss