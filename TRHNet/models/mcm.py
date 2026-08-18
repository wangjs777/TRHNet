import torch.nn
import torch.nn as nn


class MCM(nn.Module):
    def __init__(self, channels=2048):
        super(MCM, self).__init__()
        self.soft = torch.nn.Softmax(dim=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化可学习的权重参数 alpha
        # self.register_buffer(
        #     'alpha',
        #     torch.tensor(0.5)
        # )
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            bias=True
        )

    def forward(self, f1, f2):
        """
        f1:high-level
        f2:low-level
        """
        # similarity = torch.cosine_similarity(f1, f2, dim=1)
        b, c, h, w = f1.size()
        similarity = self.conv(f1 - f2)
        similarity = similarity.view(b, 1, h * w)
        similarity = self.soft(similarity)
        similarity_1 = 1.0 - similarity
        similarity_1 = self.soft(similarity_1)
        similarity = similarity.view(b, 1, h, w)
        similarity_1 = similarity_1.view(b, 1, h, w)
        # print(similarity.size())
        # 引入 alpha 作为加权系数
        f1_from_f2 = self.alpha * f2 * similarity
        f2_from_f1 = (1 - self.alpha) * f1 * similarity_1
        return f1_from_f2, f2_from_f1