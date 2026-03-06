import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA 微调层（适配器层）。

    给小白的直观理解：
    - 原模型参数非常大，直接全量训练成本很高。
    - LoRA 会在原有线性层旁边再加一条“低秩小支路”。
    - 训练时只更新这条小支路，原始大权重保持冻结。
    - 最终输出 = 原输出 + 小支路输出 * 缩放系数。

    公式：W_new = W_old + (B @ A) * scaling
    """

    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.05):
        super().__init__()

        # 保留原始预训练层，但冻结其权重（不参与梯度更新）。
        # 这样做可以显著降低显存占用和训练成本。
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False

        # 读取原始线性层的输入/输出维度。
        # d_in: 输入特征维度，d_out: 输出特征维度
        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # LoRA 的核心思想：先降维再升维。
        # A: [d_in, rank] 负责把高维信息压缩到低秩空间
        # B: [rank, d_out] 负责从低秩空间还原回输出维度
        self.lora_a = nn.Parameter(torch.zeros(d_in, rank))  # A 矩阵
        self.lora_b = nn.Parameter(torch.zeros(rank, d_out))  # B 矩阵

        # 缩放系数：控制 LoRA 分支对最终结果的影响力度。
        # 常见写法是 alpha / rank，保证不同 rank 下量级更稳定。
        self.scaling = alpha / rank

        # Dropout：训练时随机丢弃一部分输入，减少过拟合风险。
        self.dropout = nn.Dropout(p=dropout)

        # 初始化策略：A 随机、B 全零。
        # 这样初始时 LoRA 分支输出几乎为 0，不会破坏原模型已有能力。
        self.reset_parameters()

    def reset_parameters(self):
        # A 使用 Kaiming Uniform 初始化，便于后续学习。
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        # B 初始化为 0，保证刚开始不改变原模型输出。
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        # x 形状通常是 [batch_size, seq_len, d_in]
        
        # 路径 1：原始冻结层输出 y1
        original_output = self.original_layer(x)
        
        # 路径 2：LoRA 分支输出 y2 = (dropout(x) @ A) @ B
        # 注意：先把 x 转成和 LoRA 参数一致的数据类型，避免 dtype 不匹配。
        x_dropped = self.dropout(x)
        low_rank_output = (x_dropped.to(self.lora_a.dtype) @ self.lora_a) @ self.lora_b   

        # 最终输出：原模型输出 + LoRA 增量（乘缩放系数）。
        # 并转换回原输出 dtype，避免后续计算精度类型冲突。
        return original_output + (low_rank_output.to(original_output.dtype) * self.scaling)
