"""Stage C 文本条件化训练配置。"""


class TextCondConfig:
    # 文本编码器
    vocab_size = 5000
    text_embed_dim = 128
    text_hidden_dim = 256
    text_num_heads = 4
    text_num_layers = 2
    max_text_len = 64

    # 融合方式
    fusion_type = 'film'  # 'film' 或 'cross_attention'
    semantic_dim = 256

    # 训练超参
    epochs = 30
    lr = 3e-4
    batch_size = 16
    warmup_epochs = 3
    weight_decay = 1e-4
    grad_clip = 1.0

    # 损失权重
    param_weight = 1.0
    consistency_weight = 0.1  # 有/无文本一致性
    base_align_weight = 0.05  # 与 Stage B 基线对齐

    # 数据
    instruction_data_path = 'data/instruction_data.json'
