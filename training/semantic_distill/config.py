"""语义蒸馏训练配置。"""


class DistillConfig:
    """Stage A / Stage B 默认超参。"""

    # 模型维度
    image_size = 224
    visual_dim = 384
    semantic_dim = 256
    text_dim = 384  # MiniLM-L6-v2 输出维度

    # Stage A: 语义对齐
    stage_a_epochs = 30
    stage_a_lr = 5e-4
    stage_a_batch_size = 32
    stage_a_warmup = 3

    # Stage B: 参数预测
    stage_b_epochs = 50
    stage_b_lr = 1e-4
    stage_b_batch_size = 16
    stage_b_warmup = 5
    stage_b_freeze_epochs = 17  # 前 N epochs 冻结 backbone
    stage_b_backbone_lr_scale = 0.1  # 解冻后 backbone 学习率倍率

    # 共通
    weight_decay = 1e-4
    grad_clip = 1.0
    scheduler = 'cosine'
    num_workers = 4

    # 退化增强 (v7)
    degrade_prob = 0.5
    consistency_weight = 0.1

    # 数据路径 (AutoDL)
    fivek_root = '/root/autodl-tmp/fivek_jpeg'
    embedding_path = '/root/autodl-tmp/data/fivek_text_embeddings.npz'
    expert_params_path = '/root/autodl-tmp/data/fivek_expert_params.json'
    checkpoint_dir = '/root/autodl-tmp/checkpoints'
