# config/ipv6_mac.py

# ================= 硬件优化 (Mac M3) =================
device_type = 'mps'  # 使用 Apple Silicon GPU
compile = False      # MPS 目前对 torch.compile 支持不稳，必须关掉
grad_clip = 1.0      # 防止梯度爆炸

# ================= 数据参数 =================
dataset = 'base_data' # 对应 dataset.py 指向的目录
# 你的 Tokenizer 大小是 65536 + 几个特殊token。
# 我们设大一点，脚本会自动补齐到 64 的倍数。
vocab_size = 65545   

# ================= 模型架构 (Nano-6GPT) =================
# 关键设置！IPv6 地址只有 8 个段，加上 BOS/EOS 最多 10 个 token。
# 设置 32 绰绰有余。这能极大地节省显存，让你跑大 Batch。
max_seq_len = 32     

# 模型大小
# depth=6 (层数), n_head=6
# model_dim = depth * 64 = 384
depth = 6
n_head = 6
n_embd = 384  # 384 维度对 IPv6 足够了

# ================= 训练参数 =================
# 小杯子 (显存): M3 应该能轻松吞下 128 甚至 256，因为 seq_len 很短
device_batch_size = 256 

# 大水桶 (梯度累积): 
# 128 * 32 (seq) * 1 (gpu) = 4096 tokens per step
# 我们希望总 batch 大一点来稳定训练。
# 设为 65536 tokens 左右
total_batch_size = 8192 

num_iterations = 1000 

# 学习率: 这是一个全新的领域(不是微调)，LR 可以给劲一点
learning_rate = 1e-3 
min_lr = 1e-4

# 跑多久？
# 假设你有 5000万数据。跑 1个 epoch 大概需要很多步。
# 先跑 5000 步看看 Loss 下降情况。
max_iters = 5000

# ================= 杂项 =================
eval_every = 100
sample_every = 100   # 每 100 步生成几个 IP 看看
save_every = 500    # 每 500 步存个盘

core_metric_every = -1