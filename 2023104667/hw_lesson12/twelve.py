import math
import sys
import torch

# stdout 改成 UTF-8 避免中文乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==========================================================
# 任务 1: Sinusoidal 正弦位置编码
# ==========================================================
def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """生成 (seq_len, d_model) 的正弦位置编码矩阵。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    assert d_model % 2 == 0, "d_model 必须为偶数"
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def add_pe_emb(emb: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """E + pos 注入:词嵌入直接加上位置编码。"""
    return emb + pe[: emb.shape[0]]


# ==========================================================
# 任务 2: 二维向量旋转
# ==========================================================
def rotate_2d(vec, theta: float) -> torch.Tensor:
    """对二维向量按角度 theta(弧度)逆时针旋转。

    [x']   [cos θ  -sin θ] [x]
    [y'] = [sin θ   cos θ] [y]
    """
    vec = torch.as_tensor(vec, dtype=torch.float32)
    c, s = math.cos(theta), math.sin(theta)
    x, y = vec[..., 0], vec[..., 1]
    return torch.stack([c * x - s * y,
                        s * x + c * y], dim=-1)


# ==========================================================
# 任务 3: 高维 RoPE (Rotary Position Embedding)
# ==========================================================
def precompute_rope_cos_sin(seq_len: int, d_model: int):
    """预计算每个位置、每个频率对应的 cos / sin。

    返回 cos, sin 形状均为 (seq_len, d_model/2)
    """
    assert d_model % 2 == 0, "d_model 必须为偶数"
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    angle = pos * freqs                  # (seq_len, d_model/2)
    return torch.cos(angle), torch.sin(angle)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对张量 x 的最后一维做 RoPE 旋转。

    x 形状 (..., seq_len, d_model);把最后一维相邻两两当作复数
    (x_2i, x_2i+1) 的实部/虚部,对每个位置乘以 e^{i·pos·θ_i}。
    """
    x1 = x[..., 0::2]                    # 偶数位
    x2 = x[..., 1::2]                    # 奇数位
    # 旋转:[x1', x2'] = [x1·cos - x2·sin, x1·sin + x2·cos]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # stack + flatten 还原成原来交错排列的形状
    out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
    return out


def rope_qk(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
    """高维 RoPE 接口:接收 Q、K 张量与位置序列,输出旋转后的 Q、K。

    q, k: (seq_len, d_model)
    positions: (seq_len,) 每个 token 在序列中的绝对位置
    """
    d_model = q.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    angle = positions.float().unsqueeze(1) * freqs       # (seq_len, d/2)
    cos, sin = torch.cos(angle), torch.sin(angle)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


# ==========================================================
# 任务 4: 两种位置编码的前向流程对比
# ==========================================================
def forward_with_additive_pe(emb: torch.Tensor,
                             Wq: torch.Tensor, Wk: torch.Tensor):
    """E + pos 流程:
        1) token_emb + PE  ── 位置信息在进入 attention 之前注入
        2) 用线性层得到 Q、K
        3) 直接做 QK^T 点积
    位置信息以"加法"的方式与内容混在一起,Q/K 同时携带绝对位置。
    """
    seq_len, d = emb.shape
    pe = sinusoidal_pe(seq_len, d)
    x = emb + pe                         # ← 位置在这里注入
    Q = x @ Wq
    K = x @ Wk
    return Q @ K.transpose(-1, -2)


def forward_with_rope(emb: torch.Tensor,
                      Wq: torch.Tensor, Wk: torch.Tensor):
    """RoPE 流程:
        1) token_emb 不加位置,直接得到 Q、K
        2) 对 Q、K 分别做位置相关的旋转
        3) 再做 QK^T 点积
    位置信息以"旋转(乘法)"方式作用在 Q、K 上,内积自然只与
    两个 token 的相对位置差有关。
    """
    seq_len, d = emb.shape
    Q = emb @ Wq                          # ← 先生成 Q、K
    K = emb @ Wk
    positions = torch.arange(seq_len)
    Q, K = rope_qk(Q, K, positions)       # ← 位置在这里注入(旋转)
    return Q @ K.transpose(-1, -2)


# ==========================================================
# 任务 5: 数值实验 —— 验证 RoPE 的相对位置不变性
# ==========================================================
def attention_score_additive(q_vec, k_vec, pos_q, pos_k, d_model):
    """E+pos 方式下的注意力分值:把同一个内容向量放在不同的位置上。"""
    pe = sinusoidal_pe(max(pos_q, pos_k) + 1, d_model)
    q = q_vec + pe[pos_q]
    k = k_vec + pe[pos_k]
    return torch.dot(q, k).item()


def attention_score_rope(q_vec, k_vec, pos_q, pos_k):
    """RoPE 方式下的注意力分值:对 q、k 分别按位置旋转后再点积。"""
    d_model = q_vec.shape[-1]
    positions = torch.tensor([pos_q, pos_k])
    qk = torch.stack([q_vec, k_vec], dim=0)
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    angle = positions.float().unsqueeze(1) * freqs
    cos, sin = torch.cos(angle), torch.sin(angle)
    rotated = apply_rope(qk, cos, sin)
    return torch.dot(rotated[0], rotated[1]).item()


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model = 8

    # 固定一对 q、k 的"内容"向量(语义相同),只让它们的位置变化
    q_vec = torch.randn(d_model)
    k_vec = torch.randn(d_model)

    # 两组位置,相对距离都 = 4
    pairs = [(1, 5), (3, 7), (0, 4)]

    print("=" * 56)
    print("任务 1: Sinusoidal PE 矩阵 (前 4 行, d=8)")
    print("=" * 56)
    print(sinusoidal_pe(4, 8))

    print("\n" + "=" * 56)
    print("任务 2: 二维向量旋转示例")
    print("=" * 56)
    v = torch.tensor([1.0, 0.0])
    for deg in [0, 90, 180, 270]:
        r = rotate_2d(v, math.radians(deg))
        print(f"  ({v.tolist()}) 旋转 {deg:>3}°  ->  ({r[0].item():+.4f}, {r[1].item():+.4f})")

    print("\n" + "=" * 56)
    print("任务 5: E+pos vs RoPE  注意力分值对比")
    print("(内容向量固定,只改位置;三组位置的相对距离都是 4)")
    print("=" * 56)
    print(f"{'位置 (pos_q, pos_k)':<22}{'E+pos 点积':<18}{'RoPE 点积':<18}")
    print("-" * 56)
    for pq, pk in pairs:
        s_add = attention_score_additive(q_vec, k_vec, pq, pk, d_model)
        s_rope = attention_score_rope(q_vec, k_vec, pq, pk)
        print(f"({pq}, {pk}){'':<16}{s_add:<+18.6f}{s_rope:<+18.6f}")

    print("\n结论:")
    print("  · E+pos 三组数值各不相同 —— 注意力分值依赖绝对位置")
    print("  · RoPE  三组数值完全一致 —— 注意力分值只依赖相对距离")

    # ------------------------------------------------------
    # 任务 6: 文字总结
    # ------------------------------------------------------
    print("\n" + "=" * 56)
    print("任务 6: RoPE 设计更优的原因(文字总结)")
    print("=" * 56)
    print(__doc_summary__ := """
1. 相对位置天然内蕴
   RoPE 通过对 Q、K 做位置相关的复数旋转,使得 <q_m, k_n> 仅依赖
   相对差 (m - n)。Transformer 真正关心的就是 token 之间的相对
   距离,RoPE 把这个先验直接写进了内积的代数结构里。

2. 不增加可学习参数,不破坏内积空间
   位置只作为正交旋转作用在 Q、K 上,向量长度不变,因此既不会
   稀释原始语义,又不会引入需要训练的位置参数,推理时只是查
   一次 cos/sin 表。

3. 外推能力更强
   E+pos 在训练长度之外的位置上 PE 没见过,直接相加会让 Q、K
   分布漂移;RoPE 只是把旋转角度延伸,语义子空间保持稳定,
   配合 NTK / YaRN 等扩展可平滑外推到更长上下文。

4. 与多头注意力天然兼容
   旋转在每个二维子空间独立进行,可逐头、逐子空间设置不同频
   率,与多头机制契合,实现高效。

5. 实验印证
   本作业中 E+pos 在 (1,5)、(3,7)、(0,4) 三对相对距离相同的位
   置上得到三个不同分值;RoPE 则得到完全一致的分值,直接验证
   了"相对位置不变"这一关键性质 —— 这正是 LLaMA、ChatGLM、
   Qwen 等主流大模型采用 RoPE 的核心原因。
""")


