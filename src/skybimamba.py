# -*- coding: utf-8 -*-
"""Sky-BiMamba 官方架构复现（Zhang et al., Renewable Energy 256 (2026) 124361）。

按论文 §2 逐组件实现，纳入本仓锁定协议（15 帧输入、15-min-ahead、kW 域）：
  - FrameStem（§2.2）：静态分支（当前帧）+ 动态分支（4 组帧差 Cloud Motion
    Difference Maps，Eq.2-3），双分支 CNN→concat→融合卷积（Eq.4-6），
    Sigmoid+L1 注意力 mask（Eq.7-8），空间均值池化 → T×D 图像特征（Eq.9）。
    论文未给出各分支通道配置（仅披露 stem 总参数 0.196M）；本实现 ~0.22M，
    记录为已知差异。
  - PV 分支（§2.3）：多尺度正弦时间编码 12 维（Eq.10-13）+ 过去 k=6 步滑窗
    功率 → Linear+GeLU（Eq.15）→ T×D。
  - 一阶段融合（§2.4.1）：逐步外积交互分 r_t=(I_t)^T P_t（Eq.16），双轴
    softmax 非对称归一化（Eq.17-18），残差互增强（Eq.19-20）。
  - BiMamba 时序层（§2.5）：每模态 12 层（敏感性分析最优值），每层前/后向
    各一个 S6 选择性状态空间块（conv 短程 + SSM 长程，Eq.28-33），
    双向平均（Eq.34）+ FC + 残差。S6 为标准 mamba-minimal 式纯 PyTorch
    实现（逐时间步扫描；序列长 15，开销可忽略）。
  - 二阶段融合（§2.4.2）：末步隐状态交叉增强（Eq.21-25）+ 标量门控
    自适应凸组合（Eq.26-27）→ 回归头。
与论文设置的差异（协议约束，全部记录）：
  (1) 输入 15 帧（协议）vs 论文 16 帧；
  (2) 训练配方用本仓锁定协议（Adam 3e-4/batch 256/plateau/早停 8），
      不用论文的 batch 16/lr 1e-4——与全部 baseline 同一协议保证公平；
  (3) kW 域目标（协议无归一化），论文为归一化功率；
  (4) 帧差图的 X_{t±1,t±2} 邻帧全部取自观测窗口内（边界 clamp 补零），
      无目标泄漏；论文帧差定义一致但未说明边界处理。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # CUDA 融合扫描核（数值同 S6 递推；无则回退纯 PyTorch 逐时间步扫描）
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _ssfn
except Exception:  # pragma: no cover
    _ssfn = None


# ---------------------------------------------------------------- FrameStem
class _ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, pool):
        super().__init__()
        layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                  nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrameStem(nn.Module):
    """双分支（静态/动态）轻量 CNN + 注意力 mask（论文 §2.2，Eq.2-9）。"""

    def __init__(self, d_out=128):
        super().__init__()
        self.static = nn.Sequential(          # 64→4（4 次 /2）
            _ConvBNAct(3, 24, True), _ConvBNAct(24, 32, True),
            _ConvBNAct(32, 48, True), _ConvBNAct(48, 64, True))
        self.dynamic = nn.Sequential(         # 12 通道差分图
            _ConvBNAct(12, 24, True), _ConvBNAct(24, 32, True),
            _ConvBNAct(32, 48, True), _ConvBNAct(48, 64, True))
        self.fusion = nn.Sequential(
            nn.Conv2d(128, d_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_out), nn.ReLU(inplace=True))
        self.d_out = d_out

    @staticmethod
    def diff_maps(x):
        """x:(B,T,3,H,W) → 4 组帧差 concat (B,T,12,H,W)，Eq.2-3。
        边界 clamp：窗口首尾差分置零（无窗口外信息）。"""
        T = x.shape[1]
        idx = torch.arange(T, device=x.device)
        c2 = idx.clamp(min=2, max=T - 1)
        c1 = idx.clamp(min=1, max=T - 1)
        m1 = idx.clamp(max=T - 2)
        m2 = idx.clamp(max=T - 3)
        d1 = x[:, c2] - x[:, c1]
        d2 = x[:, c1] - x
        d3 = x - x[:, m1]
        d4 = x[:, m1] - x[:, m2]
        return torch.cat([d1, d2, d3, d4], dim=2)

    def forward(self, x):
        B, T = x.shape[:2]
        static = self.static(x.reshape(B * T, *x.shape[2:]))
        dynamic = self.dynamic(self.diff_maps(x).reshape(B * T, 12, *x.shape[3:]))
        fused = self.fusion(torch.cat([static, dynamic], dim=1))   # Eq.6
        sig = torch.sigmoid(fused)
        n = sig.shape[-2] * sig.shape[-1]                            # (H/16)(W/16)
        mask = n * sig / (2.0 * sig.flatten(1).sum(dim=1)
                           .clamp(min=1e-6).view(-1, 1, 1, 1))      # Eq.7
        out = (fused * mask).mean(dim=(2, 3))                        # Eq.8-9
        return out.reshape(B, T, self.d_out)


# ------------------------------------------------------------- PV 分支（§2.3）
class PVEmbed(nn.Module):
    """时间编码(12) + 过去 k 步功率 → FC+GeLU → (B,T,D)，Eq.10-15。"""

    def __init__(self, k=6, d_out=128):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(12 + k, d_out)

    def forward(self, pv_log, tf):
        # tf: (B,T,12) 时间编码；pv 滑窗（首部复制填充）
        B, T = pv_log.shape
        idx = (torch.arange(T, device=pv_log.device).unsqueeze(1)
               - torch.arange(self.k - 1, -1, -1, device=pv_log.device)
               ).clamp(min=0)                                        # (T,k)
        win = pv_log.gather(1, idx.unsqueeze(0).expand(B, -1, -1)
                            .reshape(B, T * self.k)).reshape(B, T, self.k)
        x = torch.cat([win, tf], dim=2)
        return F.gelu(self.fc(x))


# ------------------------------------------------- 一阶段交叉注意力（§2.4.1）
class StepCrossModalAttention(nn.Module):
    """逐步外积交互 + 双轴 softmax + 残差互增强，Eq.16-20。"""

    @staticmethod
    def forward_step(i, p):
        # i,p: (..., D) → 增强后的 (i', p')
        r = torch.einsum("...m,...n->...mn", i, p)                   # Eq.16
        w1 = torch.softmax(r, dim=-2)                                # Eq.17
        w2 = torch.softmax(r, dim=-1)                                # Eq.18
        return i + torch.einsum("...mn,...n->...m", w1, p), \
               p + torch.einsum("...mn,...m->...n", w2, i)           # Eq.19-20


# ------------------------------------------------------------- S6 Mamba 块
class MambaBlock(nn.Module):
    """标准选择性 SSM 块（S6，mamba-minimal 式纯 PyTorch 扫描）。
    结构对应论文 §2.5：conv1d 提取短程模式 + SSM 长程依赖 + 门控输出。"""

    def __init__(self, d_model, expand=2, d_conv=4, d_state=16):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_in = F.silu(x_in)
        x_dbl = self.x_proj(x_in)                    # (B,L,dt_rank+2N)
        dt, Bp, Cp = torch.split(
            x_dbl, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))            # (B,L,d_inner)
        A = -torch.exp(self.A_log)                   # (d_inner,N)
        if _ssfn is not None and x.is_cuda:
            # autocast 下 u/dt 可能 bf16-fp32 混杂，统一 fp32 入核（核内本就
            # fp32 状态递推），输出回到原 dtype
            y = _ssfn(x_in.transpose(1, 2).float().contiguous(),  # u (B,d,L)
                      dt.transpose(1, 2).float().contiguous(),    # delta (B,d,L)
                      A.float().contiguous(),                     # (d,N)
                      Bp.transpose(1, 2).float().contiguous(),    # (B,N,L)
                      Cp.transpose(1, 2).float().contiguous(),    # (B,N,L)
                      self.D.float()).to(x_in.dtype)              # (B,d,L)
            y = y.transpose(1, 2)
        else:
            h = x_in.new_zeros(B, self.d_inner, self.d_state)
            ys = []
            for t in range(L):
                dA = torch.exp(dt[:, t].unsqueeze(-1) * A)               # (B,d,N)
                dBx = dt[:, t].unsqueeze(-1) * Bp[:, t].unsqueeze(1) * \
                    x_in[:, t].unsqueeze(-1)                             # (B,d,N)
                h = dA * h + dBx
                ys.append(torch.einsum("bdn,bn->bd", h, Cp[:, t]) + self.D * x_in[:, t])
            y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class BiMambaLayer(nn.Module):
    """双向 Mamba 层：前/后向平均（Eq.34）+ FC + 残差（论文 §2.5）。"""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = MambaBlock(d_model)
        self.bwd = MambaBlock(d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        z = self.norm(x)
        y = 0.5 * (self.fwd(z) + self.bwd(torch.flip(z, dims=[1])).flip(dims=[1]))
        return x + self.fc(y)


# ---------------------------------------------------------------- 完整模型
class SkyBiMamba(nn.Module):
    """输入协议与本仓 ForecastDataset 对齐：
    forward(x_img, pv_log, tf) → PV(kW)。x_img (B,15,3,64,64)∈[0,1]，
    pv_log (B,15) kW，tf (B,15,12) 时间编码。"""

    def __init__(self, d_model=128, n_layers=12, pv_k=6, dropout=0.1):
        super().__init__()
        self.stem = FrameStem(d_out=d_model)
        self.pv_embed = PVEmbed(k=pv_k, d_out=d_model)
        self.img_mamba = nn.ModuleList([BiMambaLayer(d_model)
                                        for _ in range(n_layers)])
        self.pv_mamba = nn.ModuleList([BiMambaLayer(d_model)
                                       for _ in range(n_layers)])
        self.gate_fc = nn.Linear(2 * d_model, 1)                     # Eq.26
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 64),
                                  nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x_img, pv_log, tf):
        I = self.stem(x_img)                                         # (B,T,D)
        P = self.pv_embed(pv_log, tf)                                # (B,T,D)
        I, P = StepCrossModalAttention.forward_step(I, P)            # 一阶段 Eq.16-20
        for layer in self.img_mamba:
            I = layer(I)
        for layer in self.pv_mamba:
            P = layer(P)
        i_f, p_f = I[:, -1], P[:, -1]                                # 末步隐状态
        r = torch.einsum("bm,bn->bmn", i_f, p_f)                     # Eq.21
        w3 = torch.softmax(r, dim=-2)
        w4 = torch.softmax(r, dim=-1)                                # Eq.22-23
        i_e = i_f + torch.einsum("bmn,bn->bm", w3, p_f)              # Eq.24
        p_e = p_f + torch.einsum("bmn,bm->bn", w4, i_f)              # Eq.25
        gate = torch.sigmoid(self.gate_fc(torch.cat([i_e, p_e], dim=1))
                             ).squeeze(-1)                           # Eq.26
        out = (1 - gate).unsqueeze(-1) * i_e + gate.unsqueeze(-1) * p_e  # Eq.27
        return self.head(out).squeeze(-1)
