# -*- coding: utf-8 -*-
"""Baseline 模型组（PyTorch）。分层设计回答"每个信息源贡献多少"：
  - pv-only：MLP / GRU —— 不看天空图像
  - SUNSET-CNN：官方架构复现（15 帧拼 45 通道 + pv_log）
后续新模块一律在此之上做加法，协议不变。
"""
import torch
import torch.nn as nn


def frames_to_channels(x_img):
    """(B,n,3,64,64) -> (B,3n,64,64)：帧序列拼成多通道单图（官方 SUNSET 方式）。"""
    B, n = x_img.shape[:2]
    return x_img.reshape(B, n * 3, 64, 64)


class _ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = (nn.Identity() if c_in == c_out and stride == 1 else
                     nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                   nn.BatchNorm2d(c_out)))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + self.skip(x))


class SunsetCNN(nn.Module):
    """官方 SUNSET forecast 架构的 PyTorch 复现
    （Conv24-BN-Pool, Conv48-BN-Pool, concat pv_log, Dense1024 x2, out1）。"""

    def __init__(self, pv_len=15, img_frames=15, width=64):
        super().__init__()
        c_in = img_frames * 3
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        flat = 48 * (width // 4) ** 2
        self.head = nn.Sequential(
            nn.Linear(flat + pv_len, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 1),
        )

    def forward(self, x_img, x_pv):
        x = self.conv(frames_to_channels(x_img))
        x = torch.flatten(x, 1)
        x = torch.cat([x, x_pv], dim=1)
        return self.head(x).squeeze(-1)


class PVMLP(nn.Module):
    """只用 15 个历史 PV 值。"""

    def __init__(self, pv_len=15, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pv_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_img, x_pv):
        return self.net(x_pv).squeeze(-1)


class PVGRU(nn.Module):
    """pv-only 序列模型：GRU 读 15 步功率历史。"""

    def __init__(self, pv_len=15, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_img, x_pv):
        x = x_pv.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


class _PlainResBlock(nn.Module):
    """无 BN 的 ResBlock（Cambridge ConvLSTM 官方配置：Normalization=None）。"""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = (nn.Identity() if c_in == c_out and stride == 1 else
                     nn.Conv2d(c_in, c_out, 1, stride=stride))

    def forward(self, x):
        y = torch.relu(self.conv1(x))
        y = self.conv2(y)
        return torch.relu(y + self.skip(x))


class ConvLSTMCell(nn.Module):
    """标准 ConvLSTM 单元（Shi et al. 2015，无 peephole）。"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size // 2)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, B, H, W, device):
        return (torch.zeros(B, self.hidden_dim, H, W, device=device),
                torch.zeros(B, self.hidden_dim, H, W, device=device))


class SkyConvLSTM(nn.Module):
    """Cambridge ConvLSTM 官方架构复现（Nie 2022 arXiv:2211.02108 §4.1.2，
    起源 Paletta et al.）。双支路：
      图像支路：帧间共享 conv+6×PlainResBlock（stride 降采样 64→8，输出 8×8×32）
                → ConvLSTM(32→64) 逐帧时序 → 取末状态 → Dense512→64
      辅助支路：pv 历史 → Dense16×3 → LSTM(16) → 末步输出
      concat → Linear→1。官方无 BN/pooling/dropout（表 4.1），4.25M 参数。
    官方为 8 帧(2-min)/8 PV；本协议 15 帧(1-min)/15 PV，架构不变仅序列变长。
    通道配置按论文图 4.1(B) 推断（本实现 ~2.6M，小于官方，记录为已知差异）。
    """

    def __init__(self, pv_len=15, img_frames=15, width=64):
        super().__init__()
        self.nf = img_frames
        self.hw = width // 8
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1), nn.ReLU(),
            _PlainResBlock(8, 32, stride=2),
            _PlainResBlock(32, 32),
            _PlainResBlock(32, 64, stride=2),
            _PlainResBlock(64, 64),
            _PlainResBlock(64, 32, stride=2),
            _PlainResBlock(32, 32),
        )
        self.cell = ConvLSTMCell(32, 64)
        self.img_head = nn.Sequential(
            nn.Linear(64 * self.hw * self.hw, 512), nn.ReLU(),
            nn.Linear(512, 64))
        self.aux = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU())
        self.lstm = nn.LSTM(16, 16, batch_first=True)
        self.out = nn.Linear(64 + 16, 1)

    def forward(self, x_img, x_pv):
        B = x_img.shape[0]
        x = x_img.reshape(B * self.nf, 3, x_img.shape[-2], x_img.shape[-1])
        f = self.spatial(x).reshape(B, self.nf, 32, self.hw, self.hw)
        h, c = self.cell.init_state(B, self.hw, self.hw, x_img.device)
        for t in range(self.nf):
            h, c = self.cell(f[:, t], (h, c))
        z_img = self.img_head(torch.flatten(h, 1))
        z_aux, _ = self.lstm(self.aux(x_pv.unsqueeze(-1)))
        return self.out(torch.cat([z_img, z_aux[:, -1]], dim=1)).squeeze(-1)


class ConvGRUCell(nn.Module):
    """标准 ConvGRU 单元（近帧状态流用，5 步短递归）。"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_zr = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim,
                                 kernel_size, padding=kernel_size // 2)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim,
                                kernel_size, padding=kernel_size // 2)

    def forward(self, x, h):
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = torch.chunk(zr, 2, dim=1)
        h_hat = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_hat


class AWABranch(nn.Module):
    """AWA v2 编码器（Advection-Warped Aggregation，平流外推聚合，自创 2026-10-06）。

    Failure-driven 依据（results/encoder_diag.json，推理级诊断）：
    ConvLSTM 与静态 CNN 的 B 分支遮掉窗口前 10 帧损伤均 <0.012 kW，而遮最近
    5 帧 +0.07~0.14——15 帧里 2/3 时程（云平流历史）被浪费，近 5 帧是主信息源。
    v1（纯 warp 聚合，s0 test 2.964 vs CLB 2.904）证明只补运动外推、牺牲近帧
    状态处理会输：v1 晴天 0.356 vs CLB 0.263。

    v2 = 双时间尺度读出（两条流各对应诊断的一个时间尺度，非模块堆叠）：
      A 近帧状态流：末 5 帧特征 5 步 ConvGRU → 末状态 h_near
        （"现在天空什么状态"——诊断 occ_last5 主损伤源）
      B 平流外推流：全部 15 帧 warp 到预测时刻 t+15 参考系后聚合
        （"云往哪走"——诊断 occ_first10≈0 的被浪费信息）
      g_t(p) = f_t(p − s_t·v(p))，s_t=(14−t)+15 分钟；
      v 由相邻特征差回归（±0.5 格/min，零初始化=恒等起步，总位移限幅 ±8 格）。
    两流 concat → 空间压缩 → enc_b。空间编码 4×ResBlock 到 16×16（对齐 CLB
    容量、保留 warp 分辨率）。参数 ~0.86M（CLB 0.99M）。

    消融开关：align=False（位移置零=纯时间平均聚合）、
    extrap=False（warp 到 t=14 当前系而非 t+15）、near=False（去近帧流=v1+深spatial）。
    ms=True（L3a 多尺度读出）：mean‖last 扩为 mean‖last3mean‖last‖trend
    （trend = 末3帧均值 − 前12帧均值）——爬坡前兆是短时趋势，末帧单点噪声大。
    sa=True（L3b 空间注意力读出）：末帧特征经可学习空间 softmax 聚焦池化，
    显式保留"热点/日盘邻域"局部高分辨率信息（爬坡=云边缘扫过日盘的局部
    事件，全局 mean 池化稀释之）。零位移先验、零标定，与 warp 证伪链不冲突。
    """

    def __init__(self, pv_len=15, c_spatial=32, size=16, align=True, extrap=True,
                 near=True, ms=False, sa=False):
        super().__init__()
        self.nf = 15
        self.n_near = 5
        self.size = size
        self.align = align
        self.extrap = extrap
        self.near = near
        self.ms = ms
        self.sa = sa
        C = c_spatial
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            _PlainResBlock(8, C, stride=2),      # 32
            _PlainResBlock(C, C),
            _PlainResBlock(C, C, stride=2),      # 16
            _PlainResBlock(C, C),
        )
        self.motion = nn.Conv2d(2 * C, 2, 3, padding=1)
        nn.init.zeros_(self.motion.weight)        # 零位移起步 = 恒等 warp
        nn.init.zeros_(self.motion.bias)
        self.near_cell = ConvGRUCell(C, 48)
        n_read = 2 + (2 if ms else 0) + (1 if sa else 0)
        n_cat = n_read * C + (48 if near else 0)
        if sa:
            self.sa_conv = nn.Conv2d(C, 1, 1)
        self.post = nn.Sequential(
            nn.Conv2d(n_cat, 64, 3, stride=2, padding=1), nn.ReLU(),  # 8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # 基础采样网格（预测参考系坐标 p，归一 [-1,1]，align_corners=True）
        ys, xs = torch.linspace(-1, 1, size), torch.linspace(-1, 1, size)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1)
        self.register_buffer("base_grid", grid.view(1, 1, size, size, 2))
        # 每帧回溯步数（分钟）：t=0..14 → s=29..15；noextrap → s=14..0
        s = torch.arange(self.nf - 1, -1, -1, dtype=torch.float32) + (15 if extrap else 0)
        self.register_buffer("steps", s.view(1, self.nf, 1, 1, 1))
        step_norm = 2.0 / (size - 1)              # 格 → 归一坐标
        self.scale = 0.5 * step_norm              # |v|≤0.5 格/步
        self.max_disp = 8.0 * step_norm

    def forward(self, x_img):
        B, n = x_img.shape[0], self.nf
        S = self.size
        f = self.spatial(x_img.reshape(B * n, 3, x_img.shape[-2], x_img.shape[-1]))
        C = f.shape[1]
        f = f.reshape(B, n, C, S, S)
        if self.align:
            d = self.motion(torch.cat([f[:, :-1], f[:, 1:]], dim=2)
                            .reshape(B * (n - 1), 2 * C, S, S))
            v = torch.tanh(d).reshape(B, n - 1, 2, S, S).mean(dim=1)  # (B,2,S,S) 平均步速
            disp = v.unsqueeze(1) * self.steps * self.scale           # (B,n,2,S,S)
            disp = self.max_disp * torch.tanh(disp / self.max_disp)   # 限幅
            grid = self.base_grid - disp.permute(0, 1, 3, 4, 2)       # 回溯采样坐标
            g = torch.nn.functional.grid_sample(
                f.reshape(B * n, C, S, S), grid.reshape(B * n, S, S, 2),
                mode="bilinear", padding_mode="border", align_corners=True)
            g = g.reshape(B, n, C, S, S)
        else:
            g = f
        parts = [g.mean(dim=1), g[:, -1]]                              # 外推流读出
        if self.ms:                                                    # 多尺度短窗 + 趋势
            lm = g[:, -3:].mean(dim=1)
            parts = parts + [lm, lm - g[:, :-3].mean(dim=1)]
        if self.sa:                                                    # 末帧空间聚焦池化
            att = torch.softmax(self.sa_conv(g[:, -1]).flatten(2), dim=2)
            focus = (g[:, -1].flatten(2) * att).sum(dim=2)             # (B,C)
            parts.append(focus[:, :, None, None].expand(
                -1, -1, g.shape[-2], g.shape[-1]))                     # 广播为 (B,C,S,S)
        if self.near:                                                  # 近帧状态流
            h = torch.zeros(B, self.near_cell.hidden_dim, S, S,
                            device=f.device)
            for t in range(n - self.n_near, n):
                h = self.near_cell(f[:, t], h)
            parts.append(h)
        z = self.post(torch.cat(parts, dim=1))
        return torch.flatten(z, 1)                                     # (B,64*8*8)


def build(name):
    return {"mlp": PVMLP, "gru": PVGRU, "sunset": SunsetCNN,
            "convlstm": SkyConvLSTM, "reslstm": ResLSTM,
            "transformer": SkyTransformer, "patchtst": PatchTST,
            "dlinear": DLinear, "crossgnn": CrossGNN}[name]()


class PGF(nn.Module):
    """Physics-Gated Fusion v2 —— 组件由误差分解与消融证据驱动：

    1) 双分支互补（oracle 择优比最好单模型低 12.3%）：
       A=GRU(pv_log, kW 域)——消融证据：A 在 kW 域晴天 0.287，CSI 化反而
       0.525（CSI 域丢失功率水平信息），故 A 保持 kW 域；
       B=CNN(15帧)+pv_log，输出经 CSI 物理参数化（csi_B×P_theo_next）——
       证据：SUNSET 晴天 +0.183kW 系统 bias 占 MAE 39%，物理锚点可校正；
    2) 学习门控（简单方差门控/均值融合无收益）：输入=窗口 CSI/PV 统计
       + 两分支隐特征，逐样本权重。
    消融开关：use_csi（B 是否物理参数化）/ use_gate / use_image。
    """

    def __init__(self, use_csi=True, use_gate=True, use_image=True, use_fd=False,
                 use_deep=False, pv_len=15, use_cl_b=False,
                 use_awa_b=False, awa_align=True, awa_extrap=True, awa_near=True,
                 tpa_ms=False, tpa_sa=False):
        super().__init__()
        self.use_csi = use_csi
        self.use_gate = use_gate
        self.use_image = use_image
        self.use_fd = use_fd  # B 分支帧差分通道（云运动显式信号）
        self.use_deep = use_deep  # B 分支 ResNet 式加深（cloudy 容量瓶颈证据）
        self.use_cl_b = use_cl_b  # B 分支换 ConvLSTM 时空编码（阶段三，源自 ConvLSTM 复现）
        self.use_awa_b = use_awa_b  # B 分支换 AWA 平流外推聚合编码（自创，阶段四）
        # 分支 A：时序（kW 域）
        self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
        self.head_a = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        # 分支 B：图像
        if use_awa_b:
            self.awa = AWABranch(pv_len=pv_len, align=awa_align, extrap=awa_extrap,
                                 near=awa_near, ms=tpa_ms, sa=tpa_sa)
            self.enc_b = nn.Linear(64 * 8 * 8 + pv_len, 128)
        elif use_cl_b:
            # 帧间共享空间编码 + ConvLSTM 时序（官方 Cambridge 架构同款，
            # 复现单模型/集成均强于 SUNSET 复现 → 引入本模型）
            self.spatial = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=1, padding=1), nn.ReLU(),
                _PlainResBlock(8, 32, stride=2),
                _PlainResBlock(32, 32),
                _PlainResBlock(32, 64, stride=2),
                _PlainResBlock(64, 64),
                _PlainResBlock(64, 32, stride=2),
                _PlainResBlock(32, 32),
            )
            self.cell = ConvLSTMCell(32, 64)
            self.enc_b = nn.Linear(64 * 8 * 8 + pv_len, 128)
        else:
            c_in = 87 if use_fd else 45
            if use_deep:
                self.conv = nn.Sequential(  # stem + 4×res block, 输出 96×8×8
                    nn.Conv2d(c_in, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
                    _ResBlock(24, 24, stride=2),
                    _ResBlock(24, 48, stride=2),
                    _ResBlock(48, 96, stride=2),
                )
                self.enc_b = nn.Linear(96 * 8 * 8 + pv_len, 128)
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(c_in, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                self.enc_b = nn.Linear(48 * 16 * 16 + pv_len, 128)
        self.head_b = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # 门控：pv/CSI 双域统计 + 分支隐特征
        n_phys = 8  # pv: var,mean|Δ|,last; csi: var,mean|Δ|,last; p_theo_next, pv_last/pt_next
        self.gate = nn.Sequential(
            nn.Linear(n_phys + 64 + 128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_img, pv_log, csi_log, p_theo_next, return_parts=False):
        B = pv_log.shape[0]
        # 分支 A（kW 域）
        out, _ = self.gru(pv_log.unsqueeze(-1))
        h_a = out[:, -1]
        f_a = self.head_a(h_a).squeeze(-1)  # kW
        # 门控物理特征（双域统计）
        feats = torch.stack([pv_log.var(dim=1),
                             pv_log.diff(dim=1).abs().mean(dim=1),
                             pv_log[:, -1],
                             csi_log.var(dim=1),
                             csi_log.diff(dim=1).abs().mean(dim=1),
                             csi_log[:, -1],
                             p_theo_next,
                             pv_log[:, -1] / torch.clamp(p_theo_next, min=0.05)],
                            dim=1)
        # 分支 B
        if self.use_image:
            if self.use_awa_b:
                x = self.awa(x_img)
            elif self.use_cl_b:
                B_size, nf = x_img.shape[0], x_img.shape[1]
                x = x_img.reshape(B_size * nf, 3, 64, 64)
                f = self.spatial(x).reshape(B_size, nf, 32, 8, 8)
                h, c = self.cell.init_state(B_size, 8, 8, x_img.device)
                for t in range(nf):
                    h, c = self.cell(f[:, t], (h, c))
                x = torch.flatten(h, 1)
            else:
                if self.use_fd:
                    fd = x_img[:, 1:] - x_img[:, :-1]  # (B,14,3,64,64) 帧差分=云运动
                    x_in = torch.cat([x_img, fd], dim=1)  # (B,29,3,64,64)
                else:
                    x_in = x_img
                x = self.conv(frames_to_channels(x_in))
                x = torch.flatten(x, 1)
            h_b = torch.relu(self.enc_b(torch.cat([x, pv_log], dim=1)))
            f_b = self.head_b(h_b).squeeze(-1)
            if self.use_csi:  # 物理参数化：B 预测 CSI，乘理论功率回 kW
                f_b = f_b * p_theo_next
        else:
            h_b = torch.zeros(B, 128, device=pv_log.device)
            f_b = torch.zeros(B, device=pv_log.device)
        # 门控融合（kW 域）
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([feats, h_a, h_b], dim=1)).squeeze(-1))
        else:
            g = torch.full((B,), 0.5, device=pv_log.device)
        f = g * f_b + (1 - g) * f_a
        if return_parts:
            return f, g, f_a.detach(), f_b.detach()
        return f


# ----------------------------------------------------------------------
# 范文对齐 baseline 组（2026-10-12 新增，Renewable Energy 256 (2026)
# 124361 §3.3.1：ResLSTM / Transformer 多模态 + PatchTST / DLinear /
# CrossGNN 单模态时间序列；适配本协议 15 帧 1-min 输入 + t+H 单步目标）
# ----------------------------------------------------------------------
class ResLSTM(nn.Module):
    """范文 ResLSTM 复现：ResNet 式 CNN 逐帧提特征 → LSTM 时序建模；
    PV 历史走独立 LSTM 辅助支路，concat 两支路末状态出预测。
    （范文用 ResNet-18 + LSTM；此处同结构小型化，64×64 输入下等价层级。
    显存工程：帧维分块 + 梯度检查点，数学上与整批前向完全等价。）"""

    def __init__(self, pv_len=15, img_frames=15, width=64):
        super().__init__()
        self.nf = img_frames
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            _PlainResBlock(16, 32, stride=2),
            _PlainResBlock(32, 32),
            _PlainResBlock(32, 64, stride=2),
            _PlainResBlock(64, 128),
        )
        self.lstm_img = nn.LSTM(128, 128, batch_first=True)
        self.lstm_pv = nn.LSTM(1, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128 + 32, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def _frame_feat(self, x_img):
        from torch.utils.checkpoint import checkpoint
        B, n = x_img.shape[0], self.nf
        flat = x_img.reshape(B * n, 3, x_img.shape[-2], x_img.shape[-1])
        outs = []
        for c in flat.split(384):                     # 每块 384 帧
            outs.append(checkpoint(self.cnn, c, use_reentrant=False)
                        if self.training else self.cnn(c))
        return torch.cat(outs)

    def forward(self, x_img, x_pv):
        B, n = x_img.shape[0], self.nf
        f = self._frame_feat(x_img)
        z = f.mean(dim=(2, 3)).reshape(B, n, 128)          # GAP → 帧特征
        o_img, _ = self.lstm_img(z)
        o_pv, _ = self.lstm_pv(x_pv.unsqueeze(-1))
        return self.head(torch.cat([o_img[:, -1], o_pv[:, -1]], dim=1)).squeeze(-1)


class SkyTransformer(nn.Module):
    """范文 Transformer 复现：CNN 逐帧特征 + PV 值共同作为 token，
    Transformer encoder 建模长程时序依赖；mean‖last 池化出预测。
    （范文：ResNet-18 图像特征 + Transformer 时序。）"""

    def __init__(self, pv_len=15, img_frames=15, width=64, d=128):
        super().__init__()
        self.nf = img_frames
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            _PlainResBlock(16, 32, stride=2),
            _PlainResBlock(32, 32),
            _PlainResBlock(32, 64, stride=2),
            _PlainResBlock(64, 128),
        )
        self.proj = nn.Linear(128, d)
        self.pv_tok = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, img_frames + pv_len, d))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=4 * d,
                                         dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def _frame_feat(self, x_img):
        from torch.utils.checkpoint import checkpoint
        B, n = x_img.shape[0], self.nf
        flat = x_img.reshape(B * n, 3, x_img.shape[-2], x_img.shape[-1])
        outs = []
        for c in flat.split(384):                     # 每块 384 帧
            outs.append(checkpoint(self.cnn, c, use_reentrant=False)
                        if self.training else self.cnn(c))
        return torch.cat(outs)

    def forward(self, x_img, x_pv):
        B, n = x_img.shape[0], self.nf
        f = self._frame_feat(x_img)
        z = self.proj(f.mean(dim=(2, 3)).reshape(B, n, 128))
        p = self.pv_tok(x_pv.unsqueeze(-1))
        h = self.encoder(torch.cat([z, p], dim=1) + self.pos[:, : n + x_pv.shape[1]])
        return self.head(torch.cat([h.mean(dim=1), h[:, -1]], dim=1)).squeeze(-1)


class PatchTST(nn.Module):
    """PatchTST 复现（单模态 PV）：实例归一化 → patch 切块 + 线性嵌入 +
    位置编码 → Transformer encoder → 回归；预测后反归一化。"""

    def __init__(self, pv_len=15, patch_len=4, stride=2, d=64):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patch = (pv_len - patch_len) // stride + 1
        self.n_patch = n_patch
        self.embed = nn.Linear(patch_len, d)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, d))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=2, dim_feedforward=4 * d,
                                         dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(n_patch * d, 128), nn.ReLU(),
                                  nn.Linear(128, 1))

    def forward(self, x_img, x_pv):
        mu = x_pv.mean(dim=1, keepdim=True)
        sd = x_pv.std(dim=1, keepdim=True).clamp_min(1e-4)
        x = (x_pv - mu) / sd
        x = x.unfold(1, self.patch_len, self.stride)      # (B,n_patch,patch_len)
        h = self.encoder(self.embed(x) + self.pos)
        y = self.head(h.flatten(1)).squeeze(-1)
        return y * sd.squeeze(1) + mu.squeeze(1)


class DLinear(nn.Module):
    """DLinear 复现（单模态 PV）：滑动平均分解 trend/seasonal，
    两个线性层分别外推后相加。"""

    def __init__(self, pv_len=15, k=5):
        super().__init__()
        self.k = k
        self.lin_t = nn.Linear(pv_len, 1)
        self.lin_s = nn.Linear(pv_len, 1)

    def forward(self, x_img, x_pv):
        trend = torch.nn.functional.avg_pool1d(
            x_pv.unsqueeze(1), kernel_size=self.k, stride=1,
            padding=self.k // 2, count_include_pad=False).squeeze(1)
        if trend.shape[1] != x_pv.shape[1]:
            trend = trend[:, : x_pv.shape[1]]
        seasonal = x_pv - trend
        return (self.lin_t(trend) + self.lin_s(seasonal)).squeeze(-1)


class CrossGNN(nn.Module):
    """CrossGNN 复现（单模态 PV，Qu et al. 2024 思路适配）：多尺度
    （原序列 / 平滑 / 再平滑）节点图 + 可学习邻接（节点嵌入余弦相似
    ReLU 截断）+ 两层 GCN；每尺度末节点特征 concat 出预测。"""

    def __init__(self, pv_len=15, d=32):
        super().__init__()
        self.scales = [pv_len, 8, 4]
        self.val_emb = nn.ModuleList([nn.Linear(1, d) for _ in self.scales])
        self.node_emb = nn.Parameter(torch.randn(sum(self.scales), d) * 0.02)
        self.gcn1 = nn.Linear(d, d)
        self.gcn2 = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(len(self.scales) * d, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, x_img, x_pv):
        feats, idx = [], 0
        A = torch.relu(torch.nn.functional.cosine_similarity(
            self.node_emb.unsqueeze(1), self.node_emb.unsqueeze(0), dim=2))
        A = A / A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        for s_len, emb in zip(self.scales, self.val_emb):
            x = torch.nn.functional.adaptive_avg_pool1d(
                x_pv.unsqueeze(1), s_len).squeeze(1)      # (B,s_len)
            feats.append(emb(x.unsqueeze(-1)))            # (B,s_len,d)
        H = torch.cat(feats, dim=1)                        # (B,27,d)
        H = torch.relu(torch.einsum("ij,bjd->bid", A, self.gcn1(H)))
        H = torch.relu(torch.einsum("ij,bjd->bid", A, self.gcn2(H)))
        outs = [H[:, s - 1] for s in self.scales]
        return self.head(torch.cat(outs, dim=1)).squeeze(-1)
