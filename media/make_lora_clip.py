"""Tạo clip giải thích cơ chế LoRA -> media/lora_explained.mp4

Chạy: .venv/bin/python media/make_lora_clip.py   (cần ffmpeg trong PATH)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch

FPS = 24
INK, MUTED, FROZEN, LORA, BG = "#1f2937", "#6b7280", "#475569", "#ea580c", "#fbfaf7"
CMAP = "RdBu_r"  # 0 = trắng, dương = đỏ, âm = xanh

rng = np.random.default_rng(0)
D, R = 16, 2
W0 = rng.standard_normal((D, D))
A_DEMO = rng.standard_normal((R, D))
B_DEMO = rng.standard_normal((D, R))
X_VEC = rng.standard_normal((D, 1))


def train_toy(steps=300):
    """Huấn luyện LoRA thật trên bài toán đồ chơi: ΔW mục tiêu có hạng 2."""
    torch.manual_seed(0)
    dW_true = torch.randn(D, R) @ torch.randn(R, D) / np.sqrt(R)
    W = torch.tensor(W0, dtype=torch.float32)
    X = torch.randn(512, D)
    Y = X @ (W + dW_true).T
    A = (0.3 * torch.tensor(A_DEMO, dtype=torch.float32)).requires_grad_()
    B = torch.zeros(D, R, requires_grad=True)          # B = 0 lúc khởi tạo
    opt = torch.optim.Adam([A, B], lr=0.05)
    hist = []
    for _ in range(steps):
        loss = ((X @ (W + B @ A).T - Y) ** 2).mean()
        hist.append((A.detach().numpy().copy(), B.detach().numpy().copy(), loss.item()))
        opt.zero_grad(); loss.backward(); opt.step()
    return dW_true.numpy(), hist


DW_TRUE, HIST = train_toy()
DW_FINAL = HIST[-1][1] @ HIST[-1][0]
DW_MAX = np.abs(DW_TRUE).max()


# ---------- helpers
def fade(t, start, dur=0.6):
    return float(np.clip((t - start) / dur, 0, 1))


def ease(s):
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def mat(ax, M, x, y, cell, vmax, a=1.0, label=None, ec=INK, lw=2):
    """Vẽ ma trận M với góc dưới-trái tại (x, y). Trả về (x, y, w, h)."""
    rows, cols = M.shape
    w, h = cols * cell, rows * cell
    if a <= 0:
        return x, y, w, h
    ax.imshow(M, extent=[x, x + w, y, y + h], cmap=CMAP, vmin=-vmax, vmax=vmax,
              alpha=a, aspect="auto", interpolation="nearest", zorder=2)
    for i in range(1, cols):
        ax.plot([x + i * cell] * 2, [y, y + h], color="#d4d4d4", lw=0.4, alpha=a, zorder=3)
    for j in range(1, rows):
        ax.plot([x, x + w], [y + j * cell] * 2, color="#d4d4d4", lw=0.4, alpha=a, zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=lw, alpha=a, zorder=4))
    if label:
        ax.text(x + w / 2, y - 0.25, label, ha="center", va="top", fontsize=15, color=ec, alpha=a)
    return x, y, w, h


def text(ax, x, y, s, size=18, a=1.0, color=INK, **kw):
    if a > 0:
        ax.text(x, y, s, fontsize=size, color=color, alpha=a, ha=kw.pop("ha", "center"),
                va=kw.pop("va", "center"), zorder=6, **kw)


def title(ax, s, a=1.0):
    text(ax, 8, 8.35, s, size=30, a=a, weight="bold")


def caption(ax, s, a=1.0):
    if a > 0:
        ax.text(8, 0.6, s, fontsize=17, color=INK, alpha=a, ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.5", fc="#fff7ed", ec=LORA, lw=1.5, alpha=a))


def along(pts, s):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.cumsum(seg)
    d = np.clip(s, 0, 1) * cum[-1]
    i = min(int(np.searchsorted(cum, d)), len(seg) - 1)
    f = (d - (cum[i] - seg[i])) / seg[i]
    return pts[i] + f * (pts[i + 1] - pts[i])


def box(ax, x, y, w, h, label, fc, ec, a=1.0, size=16, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=2, alpha=a, zorder=4))
    text(ax, x + w / 2, y + h / 2, label, size=size, a=a, color=color)


# ---------- scenes: fn(ax, t) với t = giây tính từ đầu cảnh
def s_title(ax, t):
    a = fade(t, 0)
    text(ax, 8, 5.4, "LoRA", size=72, a=a, weight="bold", color=LORA)
    text(ax, 8, 4.2, "Low-Rank Adaptation", size=28, a=a)
    text(ax, 8, 3.2, "Cơ chế hoạt động, từng bước một", size=20, a=fade(t, 1), color=MUTED)


def s_weight(ax, t):
    title(ax, "1. Mỗi lớp của mạng nơ-ron là một ma trận W")
    c = 0.3
    h = W0 @ X_VEC
    mat(ax, h / 3, 2.4, 2.2, c, 2.5, a=fade(t, 2.5), label="h  (output)")
    text(ax, 3.4, 4.6, "=", size=34, a=fade(t, 2.5))
    mat(ax, W0, 4.0, 2.2, c, 2.5, a=fade(t, 0.3), label="W  (d × d)")
    text(ax, 9.3, 4.6, "·", size=40, a=fade(t, 1.5))
    mat(ax, X_VEC, 9.8, 2.2, c, 2.5, a=fade(t, 1.5), label="x  (input)")
    text(ax, 13.2, 5.4, "mỗi ô = 1 tham số", size=16, a=fade(t, 3.5), color=MUTED)
    text(ax, 13.2, 4.8, "đỏ: dương · xanh: âm", size=16, a=fade(t, 3.5), color=MUTED)
    text(ax, 13.2, 4.2, "trắng: ≈ 0", size=16, a=fade(t, 3.5), color=MUTED)
    caption(ax, "Fine-tuning = điều chỉnh các con số trong W để mô hình làm tốt tác vụ mới", fade(t, 4.5))


def s_full(ax, t):
    title(ax, "2. Full fine-tuning: cập nhật TẤT CẢ các số")
    noise = np.random.default_rng(int(t * 6)).standard_normal((D, D))
    W = W0 + (0.5 * noise if t > 1 else 0)
    pulse = 0.5 + 0.5 * np.sin(t * 8)
    mat(ax, W, 1.5, 2.0, 0.3, 2.5, ec=LORA if t > 1 else INK, lw=2 + 2 * pulse * (t > 1), label="W: d × d = 256 số đều đổi")
    text(ax, 7.3, 6.4, "Mô hình thật:", size=20, a=fade(t, 2.5), ha="left", weight="bold")
    lines = ["d = 4096  →  16,8 triệu số / ma trận",
             "× hàng trăm ma trận  →  hàng tỷ tham số",
             "mỗi tham số cần ~16 byte khi train (Adam)",
             "→ model 8B cần ~128 GB GPU",
             "→ mỗi tác vụ = 1 bản sao đầy đủ (16 GB)"]
    for i, s in enumerate(lines):
        text(ax, 7.3, 5.6 - 0.65 * i, s, size=17, a=fade(t, 3 + 0.8 * i), ha="left",
             color=LORA if i >= 3 else INK)
    caption(ax, "Rất đắt về bộ nhớ và lưu trữ", fade(t, 7.5))


def _lora_eq(ax, t, B, A, dW, b_label="B  (d × r)", show_dw_at=4.0, frozen_at=0.0, ba_at=1.5, dw_vmax=None):
    c = 0.24
    mat(ax, W0, 0.6, 2.6, c, 2.5, a=fade(t, frozen_at), ec=FROZEN, label="W₀  (đóng băng)")
    text(ax, 4.9, 4.52, "+", size=34, a=fade(t, ba_at))
    bv = max(np.abs(B).max(), 1e-6)
    mat(ax, B, 5.3, 2.6, c, bv, a=fade(t, ba_at), ec=LORA, label=b_label)
    text(ax, 6.12, 4.52, "×", size=28, a=fade(t, ba_at))
    mat(ax, A, 6.45, 4.28, c, np.abs(A).max(), a=fade(t, ba_at), ec=LORA, label="A  (r × d)")
    text(ax, 10.7, 4.52, "=", size=34, a=fade(t, show_dw_at))
    mat(ax, dW, 11.1, 2.6, c, dw_vmax or np.abs(B_DEMO @ A_DEMO).max(), a=fade(t, show_dw_at),
        ec=LORA, label="ΔW = B·A  (d × d, hạng ≤ r)")


def s_idea(ax, t):
    title(ax, "3. Ý tưởng LoRA: chỉ học phần thay đổi ΔW")
    _lora_eq(ax, t, B_DEMO, A_DEMO, B_DEMO @ A_DEMO)
    text(ax, 2.52, 7.0, "giữ nguyên, không train", size=15, a=fade(t, 0.5), color=FROZEN)
    text(ax, 8.4, 7.0, "chỉ train 2 ma trận mỏng này (r = 2)", size=15, a=fade(t, 2), color=LORA)
    text(ax, 8, 1.55, "Full: d×d = 256 số     LoRA: r×(d+d) = 64 số", size=18, a=fade(t, 5.5), weight="bold")
    caption(ax, "Với d = 4096, r = 8:  16.777.216  →  65.536 tham số (ít hơn 256 lần)", fade(t, 7))


def s_forward(ax, t):
    title(ax, "4. Forward: x đi qua hai nhánh song song")
    yt, yb, ym = 6.3, 2.9, 4.6
    box(ax, 0.6, ym - 0.45, 1.2, 0.9, "x", "#e0f2fe", "#0369a1", size=22)
    box(ax, 14.2, ym - 0.45, 1.2, 0.9, "h", "#dcfce7", "#15803d", size=22)
    box(ax, 5.6, yt - 0.6, 3.6, 1.2, "W₀ x   (đóng băng)", "#e2e8f0", FROZEN)
    # A: d -> r (thu hẹp), B: r -> d (mở rộng)
    ax.add_patch(Polygon([[4.6, yb + 1.0], [6.8, yb + 0.22], [6.8, yb - 0.22], [4.6, yb - 1.0]],
                         fc="#ffedd5", ec=LORA, lw=2, zorder=4))
    ax.add_patch(Polygon([[8.0, yb + 0.22], [10.2, yb + 1.0], [10.2, yb - 1.0], [8.0, yb - 0.22]],
                         fc="#ffedd5", ec=LORA, lw=2, zorder=4))
    text(ax, 5.5, yb, "A", size=22, color=LORA, weight="bold")
    text(ax, 9.3, yb, "B", size=22, color=LORA, weight="bold")
    text(ax, 5.7, yb - 1.35, "nén d → r", size=14, color=LORA)
    text(ax, 9.1, yb - 1.35, "bung r → d", size=14, color=LORA)
    text(ax, 7.4, yb + 0.55, "r chiều", size=14, color=LORA, weight="bold")
    text(ax, 11.0, yb + 0.45, "× α/r", size=17, color=LORA)
    ax.add_patch(Circle((12.6, ym), 0.42, fc="white", ec=INK, lw=2, zorder=4))
    text(ax, 12.6, ym, "+", size=26)
    top = [(1.8, ym), (3.0, ym), (3.0, yt), (5.6, yt), (9.2, yt), (12.6, yt), (12.6, ym + 0.42)]
    bot = [(1.8, ym), (3.0, ym), (3.0, yb), (4.6, yb), (10.2, yb), (12.6, yb), (12.6, ym - 0.42)]
    for pts, col in [(top, FROZEN), (bot, LORA)]:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=col, lw=2.5, zorder=3)
    ax.plot([13.02, 14.2], [ym, ym], color=INK, lw=2.5, zorder=3)
    s = (t % 3.5) / 3.0
    if s <= 1:
        for pts, col in [(top, FROZEN), (bot, LORA)]:
            px, py = along(pts, s)
            ax.add_patch(Circle((px, py), 0.17, fc=col, ec="white", lw=1.5, zorder=8))
    caption(ax, "h = W₀x + (α/r)·B(Ax)   —   nhánh LoRA đi qua \"nút cổ chai\" chỉ r chiều", fade(t, 1))


def s_init(ax, t):
    title(ax, "5. Khởi tạo: B = 0")
    A, B = HIST[0][0], HIST[0][1]
    _lora_eq(ax, t + 10, B, A, B @ A, b_label="B = 0", dw_vmax=1)
    text(ax, 5.54, 7.0, "toàn số 0", size=15, color=LORA, weight="bold")
    text(ax, 8.4, 5.0, "ngẫu nhiên", size=15, color=LORA)
    text(ax, 13.0, 7.0, "ΔW = 0", size=17, color=LORA, weight="bold", a=fade(t, 1.5))
    caption(ax, "ΔW = 0  ⇒  lúc bắt đầu, mô hình y hệt mô hình gốc (không bị \"phá\")", fade(t, 2.5))
    text(ax, 8, 1.55, "A ngẫu nhiên để gradient chảy được — nếu cả A và B đều = 0 thì không bao giờ học",
         size=15, a=fade(t, 4.5), color=MUTED)


TRAIN_DUR = 14.0


def s_train(ax, t):
    title(ax, "6. Huấn luyện: gradient chỉ cập nhật A và B")
    k = min(int(ease(t / (TRAIN_DUR - 2)) * (len(HIST) - 1)), len(HIST) - 1)
    A, B, _ = HIST[k]
    c = 0.22
    mat(ax, B, 0.8, 3.0, c, np.abs(HIST[-1][1]).max(), ec=LORA, label="B")
    text(ax, 1.6, 4.76, "×", size=24)
    mat(ax, A, 1.9, 4.54, c, np.abs(HIST[-1][0]).max(), ec=LORA, label="A")
    text(ax, 5.85, 4.76, "=", size=30)
    mat(ax, B @ A, 6.2, 3.0, c, DW_MAX, ec=LORA, label="ΔW đang học")
    mat(ax, DW_TRUE, 10.2, 4.0, 0.15, DW_MAX, ec=INK, label="ΔW cần học")
    text(ax, 11.4, 7.0, "(mục tiêu, hạng 2)", size=13, color=MUTED)
    # đường loss (thang log)
    x0, y0, w, h = 13.3, 3.0, 2.3, 3.6
    logs = np.log10([hh[2] for hh in HIST])
    lo, hi = logs.min(), logs.max()
    xs = x0 + w * np.arange(k + 1) / (len(HIST) - 1)
    ys = y0 + h * (logs[:k + 1] - lo) / (hi - lo)
    ax.plot([x0, x0, x0 + w], [y0 + h, y0, y0], color=MUTED, lw=1.2, zorder=3)
    ax.plot(xs, ys, color=LORA, lw=2.5, zorder=5)
    text(ax, x0 + w / 2, y0 + h + 0.35, "loss (log)", size=14, color=MUTED)
    text(ax, 8, 2.05, f"bước {k + 1}/{len(HIST)}     loss = {HIST[k][2]:.1e}     W₀: không đổi", size=16, color=MUTED)
    caption(ax, "ΔW = B·A tiến dần tới đúng phần thay đổi cần thiết (số liệu huấn luyện thật)", fade(t, 1))


def s_merge(ax, t):
    title(ax, "7. Sau khi train: gộp ΔW vào W₀")
    c = 0.24
    s = ease((t - 1.5) / 2.5)
    wm = W0 + DW_FINAL
    mat(ax, W0, 0.8, 2.6, c, 2.5, ec=FROZEN, label="W₀")
    text(ax, 5.1, 4.52, "+", size=34)
    mat(ax, DW_FINAL, 5.6, 2.6, c, 2.5, ec=LORA, label="ΔW = B·A")
    text(ax, 9.9, 4.52, "=", size=34, a=fade(t, 1))
    if 0 < s < 1:  # hai bản "ma" trượt về vị trí kết quả
        mat(ax, W0, 0.8 + s * (10.4 - 0.8), 2.6, c, 2.5, a=0.5 * (1 - s), ec=FROZEN)
        mat(ax, DW_FINAL, 5.6 + s * (10.4 - 5.6), 2.6, c, 2.5, a=0.5 * (1 - s), ec=LORA)
    mat(ax, wm, 10.4, 2.6, c, 2.5, a=fade(t, 3.8), ec=INK, label="W' = W₀ + ΔW")
    text(ax, 8, 1.6, "W' cùng kích thước W₀  →  vẫn chỉ 1 phép nhân ma trận", size=17, a=fade(t, 5), weight="bold")
    caption(ax, "Inference nhanh y như mô hình gốc — không tốn thêm chút chi phí nào", fade(t, 6))


ADAPTERS = [("Luật", "#ea580c"), ("Y tế", "#7c3aed"), ("Chăm sóc KH", "#0891b2")]


def s_adapters(ax, t):
    title(ax, "8. Hoặc giữ riêng: 1 mô hình gốc, nhiều adapter")
    box(ax, 5.0, 3.2, 4.0, 2.6, "Mô hình gốc\n(đóng băng)\n16 GB", "#e2e8f0", FROZEN, size=18)
    slot = (9.4, 4.0)
    active = int(t // 3) % len(ADAPTERS)
    ph = (t % 3) / 3
    for i, (name, col) in enumerate(ADAPTERS):
        home = (12.3, 5.9 - 1.3 * i)
        if i == active and t < 3 * len(ADAPTERS):
            p = ease(ph / 0.3) if ph < 0.7 else 1 - ease((ph - 0.7) / 0.3)
            x, y = home[0] + p * (slot[0] - home[0]), home[1] + p * (slot[1] - home[1])
        else:
            x, y = home
        box(ax, x, y, 2.6, 1.0, f"Adapter {name}\n~10 MB", "white", col, size=14, color=col)
    text(ax, 0.6, 6.6, "Full fine-tune 3 tác vụ:", size=16, ha="left", a=fade(t, 1), weight="bold")
    text(ax, 0.6, 6.0, "3 × 16 GB = 48 GB", size=16, ha="left", a=fade(t, 1), color=LORA)
    text(ax, 0.6, 5.0, "LoRA 3 tác vụ:", size=16, ha="left", a=fade(t, 2), weight="bold")
    text(ax, 0.6, 4.4, "16 GB + 3 × 10 MB", size=16, ha="left", a=fade(t, 2), color="#15803d")
    caption(ax, "Cắm / rút adapter tùy tác vụ — mô hình gốc không bao giờ bị sửa", fade(t, 1.5))


def s_summary(ax, t):
    title(ax, "Tóm tắt")
    pts = ["W₀ đóng băng — không có gradient, không trạng thái Adam",
           "Học ΔW = (α/r)·B·A với r nhỏ (4–64): chỉ ~0,1–1% tham số",
           "B = 0 lúc đầu → bắt đầu đúng từ mô hình gốc",
           "Train xong: gộp vào W₀ (không chậm) hoặc giữ adapter vài MB",
           "QLoRA: nén W₀ xuống 4-bit → fine-tune model 7B trên 1 GPU"]
    for i, s in enumerate(pts):
        text(ax, 1.6, 6.8 - 1.0 * i, f"•  {s}", size=20, a=fade(t, 0.5 + 1.2 * i), ha="left")
    text(ax, 8, 1.0, "Xem code chi tiết: lora_finetune_scratch.ipynb", size=16, a=fade(t, 7), color=MUTED)


SCENES = [(4, s_title), (8, s_weight), (10, s_full), (10, s_idea), (9, s_forward),
          (8, s_init), (TRAIN_DUR, s_train), (9, s_merge), (10, s_adapters), (10, s_summary)]


def render(ax, sec):
    ax.clear()
    ax.set_facecolor(BG)
    start = 0
    for dur, fn in SCENES:
        if sec < start + dur:
            break
        start += dur
    t = sec - start
    fn(ax, t)
    # mờ dần ở đầu và cuối mỗi cảnh
    edge = min(t, dur - t)
    if edge < 0.35:
        ax.add_patch(Rectangle((0, 0), 16, 9, fc=BG, alpha=1 - edge / 0.35, zorder=20))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")


def make_fig():
    fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax


if __name__ == "__main__":
    assert HIST[-1][2] < 1e-3 * HIST[0][2], "toy training không hội tụ"
    fig, ax = make_fig()
    total = sum(d for d, _ in SCENES)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_explained.mp4")
    anim = FuncAnimation(fig, lambda i: render(ax, i / FPS), frames=int(total * FPS))
    anim.save(out, writer=FFMpegWriter(fps=FPS, bitrate=1800, extra_args=["-pix_fmt", "yuv420p"]))
    print("saved", out, f"({total}s)")
