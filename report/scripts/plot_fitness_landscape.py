"""适应性景观示意图：左为崎岖的 Kernel 代码空间，右为平滑的 Schema 设计空间。

用法：python3 plot_fitness_landscape.py [输出 PNG 路径]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
FONT_DIR = next(
    p
    for p in (HERE.parent / "fonts", Path("/Users/hanyong/KernelSchema/report/fonts"))
    if (p / "Alibaba-PuHuiTi-Regular.otf").exists()
)
for _name in ("Alibaba-PuHuiTi-Regular.otf", "Alibaba-PuHuiTi-Medium.otf"):
    font_manager.fontManager.addfont(str(FONT_DIR / _name))
FONT_REG = font_manager.FontProperties(fname=str(FONT_DIR / "Alibaba-PuHuiTi-Regular.otf"))
FONT_MED = font_manager.FontProperties(fname=str(FONT_DIR / "Alibaba-PuHuiTi-Medium.otf"))
plt.rcParams["font.family"] = FONT_REG.get_name()
plt.rcParams["axes.unicode_minus"] = False

BLUE, TEAL, AMBER = "#176B93", "#16877E", "#C58B28"
INK, MUTED = "#2B3440", "#6B7785"

N = 321          # 曲面采样分辨率
GRID_EVERY = 8   # 每隔多少个采样格画一条网格线（烘焙进面片颜色，可被正确遮挡）
L = 3.0
ZLIM = (-0.55, 1.15)
VIEW = dict(elev=30, azim=-58)

xs = np.linspace(-L, L, N)
X, Y = np.meshgrid(xs, xs)


def gauss(x0, y0, h, sx, sy=None):
    sy = sx if sy is None else sy
    return h * np.exp(-(((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2) / 2)


def smooth_surface():
    return (
        gauss(0.45, 0.95, 1.00, 0.85, 0.95)
        + gauss(-1.45, -0.55, 0.68, 0.72)
        + gauss(1.75, -1.35, 0.46, 0.60)
        - 0.10
    )


def rugged_surface(seed=11):
    rng = np.random.default_rng(seed)
    Z = 0.10 * np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / (2 * 1.8**2))
    for sigma, amp in ((26, 0.075), (11, 0.045), (5, 0.012)):
        noise = gaussian_filter(rng.standard_normal((N, N)), sigma, mode="reflect")
        Z += amp * noise / noise.std()
    peaks = [
        (-0.75, -0.95, 0.46, 0.17),   # 路径 A 的停滞点
        (0.85, -0.55, 0.52, 0.18),    # 路径 B 的停滞点
        (0.45, 1.65, 1.02, 0.15),     # 全局最高峰（未被到达）
        (-1.75, 1.05, 0.58, 0.16),
        (2.15, 1.75, 0.55, 0.15),
        (-2.25, -1.95, 0.38, 0.17),
        (-0.20, 2.40, 0.42, 0.14),
        (-2.40, 2.35, 0.32, 0.15),
        (2.45, 0.45, 0.42, 0.14),
        (-1.15, 0.10, 0.34, 0.15),
        (1.20, 0.75, 0.40, 0.14),
        (-0.40, 0.90, 0.30, 0.13),
    ]
    for x0, y0, h, s in peaks:
        Z += gauss(x0, y0, h, s)
    # 断崖：一条斜向的陡峭台阶，台阶外侧整体下沉
    d = (0.80 * X - 0.60 * Y) - 1.25
    Z -= 0.42 / (1 + np.exp(-d / 0.03)) * np.exp(-((Y + 1.0) ** 2) / 5)
    # 深坑：编译失败、死锁或超时
    for x0, y0, h, s in ((-0.10, -0.20, 0.36, 0.12), (1.75, 0.95, 0.30, 0.12), (-2.0, -0.4, 0.30, 0.14)):
        Z -= gauss(x0, y0, h, s)
    # 共同起点所在的洼地
    Z -= gauss(-0.30, -2.30, 0.12, 0.45)
    return Z


def make_interp(Z):
    f = RegularGridInterpolator((xs, xs), Z.T, bounds_error=False, fill_value=None)
    return lambda x, y: f(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))


def climb(Z, x, y, step=0.02, iters=400):
    """从 (x, y) 沿梯度爬升到所在的局部峰。"""
    gy, gx = np.gradient(Z, xs, xs)
    fx, fy = make_interp(gx), make_interp(gy)
    pts = []
    for _ in range(iters):
        g = np.array([fx(x, y)[0], fy(x, y)[0]])
        n = np.linalg.norm(g)
        if n < 1e-3:
            break
        x, y = x + step * g[0] / n, y + step * g[1] / n
        pts.append((x, y))
        if len(pts) > 6 and np.hypot(*(np.array(pts[-1]) - pts[-5])) < 2.5 * step:
            break
    return np.array(pts) if pts else np.empty((0, 2))


def path_through(Z, waypoints, finish_climb=True, n=160):
    w = np.asarray(waypoints, float)
    t = np.r_[0, np.cumsum(np.hypot(*np.diff(w, axis=0).T))]
    cs = CubicSpline(t, w, bc_type="natural")
    p = cs(np.linspace(0, t[-1], n))
    if finish_climb:
        tail = climb(Z, *p[-1])
        if len(tail):
            p = np.vstack([p, tail])
    z = make_interp(Z)(p[:, 0], p[:, 1])
    return p[:, 0], p[:, 1], z


class Arrow3D(FancyArrowPatch):
    def __init__(self, p0, p1, **kw):
        super().__init__((0, 0), (0, 0), **kw)
        self._p = np.array([p0, p1], float)

    def do_3d_projection(self, renderer=None):
        M = self.axes.M
        h = np.c_[self._p, np.ones(2)] @ M.T
        xy = h[:, :2] / h[:, 3:4]
        self.set_positions(tuple(xy[0]), tuple(xy[1]))
        return np.min(h[:, 2] / h[:, 3])


def shade(Z):
    cmap = LinearSegmentedColormap.from_list("land", ["#D9DEE3", "#E9ECEF", "#F4F6F8"])
    ls = LightSource(azdeg=300, altdeg=40)
    norm = (Z - ZLIM[0]) / (ZLIM[1] - ZLIM[0])
    rgb = ls.shade_rgb(cmap(norm)[..., :3], Z, vert_exag=1.2, blend_mode="soft", dx=xs[1] - xs[0], dy=xs[1] - xs[0])
    rgb = 0.55 * rgb + 0.45 * cmap(norm)[..., :3]
    face = rgb[:-1, :-1].copy()
    grid = np.zeros(face.shape[:2], bool)
    grid[::GRID_EVERY, :] = True
    grid[:, ::GRID_EVERY] = True
    face[grid] = face[grid] * 0.80 + np.array([0.42, 0.48, 0.55]) * 0.20
    return np.concatenate([face, np.ones(face.shape[:2] + (1,))], axis=-1)


def draw_surface(ax, Z):
    fc = shade(Z)
    ax.plot_surface(X, Y, Z, facecolors=fc, rcount=N, ccount=N, linewidth=0.25, antialiased=False, shade=False, zorder=1)


def draw_path(ax, x, y, z, color, end="arrow", lift=0.025):
    z = z + lift
    ax.plot(x, y, z, color="white", lw=4.2, alpha=0.9, solid_capstyle="round", zorder=4)
    ax.plot(x, y, z, color=color, lw=2.0, solid_capstyle="round", zorder=5)
    if end == "arrow":
        k = max(2, len(x) // 18)
        ax.add_artist(Arrow3D((x[-k], y[-k], z[-k]), (x[-1], y[-1], z[-1]), arrowstyle="-|>,head_length=0.55,head_width=0.3",
                              mutation_scale=14, color=color, lw=0, zorder=6))
    elif end == "stuck":
        ax.scatter([x[-1]], [y[-1]], [z[-1] + 0.01], s=46, marker="X", color=color, edgecolors="white", linewidths=0.8,
                   zorder=6, depthshade=False)


def draw_start(ax, x, y, z):
    ax.scatter([x], [y], [z + 0.03], s=40, color=INK, edgecolors="white", linewidths=1.0, zorder=7, depthshade=False)


def style_axes(ax, floor_label):
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(*ZLIM)
    ax.set_box_aspect((1, 1, 0.42), zoom=1.18)
    ax.view_init(**VIEW)
    ax.set_axis_off()
    ax.computed_zorder = False
    # 纵轴：性能
    x0, y0 = -L, -L
    ax.add_artist(Arrow3D((x0, y0, ZLIM[0] + 0.05), (x0, y0, ZLIM[1]), arrowstyle="-|>,head_length=0.5,head_width=0.22",
                          mutation_scale=12, color=MUTED, lw=0.9, zorder=8, clip_on=False))
    ax.text(x0 + 0.12, y0 + 0.12, ZLIM[1] - 0.05, "性能", color=MUTED, fontsize=10.5, ha="left", va="center", zorder=9, clip_on=False)
    # 底面标签，沿前左边缘书写
    ax.text(-0.2, -L - 0.75, ZLIM[0] + 0.05, floor_label, zdir="x", color=INK, fontsize=11.5, fontproperties=FONT_MED,
            ha="center", va="center", zorder=9)


def main(out):
    fig = plt.figure(figsize=(12.0, 4.4), dpi=300)
    fig.patch.set_facecolor("white")
    axl = fig.add_axes([0.045, 0.00, 0.475, 1.00], projection="3d")
    axr = fig.add_axes([0.525, 0.00, 0.475, 1.00], projection="3d")

    # 左：崎岖的 Kernel 代码空间
    Zl = rugged_surface()
    draw_surface(axl, Zl)
    style_axes(axl, "Kernel 代码空间")
    start = (-0.30, -2.30)
    fl = make_interp(Zl)
    paths_l = [
        (BLUE, [start, (-0.75, -1.75), (-0.80, -1.10)], "stuck", True),
        (TEAL, [start, (0.25, -1.65), (0.75, -0.80)], "stuck", True),
        (AMBER, [start, (0.60, -2.10), (1.45, -1.55), (2.05, -1.05)], "stuck", False),
    ]
    for color, wp, end, fin in paths_l:
        draw_path(axl, *path_through(Zl, wp, finish_climb=fin), color, end=end)
    draw_start(axl, *start, fl(*start)[0])
    axl.set_title("崎岖：多断崖，易困于局部峰", fontproperties=FONT_MED, fontsize=13, color=INK, y=0.93)

    # 右：平滑的 Schema 设计空间
    Zr = smooth_surface()
    draw_surface(axr, Zr)
    style_axes(axr, "Schema 设计空间")
    fr = make_interp(Zr)
    paths_r = [
        (BLUE, [start, (-0.55, -1.30), (-0.10, 0.10), (0.45, 0.70)]),
        (TEAL, [start, (0.60, -1.70), (1.10, -0.40), (0.70, 0.60)]),
        (AMBER, [start, (-0.95, -1.70), (-1.35, -1.00)]),
    ]
    for color, wp in paths_r:
        draw_path(axr, *path_through(Zr, wp), color, end="arrow")
    draw_start(axr, *start, fr(*start)[0])
    axr.set_title("平滑：顺坡攀升，可达主峰", fontproperties=FONT_MED, fontsize=13, color=INK, y=0.93)

    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "fitness_landscape.png"))
