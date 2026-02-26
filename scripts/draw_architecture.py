"""
Generate AME-2 architecture diagram.
Saves: docs/architecture.png
Run:   python scripts/draw_architecture.py
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import to_rgba

os.makedirs("docs", exist_ok=True)

# ─── colour palette ───────────────────────────────────────────────────────────
C_MAP   = "#4A90D9"   # blue  – mapping / perception
C_PROP  = "#E8762A"   # orange – proprioception
C_ENC   = "#6BAA5A"   # green  – encoder / attention
C_DEC   = "#9B59B6"   # purple – decoder / output
C_CRIT  = "#C0392B"   # red    – critic (training only)
C_BG    = "#F8F8F8"
C_LABEL = "#2C3E50"
C_ARROW = "#555555"
C_DASH  = "#AAAAAA"

ALPHA_BOX  = 0.88
ALPHA_LITE = 0.35

# ─── helpers ──────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, color, label, sub="", alpha=ALPHA_BOX,
        fontsize=8.5, subsize=7, bold=False, corner=0.05):
    """Draw a rounded box with optional sub-label."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={corner}",
        linewidth=1.2,
        edgecolor=color,
        facecolor=to_rgba(color, alpha),
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (h * 0.13 if sub else 0), label,
            ha="center", va="center",
            fontsize=fontsize, color=C_LABEL, fontweight=weight, zorder=5)
    if sub:
        ax.text(x, y - h * 0.22, sub,
                ha="center", va="center",
                fontsize=subsize, color=C_LABEL, alpha=0.75, zorder=5)


def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW,
          lw=1.4, head=7, style="->", curve=0.0):
    """Draw an arrow from (x0,y0) to (x1,y1)."""
    rad = f"arc3,rad={curve}" if curve else "arc3,rad=0"
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            mutation_scale=head,
            connectionstyle=rad,
        ),
        zorder=4,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.01, my, label,
                ha="left", va="center",
                fontsize=6.5, color=color, zorder=6)


def hline(ax, x0, x1, y, color=C_DASH, lw=1.0, ls="--"):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=3)


def vline(ax, x, y0, y1, color=C_DASH, lw=1.0, ls="--"):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=ls, zorder=3)


def section_label(ax, x, y, text, color="#888888"):
    ax.text(x, y, text, fontsize=7.5, color=color,
            fontstyle="italic", ha="center", va="center", zorder=6)


# ─── figure ───────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 11
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_facecolor(C_BG)
fig.patch.set_facecolor(C_BG)
ax.axis("off")

# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.38,
        "AME-2: Attention-Based Neural Map Encoding — Policy Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C_LABEL)
ax.text(FIG_W / 2, FIG_H - 0.75,
        "arXiv:2601.08485  ·  Zhang, Klemm, Yang, Hutter (ETH Zurich RSL)",
        ha="center", va="center", fontsize=8, color="#777777")

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN LAYOUT  (x centres)
#   Col A:  raw inputs           x ≈ 1.5
#   Col B:  mapping pipeline     x ≈ 3.6
#   Col C:  neural map / GT map  x ≈ 5.6
#   Col D:  AME2Encoder          x ≈ 8.0
#   Col E:  prop encoders        x ≈ 11.5
#   Col F:  fusion / decoder     x ≈ 14.0
#   Col G:  output               x ≈ 16.5
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# ROW A  ─  Student (deployed) path   y ≈ 7.0 … 8.8
# ─────────────────────────────────────────────────────────────────────────────

Y_STU = 8.0   # centre row for student path

# A1: Depth cloud input
box(ax, 1.3, Y_STU, 1.6, 0.65, C_MAP,
    "Depth Cloud", "31×51 @ 4 cm", bold=True)

# A2: MappingNet
box(ax, 3.5, Y_STU + 0.5, 2.0, 0.70, C_MAP,
    "MappingNet", "U-Net (9 475 params)\n16-ch + pool + skip", subsize=6.5)

box(ax, 3.5, Y_STU - 0.4, 2.0, 0.55, C_MAP,
    "β-NLL Loss", "Eq. 9  +  TV weight Eq. 10",
    subsize=6.5, alpha=ALPHA_LITE)

arrow(ax, 2.1, Y_STU, 2.5, Y_STU + 0.5)
ax.text(2.35, Y_STU + 0.15, "raw\nelev", fontsize=6, color=C_MAP, ha="center")

arrow(ax, 3.5, Y_STU + 0.15, 3.5, Y_STU - 0.12, style="-", lw=1.0,
      color=C_DASH)   # connect MappingNet → loss (dashed conceptual)

# MappingNet output labels
ax.text(4.7, Y_STU + 0.63, "ê  (elev)", fontsize=6.5, color=C_MAP)
ax.text(4.7, Y_STU + 0.40, "log σ²",     fontsize=6.5, color=C_MAP)

arrow(ax, 4.5, Y_STU + 0.5, 5.2, Y_STU + 0.5)

# A3: WTA Fusion
box(ax, 6.35, Y_STU + 0.5, 1.9, 0.70, C_MAP,
    "WTA Fusion", "Prob. WTA  Eq.6–8\nGlobal 400×400 @ 8 cm", subsize=6.5)

arrow(ax, 7.3, Y_STU + 0.5, 7.85, Y_STU + 0.5)

# A4: Neural map crop
box(ax, 8.55, Y_STU + 0.5, 1.5, 0.70, C_MAP,
    "Neural Map", "Crop  14×36 @ 8 cm\n(elev, nx, ny, var)  4ch", subsize=6.5)

# Partial map randomization annotation
ax.text(7.6, Y_STU + 1.10, "± map drift, cell dropout,\npartial map (student training)",
        fontsize=6, color="#888888", ha="center", style="italic")
arrow(ax, 7.6, Y_STU + 0.88, 7.6, Y_STU + 0.87, style="->", color="#AAAAAA", lw=0.8)

# ─────────────────────────────────────────────────────────────────────────────
# ROW B  ─  Proprioception path   y ≈ 5.5
# ─────────────────────────────────────────────────────────────────────────────

Y_PRO = 5.6

# B1: history input
box(ax, 1.3, Y_PRO, 1.6, 0.65, C_PROP,
    "Prop History", "T=20 steps\n42D / step", bold=True)

# B2: LSIO encoder
box(ax, 3.5, Y_PRO, 2.1, 0.85, C_PROP,
    "LSIO", "Short: last 4×42D → 168D\nLong:  Conv1d×2    →  16D\nOut:   concat      → 184D",
    subsize=6.2)

arrow(ax, 2.1, Y_PRO, 2.44, Y_PRO)

# B3: Command input
box(ax, 3.5, Y_PRO - 1.05, 1.5, 0.55, C_PROP,
    "cmd_actor", "[d_xy, sin θ, cos θ]\n3D (continuous deploy)", subsize=6.5)

# concat symbol
cx, cy = 5.45, Y_PRO - 0.30
circle = plt.Circle((cx, cy), 0.22, color=C_PROP, fill=True, alpha=0.2, zorder=4)
ax.add_patch(circle)
ax.text(cx, cy, "cat", fontsize=7, ha="center", va="center", color=C_PROP, fontweight="bold")

arrow(ax, 4.55, Y_PRO, cx - 0.22, cy + 0.10)
arrow(ax, 4.25, Y_PRO - 1.05, cx - 0.12, cy - 0.15)

# B4: Student prop MLP
box(ax, 6.7, Y_PRO - 0.30, 1.6, 0.60, C_PROP,
    "Prop MLP", "Linear(187→256)\n→  prop_emb  128D", subsize=6.5)

arrow(ax, cx + 0.22, cy, 5.9, cy)

# ─────────────────────────────────────────────────────────────────────────────
# TEACHER path highlight  (dashed box, lighter)
# ─────────────────────────────────────────────────────────────────────────────

Y_TEA = 3.5

box(ax, 1.3, Y_TEA, 1.6, 0.60, C_PROP,
    "Prop (48D)", "base_vel(3)+hist(42)\n+cmd_actor(3)", alpha=ALPHA_LITE)
box(ax, 3.5, Y_TEA, 1.8, 0.60, C_PROP,
    "TeacherPropEncoder", "MLP(48→256→128)\nprop_emb 128D", subsize=6.5, alpha=ALPHA_LITE)
arrow(ax, 2.1, Y_TEA, 2.6, Y_TEA, color=C_DASH)

box(ax, 1.3, Y_TEA - 1.0, 1.6, 0.60, C_MAP,
    "GT Map (3ch)", "RayCaster 14×36\n(x_rel,y_rel,z_rel)", alpha=ALPHA_LITE)

# Teacher label
ax.text(0.35, Y_TEA - 0.0, "Teacher\n(train only)",
        fontsize=7, color="#AAAAAA", ha="center", va="center", style="italic")
ax.text(0.35, Y_TEA - 1.0, "Teacher\nmap (GT)",
        fontsize=7, color="#AAAAAA", ha="center", va="center", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# AME-2 ENCODER  (centre stage)   x ≈ 9.5 … 13.5
# ─────────────────────────────────────────────────────────────────────────────

EX = 10.5   # encoder block centre x
EY = 7.2    # encoder centre y

# Background region
enc_bg = FancyBboxPatch(
    (9.05, 4.60), 4.5, 4.05,
    boxstyle="round,pad=0.12",
    linewidth=1.8, edgecolor=C_ENC,
    facecolor=to_rgba(C_ENC, 0.06),
    linestyle="--",
)
ax.add_patch(enc_bg)
ax.text(11.3, 8.55, "AME-2 Encoder", fontsize=9.5,
        color=C_ENC, fontweight="bold", ha="center")

# E1: CNN on map
box(ax, EX, 8.0, 1.8, 0.58, C_ENC,
    "Local CNN", "Conv2d × 2  →  64ch\nper-cell local feats", subsize=6.5)

# E2: Coord Pos Emb
box(ax, EX, 7.1, 1.8, 0.55, C_ENC,
    "CoordPosEmb", "MLP(2→64)\nper-cell pos encoding", subsize=6.5)

# E3: Fusion MLP  →  K,V
box(ax, EX, 6.25, 1.8, 0.55, C_ENC,
    "Fusion MLP", "Linear(128→64)\nPointwise K, V  64D", subsize=6.5)

arrow(ax, EX, 7.71, EX, 7.37)
arrow(ax, EX, 6.83, EX, 6.52)

# Arrows from map to CNN and PosEmb
arrow(ax, 9.30, 8.0, 9.10, 8.0)   # from neural map to CNN
arrow(ax, 9.30, 7.10, 9.10, 7.10)  # from neural map to PosEmb
ax.annotate("", xy=(EX - 0.9, 7.1), xytext=(EX - 0.9, 8.0),
            arrowprops=dict(arrowstyle="-", color=C_MAP, lw=1.0))

# E4: Global MLP + MaxPool
box(ax, 12.2, 7.0, 1.9, 0.58, C_ENC,
    "Global MLP", "Linear(64→128)\n+ MaxPool   global 128D", subsize=6.5)

arrow(ax, EX + 0.9, 6.25, 11.0, 6.80, curve=-0.2)  # from Fusion → Global
ax.text(11.0, 6.45, "pointwise\nfeats", fontsize=6, color=C_ENC, ha="center")

# E5: Query projection
box(ax, 12.2, 5.8, 1.9, 0.58, C_ENC,
    "Query Proj", "MLP(128+128→64)\nQ  64D", subsize=6.5)

arrow(ax, 12.2, 6.71, 12.2, 6.09)

# prop_emb → query
arrow(ax, 8.50, Y_PRO - 0.30, 12.2, 5.80, color=C_PROP, curve=-0.30)
ax.text(11.0, 4.90, "prop_emb 128D", fontsize=6.5, color=C_PROP, ha="center")

# E6: Multi-Head Attention
box(ax, EX, 5.15, 1.8, 0.62, C_ENC,
    "MHA  (h=16)", "Q → query_proj\nK,V → pointwise feats", subsize=6.5, bold=True)

arrow(ax, EX + 0.9, 5.95, EX + 0.85, 5.46, curve=0.1)  # K,V from fusion
arrow(ax, 12.2, 5.51, EX + 0.9, 5.25, color=C_ENC)       # Q from query_proj

# E7: Concat  weighted_local + global
box(ax, 12.2, 4.85, 1.9, 0.55, C_ENC,
    "cat[ wlocal ‖ global ]",
    "64D + 128D = map_emb 192D", subsize=6.5)

arrow(ax, EX + 0.9, 5.15, 11.25, 4.85, curve=0.2)   # weighted local
arrow(ax, 12.2, 6.71, 12.2, 5.12, color=C_DASH, lw=1.0)  # global feat down

# ─────────────────────────────────────────────────────────────────────────────
# DECODER
# ─────────────────────────────────────────────────────────────────────────────

box(ax, 14.8, 5.6, 2.0, 0.62, C_DEC,
    "Decoder MLP",
    "cat[map_emb 192, prop_emb 128]\nLinear(320→512→256→12)", subsize=6.5, bold=True)

# map_emb → cat
arrow(ax, 13.15, 4.85, 14.8, 5.28, color=C_ENC)
# prop_emb → cat
arrow(ax, 8.50, Y_PRO - 0.30, 14.8, 5.28, color=C_PROP, curve=-0.15)

# Output
box(ax, 14.8, 4.3, 1.8, 0.58, C_DEC,
    "Actions", "Joint PD targets\n12D @ 50 Hz", bold=True)
arrow(ax, 14.8, 5.29, 14.8, 4.59)

# ─────────────────────────────────────────────────────────────────────────────
# ASYMMETRIC CRITIC  (training only, bottom strip)
# ─────────────────────────────────────────────────────────────────────────────

Y_CR = 2.2

# critic background
crit_bg = FancyBboxPatch(
    (1.0, 1.45), 15.5, 1.5,
    boxstyle="round,pad=0.12",
    linewidth=1.4, edgecolor=C_CRIT,
    facecolor=to_rgba(C_CRIT, 0.04),
    linestyle="--",
)
ax.add_patch(crit_bg)
ax.text(8.8, 1.57, "Asymmetric Critic  (training only — shared for Teacher & Student)",
        fontsize=7.5, color=C_CRIT, ha="center", fontstyle="italic")

box(ax, 2.5, Y_CR, 1.8, 0.60, C_CRIT,
    "Critic Prop (50D)",
    "base_vel+hist+cmd_critic(5D)", subsize=6.5)
box(ax, 5.0, Y_CR, 2.0, 0.60, C_CRIT,
    "TeacherPropEncoder",
    "MLP(50→256→128)  128D", subsize=6.5)
arrow(ax, 3.4, Y_CR, 4.0, Y_CR)

box(ax, 7.5, Y_CR, 1.8, 0.60, C_CRIT,
    "GT Map (3ch)",
    "RayCaster 14×36 8cm", subsize=6.5)

box(ax, 10.0, Y_CR, 2.0, 0.60, C_CRIT,
    "AME2Encoder (critic)",
    "same arch, separate weights\nmap_emb 192D", subsize=6.5)
arrow(ax, 8.4, Y_CR, 9.0, Y_CR)
arrow(ax, 6.0, Y_CR, 6.5, Y_CR, color=C_DASH)  # prop_emb to encoder (dashed)

box(ax, 12.5, Y_CR, 1.8, 0.60, C_CRIT,
    "Contact (4D)",
    "per-foot contact state", subsize=6.5)

box(ax, 14.9, Y_CR, 1.8, 0.62, C_CRIT,
    "MoE Gate + 4 Experts",
    "contact → gate weights\nΣ gate_i · expert_i → V",
    bold=True, subsize=6.5)
arrow(ax, 11.0, Y_CR, 11.8, Y_CR)  # map_emb
arrow(ax, 13.4, Y_CR, 14.0, Y_CR)  # contact

# L-R symmetry note
ax.text(14.9, Y_CR - 0.65,
        "L-R symmetry augmentation applied here\n(flip → V_flip; V = 0.5·(V_orig + V_flip))",
        fontsize=6.5, color=C_CRIT, ha="center", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PHASE LABELS  (right margin)
# ─────────────────────────────────────────────────────────────────────────────

for y_pos, phase, detail, col in [
    (8.2, "Phase 0", "MappingNet pretraining\n(no Isaac Sim)", C_MAP),
    (6.5, "Phase 1", "Teacher PPO  80k iter\nEntropy decay + noise curriculum", C_PROP),
    (4.8, "Phase 2", "Student Distillation + PPO  40k iter\n5k pure distill → 35k PPO+distill", C_DEC),
]:
    badge = FancyBboxPatch(
        (16.1, y_pos - 0.50), 1.75, 1.10,
        boxstyle="round,pad=0.08",
        linewidth=1.0, edgecolor=col,
        facecolor=to_rgba(col, 0.10),
    )
    ax.add_patch(badge)
    ax.text(16.98, y_pos + 0.12, phase,
            fontsize=8, color=col, fontweight="bold", ha="center")
    ax.text(16.98, y_pos - 0.22, detail,
            fontsize=6.2, color=col, ha="center")

# ─────────────────────────────────────────────────────────────────────────────
# Connector: neural map → AME2Encoder
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, 9.30, Y_STU + 0.50, 9.30, 8.0, color=C_MAP, curve=0.0)
ax.annotate("", xy=(9.10, 8.0), xytext=(9.30, 8.0),
            arrowprops=dict(arrowstyle="->", color=C_MAP, lw=1.4))

# Teacher GT map → Teacher AME2Encoder  (dashed)
arrow(ax, 2.1, Y_TEA - 1.0, 9.10, 7.60, color=C_DASH, lw=1.0, curve=-0.08)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_MAP,   label="Perception / Mapping"),
    mpatches.Patch(color=C_PROP,  label="Proprioception"),
    mpatches.Patch(color=C_ENC,   label="AME-2 Encoder"),
    mpatches.Patch(color=C_DEC,   label="Decoder / Output"),
    mpatches.Patch(color=C_CRIT,  label="Asymmetric Critic (train only)"),
    mpatches.Patch(color=C_DASH,  label="Teacher path (dashed)"),
]
ax.legend(handles=legend_items, loc="lower left",
          bbox_to_anchor=(0.0, 0.0),
          ncol=3, fontsize=7.5,
          framealpha=0.7, edgecolor="#CCCCCC")

# ─────────────────────────────────────────────────────────────────────────────
# Student vs Teacher separator
# ─────────────────────────────────────────────────────────────────────────────
hline(ax, 0.05, 16.0, 4.3, color="#CCCCCC", lw=1.2)
ax.text(0.25, 4.0, "▲ Deployed (Student)", fontsize=7, color="#888888")
ax.text(0.25, 3.7, "▼ Training-only", fontsize=7, color="#888888")

# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 1])
out_path = os.path.join("docs", "architecture.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
print(f"Saved → {out_path}")
