"""
AME-2 Architecture Diagram — clean grid layout, v3.
Key design principle: every inter-box connection is either purely horizontal
OR purely vertical.  No diagonals, no long curved arrows.

Layout (top → bottom):
  Band 1  MAPPING PIPELINE
  Band 2  AME-2 ENCODER   (3 internal columns, see below)
  Band 3  PROPRIOCEPTION
  Band 4  DECODER
  Band 5  ASYMMETRIC CRITIC  (training only)

Encoder internal columns (all vertical connections are short):
  ColL x≈7.8  : Local CNN → CoordPosEmb → Fusion MLP → [K,V ↓ MHA ↓ weighted_local]
  ColR x≈11.5 : Global MLP → global_feat ↓ [Query Proj ← prop_emb] → Q → [MHA]
  ColO x≈15.8 : cat[wL‖global] = map_emb 192D

Saves: docs/architecture.png
Run:   python scripts/draw_architecture.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import to_rgba

os.makedirs("docs", exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
BG     = "#FAFAFA"
C_MAP  = "#2980B9"
C_PRO  = "#D35400"
C_ENC  = "#27AE60"
C_DEC  = "#8E44AD"
C_CRIT = "#C0392B"
C_NOTE = "#7F8C8D"
TEXT   = "#2C3E50"
ALPHA  = 0.15

FIG_W, FIG_H = 21, 14.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

# ── Drawing helpers ───────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, fc, label, sub="",
         lfs=8.5, sfs=6.8, bold=False, ls="solid", alpha=ALPHA):
    patch = FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.07",
        linewidth=1.5 if ls=="solid" else 1.0,
        linestyle=ls, edgecolor=fc,
        facecolor=to_rgba(fc, alpha), zorder=3)
    ax.add_patch(patch)
    dy = h*0.13 if sub else 0
    ax.text(cx, cy+dy, label, ha="center", va="center",
            fontsize=lfs, fontweight="bold" if bold else "normal",
            color=TEXT, zorder=5)
    if sub:
        ax.text(cx, cy-h*0.22, sub, ha="center", va="center",
                fontsize=sfs, color=TEXT, alpha=0.78, zorder=5)

def bg_box(ax, x0, y0, x1, y1, color, label="", lfs=8.5):
    patch = FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0.12",
        linewidth=1.3, linestyle="--",
        edgecolor=color,
        facecolor=to_rgba(color, 0.04), zorder=1)
    ax.add_patch(patch)
    if label:
        ax.text((x0+x1)/2, y1-0.18, label, ha="center", va="top",
                fontsize=lfs, color=color, fontweight="bold", zorder=4)

def harrow(ax, x0, x1, y, color=TEXT, lw=1.5, lbl="", above=True):
    """Horizontal arrow x0→x1 at height y."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=9), zorder=6)
    if lbl:
        dy = 0.11 if above else -0.13
        ax.text((x0+x1)/2, y+dy, lbl, ha="center",
                va="bottom" if above else "top",
                fontsize=6.5, color=color, zorder=6)

def varrow(ax, x, y0, y1, color=TEXT, lw=1.5, lbl="", right=True):
    """Vertical arrow y0→y1 at x."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=9), zorder=6)
    if lbl:
        dx = 0.10 if right else -0.10
        ax.text(x+dx, (y0+y1)/2, lbl, ha="left" if right else "right",
                va="center", fontsize=6.5, color=color, zorder=6)

def Larrow(ax, x0, y0, x1, y1, color=TEXT, lw=1.4, lbl=""):
    """L-shaped connector: horizontal x0→x1, then vertical to y1, arrow at end."""
    ax.plot([x0, x1], [y0, y0], color=color, lw=lw, zorder=5)
    ax.annotate("", xy=(x1, y1), xytext=(x1, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=9), zorder=6)
    if lbl:
        ax.text((x0+x1)/2, y0+0.10, lbl, ha="center", va="bottom",
                fontsize=6.5, color=color, zorder=6)

def dot(ax, x, y, color=TEXT, r=0.07):
    """Junction dot."""
    ax.add_patch(plt.Circle((x,y), r, color=color, zorder=7))

def note(ax, x, y, text, color=C_NOTE, fs=6.5, ha="center", style="italic"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fs, color=color, fontstyle=style, zorder=5)


# ═════════════════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, FIG_H-0.38,
        "AME-2: Attention-Based Neural Map Encoding — Policy Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT)
ax.text(FIG_W/2, FIG_H-0.80,
        "Zhang, Klemm, Yang, Hutter (ETH Zurich RSL) · arXiv:2601.08485",
        ha="center", va="center", fontsize=8.5, color=C_NOTE)


# ═════════════════════════════════════════════════════════════════════════════
# BAND 1 – MAPPING PIPELINE   y = 10.3 – 11.8
# ═════════════════════════════════════════════════════════════════════════════
Y1 = 10.95
bg_box(ax, 0.3, 10.25, 18.8, 11.85, C_MAP, "  Mapping Pipeline  (Sec. V)")

rbox(ax,  1.6, Y1, 2.0, 0.78, C_MAP, "Depth Cloud",
     "31×51 cells @ 4 cm", bold=True)
rbox(ax,  4.7, Y1, 2.5, 0.78, C_MAP, "MappingNet",
     "Lightweight U-Net  (9 475 params)\nβ-NLL loss Eq.9  +  TV weight Eq.10")
rbox(ax,  8.3, Y1, 2.7, 0.78, C_MAP, "WTA Fusion",
     "Probabilistic WTA  Eq.6–8\nGlobal map  400×400 @ 8 cm")
rbox(ax, 12.1, Y1, 2.7, 0.78, C_MAP, "Neural Map",
     "Crop  14×36 @ 8 cm\n(elev, nx, ny, var)  4 ch", bold=True)
rbox(ax, 16.0, Y1, 2.7, 0.78, C_MAP, "Policy Map Input",
     "Student: neural map  4 ch\nTeacher: GT RayCaster  3 ch")

harrow(ax,  2.60,  3.45, Y1, color=C_MAP, lbl="raw elev")
harrow(ax,  5.95,  6.95, Y1, color=C_MAP, lbl="elev + log σ²")
harrow(ax,  9.65, 10.75, Y1, color=C_MAP, lbl="crop + normals")
harrow(ax, 13.45, 14.65, Y1, color=C_MAP, lbl="4ch 14×36")

note(ax, 20.0, Y1+0.22, "Phase 0", color=C_MAP, fs=8, style="normal")
note(ax, 20.0, Y1-0.18, "Pretrain\n(no Isaac Sim)", color=C_MAP, fs=7)


# ═════════════════════════════════════════════════════════════════════════════
# BAND 2 – AME-2 ENCODER   y = 6.8 – 10.1
#
#  Three encoder rows (all flows within each row are left-to-right):
#
#    RowA (y=9.55): [Local CNN] → [CoordPosEmb] → [Fusion MLP/K,V] → [Global MLP]
#                                                          |                  |
#                                                     K,V (↓ short)   global_feat (↓ short)
#                                                          |                  |
#    RowB (y=8.35): ──────────── [MHA (h=16)] ←Q─── [Query Proj] ←prop (↑ from Band3)
#                                     |
#                             weighted_local (→ right)
#                                     |
#    RowC (y=7.25): ──────────────── [cat(wL‖global) = map_emb 192D]  ───→ out
#                                              ↑
#                               global_feat (via fork from RowA→RowB vertical line)
# ═════════════════════════════════════════════════════════════════════════════
bg_box(ax, 0.3, 6.75, 18.8, 10.10, C_ENC, "  AME-2 Encoder  (Sec. IV-A, Fig. 3)")

EY_A = 9.52   # encoder Row A  (local + global path)
EY_B = 8.38   # encoder Row B  (MHA + Query Proj)
EY_C = 7.25   # encoder Row C  (map_emb output)

BH_E = 0.72   # encoder box height

# ── Column x-positions ────────────────────────────────────────────────────────
XL1 = 2.2    # Local CNN
XL2 = 5.0    # CoordPosEmb
XL3 = 8.0    # Fusion MLP / K,V  (ColL — K,V go straight down)
XR1 = 12.5   # Global MLP        (ColR — global_feat goes straight down)
XQ  = 12.5   # Query Proj        (same column as Global MLP → pure vertical!)
XM  = 8.0    # MHA               (same column as Fusion MLP → K,V pure vertical!)
XO  = 16.5   # map_emb (cat output)

# ── Row A ─────────────────────────────────────────────────────────────────────
rbox(ax, XL1, EY_A, 2.0, BH_E, C_ENC, "Local CNN",
     "Conv2d×2 → 64D\nper-cell feats")
rbox(ax, XL2, EY_A, 2.2, BH_E, C_ENC, "CoordPosEmb",
     "MLP(2→64)\nper-cell pos encoding")
rbox(ax, XL3, EY_A, 2.4, BH_E, C_ENC, "Fusion MLP",
     "Linear(128→64)\nPointwise K,V  64D", bold=True)
rbox(ax, XR1, EY_A, 2.6, BH_E, C_ENC, "Global MLP",
     "Linear(64→128) + MaxPool\nglobal_feat  128D", bold=True)

# Row A arrows (all horizontal, left→right)
harrow(ax, XL1+1.0, XL2-1.1, EY_A, color=C_ENC)
harrow(ax, XL2+1.1, XL3-1.2, EY_A, color=C_ENC, lbl="local + pe")
harrow(ax, XL3+1.2, XR1-1.3, EY_A, color=C_ENC, lbl="pointwise feats 64D")

# ── Vertical connections A→B (SHORT — same column) ────────────────────────────
# K,V: Fusion MLP → MHA  (pure vertical at x=XL3=XM)
varrow(ax, XM, EY_A-BH_E/2, EY_B+BH_E/2, color=C_ENC, lbl="K,V 64D", right=True)

# global_feat: Global MLP → fork → Query Proj AND cat
# Draw a vertical bus from Global MLP down through Row B to Row C level
BUS_X = XR1   # x=12.5

# bus top → Query Proj top (pure vertical ↓)
varrow(ax, BUS_X, EY_A-BH_E/2, EY_B+BH_E/2, color=C_ENC,
       lbl="global\nfeat 128D", right=True)

# fork dot at EY_B+BH_E/2 (top of Query Proj)
dot(ax, BUS_X, EY_B+BH_E/2, color=C_ENC, r=0.08)

# bus continues down past Query Proj to Row C (for cat)
ax.plot([BUS_X, BUS_X], [EY_B-BH_E/2, EY_C+BH_E/2],
        color=C_ENC, lw=1.4, zorder=5)
dot(ax, BUS_X, EY_C+BH_E/2, color=C_ENC, r=0.08)
# horizontal segment from bus to cat at XO
ax.plot([BUS_X, XO-1.3], [EY_C+BH_E/2, EY_C+BH_E/2],
        color=C_ENC, lw=1.4, zorder=5)
ax.annotate("", xy=(XO-1.3, EY_C+BH_E/2),
            xytext=(XO-1.3-0.01, EY_C+BH_E/2),
            arrowprops=dict(arrowstyle="->", color=C_ENC,
                            lw=1.4, mutation_scale=8), zorder=6)

# ── Row B ─────────────────────────────────────────────────────────────────────
rbox(ax, XM,  EY_B, 2.4, BH_E, C_ENC, "MHA  (h=16)",
     "Q  64D  ←  Query Proj\nK,V  64D/cell  ↓  Fusion MLP", bold=True)
rbox(ax, XQ,  EY_B, 2.6, BH_E, C_ENC, "Query Proj",
     "cat(global 128D, prop 128D)\nMLP → Q  64D", bold=True)

# Q: Query Proj → MHA  (short horizontal, going LEFT)
harrow(ax, XQ-1.3, XM+1.2, EY_B, color=C_ENC, lbl="Q  64D", above=False)

# ── Row C ─────────────────────────────────────────────────────────────────────
rbox(ax, XO, EY_C, 2.6, BH_E, C_ENC, "map_emb  192D",
     "cat[ weighted_local 64\n      ‖  global_feat  128 ]", bold=True)

# weighted_local: MHA → cat  (horizontal right at Row B level, then down)
ax.plot([XM+1.2, XO-1.3], [EY_B, EY_B], color=C_ENC, lw=1.4, zorder=5)
ax.plot([XO-1.3, XO-1.3], [EY_B, EY_C+BH_E/2],
        color=C_ENC, lw=1.4, zorder=5)
ax.annotate("", xy=(XO-1.3, EY_C+BH_E/2),
            xytext=(XO-1.3, EY_C+BH_E/2+0.01),
            arrowprops=dict(arrowstyle="->", color=C_ENC,
                            lw=1.4, mutation_scale=8), zorder=6)
ax.text((XM+1.2+XO-1.3)/2, EY_B+0.10, "weighted_local 64D",
        ha="center", va="bottom", fontsize=6.5, color=C_ENC, zorder=6)

# ── Map input: Policy Map Input → encoder top ─────────────────────────────────
# Vertical drop from Band 1 to encoder Row A
# Enter at x=XL1 (Local CNN) and x=XL2 (CoordPosEmb) — use a T-bus
MAP_BUS_Y = EY_A + BH_E/2   # just above Row A boxes
varrow(ax, 16.0, 10.25, MAP_BUS_Y, color=C_MAP)   # Policy Map Input drops down

# horizontal bus at MAP_BUS_Y from x=XL1 to x=XL2
ax.plot([XL1, XL2], [MAP_BUS_Y, MAP_BUS_Y], color=C_MAP, lw=1.4, zorder=5)
dot(ax, XL2, MAP_BUS_Y, color=C_MAP, r=0.07)
# arrows down into Local CNN and CoordPosEmb
ax.annotate("", xy=(XL1, EY_A+BH_E/2),
            xytext=(XL1, MAP_BUS_Y),
            arrowprops=dict(arrowstyle="->", color=C_MAP,
                            lw=1.4, mutation_scale=8), zorder=6)
ax.annotate("", xy=(XL2, EY_A+BH_E/2),
            xytext=(XL2, MAP_BUS_Y),
            arrowprops=dict(arrowstyle="->", color=C_MAP,
                            lw=1.4, mutation_scale=8), zorder=6)
ax.text(XL1-0.15, (10.25+MAP_BUS_Y)/2, "map\n4ch",
        ha="right", va="center", fontsize=6.5, color=C_MAP)


# ═════════════════════════════════════════════════════════════════════════════
# BAND 3 – PROPRIOCEPTION ENCODER   y = 4.55 – 6.60
#
# prop_emb is placed at x = XQ (= 12.5) so the vertical connection
# prop_emb → Query Proj is a pure short vertical arrow (no horizontal detour).
# ═════════════════════════════════════════════════════════════════════════════
Y3   = 5.55
Y3B  = 6.60   # band top
Y3T  = 4.55   # band bottom
bg_box(ax, 0.3, Y3T, 18.8, Y3B, C_PRO, "  Proprioception Encoder  (Sec. IV-A)")

BH3 = 0.80

rbox(ax,  1.8, Y3, 2.0, BH3, C_PRO, "Prop History",
     "T=20 steps\n42D/step  (no v_base, no cmd)", bold=True)
rbox(ax,  4.8, Y3, 2.5, BH3, C_PRO, "LSIO",
     "Short: last 4×42 → 168D\nLong:  Conv1d×2  → 16D\nOut:   184D")
rbox(ax,  7.8, Y3, 1.6, BH3, C_PRO, "cmd_actor",
     "[clip(d_xy,2m),\nsin θ,  cos θ]  3D")
rbox(ax, 10.6, Y3, 2.4, BH3, C_PRO, "Prop MLP",
     "cat(LSIO 184, cmd 3)\nLinear(187→256→128)", bold=True)
rbox(ax, XQ,   Y3, 2.0, BH3, C_PRO, "prop_emb",   # XQ = 12.5 = same col as Query Proj!
     "128D")

harrow(ax,  2.80,  3.70, Y3, color=C_PRO)
harrow(ax,  6.05,  6.70, Y3-0.06, color=C_PRO, lbl="LSIO 184D + cat(cmd)")
harrow(ax,  8.60,  9.40, Y3, color=C_PRO)
harrow(ax, 11.80, 12.00, Y3, color=C_PRO, lbl="prop_emb 128D")

# prop_emb → Query Proj: pure vertical ↑  (same x=XQ=12.5)
varrow(ax, XQ, Y3B, EY_B-BH_E/2, color=C_PRO,
       lbl="prop_emb\n128D", right=False)

# Teacher label
note(ax, 1.8, Y3T+0.20,
     "Teacher: MLP(48D→256→128)  (base_lin_vel included, plain MLP)",
     color=C_NOTE, fs=6.5)


# ═════════════════════════════════════════════════════════════════════════════
# BAND 4 – DECODER   y = 2.80 – 4.40
# ═════════════════════════════════════════════════════════════════════════════
Y4   = 3.60
Y4B  = 4.40   # band top
Y4T  = 2.80   # band bottom
bg_box(ax, 0.3, Y4T, 18.8, Y4B, C_DEC, "  Decoder  (Sec. IV-A)")

BH4 = 0.76

rbox(ax,  5.5, Y4, 2.4, BH4, C_DEC, "cat[ map ‖ prop ]",
     "192D + 128D = 320D")
rbox(ax, 10.0, Y4, 3.0, BH4, C_DEC, "Decoder MLP",
     "Linear(320→512→256→12)\n@ 50 Hz", bold=True)
rbox(ax, 15.0, Y4, 2.4, BH4, C_DEC, "Actions",
     "Joint PD targets  12D\n(tracked @ 400 Hz)", bold=True)

harrow(ax,  6.70,  8.50, Y4, color=C_DEC, lbl="320D")
harrow(ax, 11.50, 13.80, Y4, color=C_DEC, lbl="12D")

# map_emb → cat: vertical from encoder Row C to decoder
# map_emb is at (XO=16.5, EY_C). cat is at (5.5, Y4).
# Use L-shape: horizontal from 16.5 → 5.5 at y=EY_C-BH_E/2, then down to Y4
Larrow(ax, XO, EY_C-BH_E/2, 5.5, Y4+BH4/2,
       color=C_ENC, lbl="map_emb 192D")

# prop_emb → cat: vertical from prop band to decoder
# prop_emb bus is at x=XQ=12.5. Cat is at x=5.5.
# The prop_emb box bottom is at Y3-BH3/2=5.15. Decoder cat is at Y4+BH4/2.
# Use L-shape from (12.5, Y3T=4.55) down to (5.5, Y4+BH4/2)
ax.plot([XQ, 5.5], [Y3T, Y3T], color=C_PRO, lw=1.4, ls="--", zorder=5)
ax.annotate("", xy=(5.5, Y4+BH4/2), xytext=(5.5, Y3T),
            arrowprops=dict(arrowstyle="->", color=C_PRO,
                            lw=1.4, mutation_scale=8), zorder=6)
ax.text(9.0, Y3T+0.10, "prop_emb 128D",
        ha="center", va="bottom", fontsize=6.5, color=C_PRO, zorder=6)

# Phase labels
note(ax, 20.0, Y4+0.30, "Phase 1", color=C_PRO, fs=8, style="normal")
note(ax, 20.0, Y4+0.02, "Teacher PPO  80k", color=C_PRO, fs=7)
note(ax, 20.0, Y4-0.25, "Phase 2", color=C_DEC, fs=8, style="normal")
note(ax, 20.0, Y4-0.52, "Student  40k iter", color=C_DEC, fs=7)


# ═════════════════════════════════════════════════════════════════════════════
# BAND 5 – ASYMMETRIC CRITIC   y = 0.85 – 2.65
# ═════════════════════════════════════════════════════════════════════════════
Y5   = 1.75
bg_box(ax, 0.3, 0.85, 18.8, 2.65, C_CRIT,
       "  Asymmetric Critic  (Sec. IV-B · training only · Teacher = Student architecture)")

BH5 = 0.68

rbox(ax,  1.9, Y5, 2.0, BH5, C_CRIT, "Critic Prop",
     "50D = base_vel(3)\n+hist(42)+cmd_critic(5)")
rbox(ax,  4.8, Y5, 2.3, BH5, C_CRIT, "TeacherPropEnc",
     "MLP(50→256→128)\nprop_emb  128D")
rbox(ax,  7.9, Y5, 2.1, BH5, C_CRIT, "GT Map (3ch)",
     "RayCaster 14×36\n(x,y,z) per cell")
rbox(ax, 11.1, Y5, 2.5, BH5, C_CRIT, "AME2Encoder",
     "same arch, separate weights\nmap_emb  192D")
rbox(ax, 14.0, Y5, 2.0, BH5, C_CRIT, "Contact (4D)",
     "per-foot\ncontact state")
rbox(ax, 17.2, Y5, 2.4, BH5+0.12, C_CRIT, "MoE  (4 experts)",
     "gate(contact) → weights\nΣ gateᵢ · expertᵢ(state)\n→ V(state)", bold=True)

harrow(ax,  2.90,  3.65, Y5, color=C_CRIT)
harrow(ax,  5.95,  6.85, Y5, color=C_CRIT)
harrow(ax,  8.95,  9.85, Y5, color=C_CRIT)
harrow(ax, 12.35, 13.00, Y5, color=C_CRIT)
harrow(ax, 15.00, 16.00, Y5, color=C_CRIT)

note(ax, 10.5, Y5-0.60,
     "L-R symmetry augmentation:  V = 0.5 · (V_orig + V_flip)  "
     "— applied to critic only, not actor  (Sec. IV-B)",
     color=C_CRIT, fs=6.8)


# ═════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═════════════════════════════════════════════════════════════════════════════
handles = [
    mpatches.Patch(color=C_MAP,  label="Perception / Mapping"),
    mpatches.Patch(color=C_PRO,  label="Proprioception  (student: LSIO  |  teacher: plain MLP)"),
    mpatches.Patch(color=C_ENC,  label="AME-2 Encoder  (CNN + CoordPosEmb + MHA)"),
    mpatches.Patch(color=C_DEC,  label="Decoder / Actions"),
    mpatches.Patch(color=C_CRIT, label="Asymmetric MoE Critic  (training only)"),
]
ax.legend(handles=handles, loc="lower center",
          bbox_to_anchor=(0.48, -0.015),
          ncol=3, fontsize=8,
          framealpha=0.88, edgecolor="#BBBBBB")


# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.03, 1, 1])
out = "docs/architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"Saved → {out}")
