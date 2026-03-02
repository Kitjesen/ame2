import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --- Color scheme ---
C_ENV = '#2563EB'       # blue - environment/observations
C_ENV_LIGHT = '#DBEAFE'
C_MAP = '#7C3AED'       # purple - mapping pipeline
C_MAP_LIGHT = '#EDE9FE'
C_ACTOR = '#059669'     # green - actor path
C_ACTOR_LIGHT = '#D1FAE5'
C_CRITIC = '#D97706'    # orange/amber - critic path
C_CRITIC_LIGHT = '#FEF3C7'
C_ARROW = '#374151'
C_TEXT = '#111827'
C_DIM = '#6B7280'
C_BG = '#F8FAFC'
C_WHITE = '#FFFFFF'
C_BORDER = '#374151'

fig, ax = plt.subplots(1, 1, figsize=(28, 20), dpi=150)
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 28)
ax.set_ylim(0, 20)
ax.axis('off')

# ============================================================
# Helper functions
# ============================================================
def draw_box(x, y, w, h, label, color, fontsize=7.5, text_color=C_TEXT,
             edge_color=None, lw=1.5, alpha=1.0, sublabel=None, sublabel_fs=6,
             bold=False):
    if edge_color is None:
        edge_color = color
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.12", facecolor=color, edgecolor=edge_color,
        linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.12, label, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.18, sublabel, ha='center', va='center',
                fontsize=sublabel_fs, color=C_DIM, fontstyle='italic', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=3)
    return (x, y, w, h)

def draw_section_box(x, y, w, h, label, color, alpha=0.12, fontsize=9):
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.2", facecolor=color, edgecolor=color,
        linewidth=2, alpha=alpha, zorder=0)
    ax.add_patch(box)
    ax.text(x + 0.3, y + h - 0.3, label, ha='left', va='top',
            fontsize=fontsize, color=color, fontweight='bold', zorder=1)

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.2, style='->', connectionstyle=None):
    kw = dict(arrowstyle=style, mutation_scale=12, color=color, linewidth=lw, zorder=4)
    if connectionstyle:
        kw['connectionstyle'] = connectionstyle
    a = FancyArrowPatch((x1, y1), (x2, y2), **kw)
    ax.add_patch(a)

def dim_label(x, y, text, fontsize=5.5, color=C_DIM):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=color,
            fontstyle='italic', zorder=5,
            bbox=dict(boxstyle='round,pad=0.08', facecolor=C_WHITE, edgecolor='none', alpha=0.8))

# ============================================================
# TITLE
# ============================================================
ax.text(14, 19.5, 'AME-2: Adaptive Map Encoder Architecture', ha='center', va='center',
        fontsize=16, fontweight='bold', color=C_TEXT)
ax.text(14, 19.1, 'End-to-End Data Flow — Teacher & Student Pipelines', ha='center', va='center',
        fontsize=10, color=C_DIM)

# ============================================================
# SECTION 1: Isaac Lab Environment (top)
# ============================================================
draw_section_box(0.3, 17.0, 27.4, 1.8, 'Isaac Lab Environment', C_ENV)

draw_box(1.0, 17.3, 4.5, 0.9, 'obs_buf["policy"]', C_ENV_LIGHT, sublabel='(B, 48)', bold=True, edge_color=C_ENV)
draw_box(6.5, 17.3, 5.5, 0.9, 'obs_buf["teacher_privileged"]', C_ENV_LIGHT, sublabel='(B, 1593)', bold=True, edge_color=C_ENV)
draw_box(13.0, 17.3, 5.0, 0.9, 'obs_buf["teacher_map"]', C_ENV_LIGHT, sublabel='(B, 1512)', bold=True, edge_color=C_ENV)
draw_box(19.0, 17.3, 4.5, 0.9, 'obs_buf["critic_extra"]', C_ENV_LIGHT, sublabel='(B, 5)', bold=True, edge_color=C_ENV)
draw_box(24.5, 17.3, 3.0, 0.9, 'robot_pose', C_ENV_LIGHT, sublabel='(B, 3)', bold=True, edge_color=C_ENV)

# ============================================================
# SECTION 2: AME2MapEnvWrapper (middle)
# ============================================================
draw_section_box(0.3, 11.3, 27.4, 5.5, 'AME2MapEnvWrapper — Observation Processing', C_MAP)

# --- Mapping pipeline (left-center) ---
# raw_scan extraction
draw_box(6.5, 15.8, 3.5, 0.6, 'raw_scan[:, :1581]', C_MAP_LIGHT, sublabel='(B,1,31,51)', edge_color=C_MAP, fontsize=6.5)
arrow(9.25, 17.3, 8.25, 16.4, C_MAP)

# MappingNet
draw_box(5.5, 14.5, 5.5, 1.0, 'MappingNet (U-Net)', C_MAP_LIGHT, edge_color=C_MAP, bold=True,
         sublabel='Conv→Pool→Upsample→Gate')
arrow(8.25, 15.8, 8.25, 15.5, C_MAP)
dim_label(9.2, 15.65, 'noise + dropout')

# MappingNet outputs
draw_box(5.5, 13.4, 2.5, 0.6, 'elev', C_MAP_LIGHT, sublabel='(B,1,31,51)', edge_color=C_MAP, fontsize=6.5)
draw_box(8.5, 13.4, 2.5, 0.6, 'log_var', C_MAP_LIGHT, sublabel='(B,1,31,51)', edge_color=C_MAP, fontsize=6.5)
arrow(7.5, 14.5, 6.75, 14.0, C_MAP)
arrow(9.0, 14.5, 9.75, 14.0, C_MAP)

# WTA Map Fusion
draw_box(5.0, 12.0, 6.5, 1.0, 'WTAMapFusion', C_MAP_LIGHT, edge_color=C_MAP, bold=True,
         sublabel='Global Map (400×400) + WTA Update + Crop')
arrow(6.75, 13.4, 7.5, 13.0, C_MAP)
arrow(9.75, 13.4, 8.75, 13.0, C_MAP)
# pose arrow into WTA
arrow(26.0, 17.3, 26.0, 16.2, C_MAP, style='->')
arrow(26.0, 16.2, 11.5, 12.7, C_MAP, connectionstyle='arc3,rad=-0.15')
dim_label(19.0, 15.0, 'poses (B,3)')

# WTA outputs
draw_box(3.5, 11.0, 4.0, 0.7, 'student_map', C_MAP_LIGHT, sublabel='(B, 4, 14, 36)', edge_color=C_MAP, fontsize=6.5, bold=True)
draw_box(8.0, 11.0, 4.0, 0.7, 'teacher_map', C_MAP_LIGHT, sublabel='(B, 3, 14, 36)', edge_color=C_MAP, fontsize=6.5, bold=True)
arrow(7.0, 12.0, 5.5, 11.7, C_MAP)
arrow(9.5, 12.0, 10.0, 11.7, C_MAP)

# GT map reshape
draw_box(13.0, 15.5, 3.5, 0.7, 'Reshape + Normals', C_MAP_LIGHT, sublabel='3×14×36', edge_color=C_MAP, fontsize=6.5)
arrow(15.5, 17.3, 14.75, 16.2, C_MAP)
arrow(14.75, 15.5, 10.5, 12.9, C_MAP, connectionstyle='arc3,rad=0.2')
dim_label(12.0, 14.5, 'GT → teacher_map')

# --- Proprioception slicing (right side) ---
# prop slicing
draw_box(1.0, 15.3, 2.0, 0.7, 'Slice [:, 3:45]', C_ENV_LIGHT, sublabel='hist_step', edge_color=C_ENV, fontsize=6)
arrow(3.25, 17.3, 2.0, 16.0, C_ENV)

draw_box(1.0, 14.2, 2.0, 0.7, 'LSIO Buffer', C_ENV_LIGHT, sublabel='(B, 20, 42)', edge_color=C_ENV, fontsize=6)
arrow(2.0, 15.3, 2.0, 14.9, C_ENV)

draw_box(1.0, 13.1, 2.0, 0.5, 'commands', C_ENV_LIGHT, sublabel='(B, 3)', edge_color=C_ENV, fontsize=6)
arrow(3.25, 17.3, 2.0, 13.6, C_ENV, connectionstyle='arc3,rad=0.2')
dim_label(1.5, 16.7, '[:, -3:]')

# critic_prop construction
draw_box(18.5, 15.3, 3.5, 0.7, 'Concat: prop_base + critic_extra', C_CRITIC_LIGHT, sublabel='(B, 50)', edge_color=C_CRITIC, fontsize=6)
arrow(3.25, 17.3, 19.0, 16.0, C_ENV, connectionstyle='arc3,rad=-0.08')
arrow(21.25, 17.3, 21.0, 16.0, C_CRITIC)
dim_label(16.5, 16.3, '[:, :45]  (B,45)')

# contact processing
draw_box(18.5, 13.8, 3.5, 0.7, 'Contact Threshold', C_CRITIC_LIGHT, sublabel='(B, 4) binary', edge_color=C_CRITIC, fontsize=6)
arrow(9.25, 17.3, 20.25, 14.5, C_CRITIC, connectionstyle='arc3,rad=-0.12')
dim_label(15.0, 15.8, 'priv[:, 1581:] → norm → >1N')

# ============================================================
# SECTION 3: Actor Path (left)
# ============================================================
draw_section_box(0.3, 1.5, 13.2, 9.5, 'Actor Path — AME2Policy', C_ACTOR)

# Teacher PropEncoder
draw_box(0.8, 9.5, 4.5, 0.8, 'TeacherPropEncoder', C_ACTOR_LIGHT, edge_color=C_ACTOR, bold=True,
         sublabel='Linear(48→256)→ELU→Linear(256→128)→ELU')
arrow(3.25, 11.0, 3.05, 10.3, C_ACTOR)
dim_label(4.0, 10.7, 'prop (B,48)')

# Student PropEncoder
draw_box(0.8, 8.2, 4.5, 0.8, 'StudentPropEncoder + LSIO', C_ACTOR_LIGHT, edge_color=C_ACTOR, bold=True,
         sublabel='Conv1d(42→32→16) + MLP(187→256→128)')
arrow(2.0, 11.0, 2.0, 10.5, C_ACTOR, style='->')
# draw dashed lines from history & commands
arrow(2.0, 13.1, 2.0, 10.5, C_ACTOR, style='->')
dim_label(0.8, 12.2, 'history\n(B,20,42)')
dim_label(0.7, 10.6, 'commands (B,3)')

# prop_emb output
draw_box(1.5, 7.2, 3.0, 0.6, 'prop_emb', C_ACTOR_LIGHT, sublabel='(B, 128)', edge_color=C_ACTOR, bold=True)
arrow(3.05, 9.5, 3.0, 7.8, C_ACTOR)
arrow(3.05, 8.2, 3.0, 7.8, C_ACTOR)

# AME2Encoder
draw_box(5.5, 5.8, 7.5, 4.5, '', C_ACTOR_LIGHT, edge_color=C_ACTOR, alpha=0.3)
ax.text(9.25, 10.0, 'AME2Encoder', ha='center', va='center', fontsize=9,
        color=C_ACTOR, fontweight='bold', zorder=3)

# CNN
draw_box(5.8, 9.0, 3.2, 0.6, 'CNN (Conv2d×2)', C_WHITE, sublabel='→ (B,64,14,36)', edge_color=C_ACTOR, fontsize=6)
arrow(5.5, 11.0, 7.4, 9.6, C_ACTOR)
dim_label(4.6, 10.3, 'map')

# Positional Embedding
draw_box(9.3, 9.0, 3.2, 0.6, 'CoordPosEmbed', C_WHITE, sublabel='MLP(2→64→64)', edge_color=C_ACTOR, fontsize=6)

# Fusion MLP
draw_box(5.8, 8.0, 3.2, 0.6, 'Fusion MLP', C_WHITE, sublabel='Linear(128→64)→ELU', edge_color=C_ACTOR, fontsize=6)
arrow(7.4, 9.0, 7.4, 8.6, C_ACTOR)
arrow(10.9, 9.0, 8.5, 8.6, C_ACTOR, connectionstyle='arc3,rad=0.2')

# Global branch
draw_box(9.3, 8.0, 3.2, 0.6, 'Global Branch', C_WHITE, sublabel='MLP→MaxPool→(B,128)', edge_color=C_ACTOR, fontsize=6)
arrow(7.4, 8.0, 10.9, 8.6, C_ACTOR, connectionstyle='arc3,rad=-0.3')

# Query
draw_box(5.8, 7.0, 3.2, 0.6, 'Query Proj', C_WHITE, sublabel='cat(global,prop)→Linear→(B,1,64)', edge_color=C_ACTOR, fontsize=6)
arrow(10.9, 8.0, 8.0, 7.6, C_ACTOR, connectionstyle='arc3,rad=0.15')
arrow(4.5, 7.5, 5.8, 7.3, C_ACTOR)

# MHA
draw_box(9.3, 7.0, 3.2, 0.6, 'MultiHead Attn', C_WHITE, sublabel='16 heads, 4 dim/head', edge_color=C_ACTOR, fontsize=6)
arrow(9.0, 7.3, 9.3, 7.3, C_ACTOR)
arrow(7.4, 8.0, 10.9, 7.6, C_ACTOR, connectionstyle='arc3,rad=-0.2')
dim_label(10.0, 6.7, 'K=V: local (B,504,64)')

# map_emb output
draw_box(7.5, 6.0, 3.5, 0.6, 'map_emb', C_ACTOR_LIGHT, sublabel='cat(attn,global) = (B, 192)', edge_color=C_ACTOR, bold=True, fontsize=6.5)
arrow(10.9, 7.0, 9.25, 6.6, C_ACTOR)
arrow(10.9, 8.0, 10.5, 6.6, C_ACTOR, connectionstyle='arc3,rad=-0.2')

# Concatenation
draw_box(3.5, 4.5, 6.5, 0.7, 'Concatenate', C_ACTOR_LIGHT, sublabel='cat(map_emb, prop_emb) = (B, 320)', edge_color=C_ACTOR, bold=True)
arrow(3.0, 7.2, 5.5, 5.2, C_ACTOR, connectionstyle='arc3,rad=0.2')
arrow(9.25, 6.0, 7.5, 5.2, C_ACTOR)

# Decoder
draw_box(3.0, 3.2, 7.5, 0.8, 'Decoder MLP', C_ACTOR_LIGHT, edge_color=C_ACTOR, bold=True,
         sublabel='Linear(320→512)→ELU→Linear(512→256)→ELU→Linear(256→12)')
arrow(6.75, 4.5, 6.75, 4.0, C_ACTOR)

# Actions output
draw_box(5.0, 2.0, 3.5, 0.7, 'Actions', C_ACTOR_LIGHT, sublabel='(B, 12)', edge_color=C_ACTOR, bold=True, fontsize=9)
arrow(6.75, 3.2, 6.75, 2.7, C_ACTOR)

# ============================================================
# SECTION 4: Critic Path (right)
# ============================================================
draw_section_box(14.0, 1.5, 13.5, 9.5, 'Critic Path — AsymmetricCritic', C_CRITIC)

# CriticMapEncoder
draw_box(14.5, 9.0, 5.5, 1.0, 'CriticMapEncoder', C_CRITIC_LIGHT, edge_color=C_CRITIC, bold=True,
         sublabel='Conv2d(3→32→64)→AvgPool→Linear(64→192)→ELU')
arrow(10.0, 11.0, 17.25, 10.0, C_CRITIC, connectionstyle='arc3,rad=-0.15')
dim_label(13.0, 10.3, 'map_teacher\n(B,3,14,36)')

# Critic map_emb
draw_box(15.2, 8.0, 3.5, 0.6, 'map_emb', C_CRITIC_LIGHT, sublabel='(B, 192)', edge_color=C_CRITIC, bold=True)
arrow(17.25, 9.0, 16.95, 8.6, C_CRITIC)

# TeacherPropEncoder for critic
draw_box(20.5, 9.0, 5.5, 1.0, 'TeacherPropEncoder', C_CRITIC_LIGHT, edge_color=C_CRITIC, bold=True,
         sublabel='Linear(50→256)→ELU→Linear(256→128)→ELU')
arrow(20.25, 15.3, 23.25, 10.0, C_CRITIC, connectionstyle='arc3,rad=-0.1')
dim_label(22.0, 12.5, 'critic_prop (B,50)')

# Critic prop_emb
draw_box(21.5, 8.0, 3.0, 0.6, 'prop_emb', C_CRITIC_LIGHT, sublabel='(B, 128)', edge_color=C_CRITIC, bold=True)
arrow(23.25, 9.0, 23.0, 8.6, C_CRITIC)

# State concatenation
draw_box(16.0, 7.0, 7.0, 0.7, 'state = cat(map_emb, prop_emb)', C_CRITIC_LIGHT, sublabel='(B, 320)', edge_color=C_CRITIC, bold=True)
arrow(16.95, 8.0, 18.5, 7.7, C_CRITIC)
arrow(23.0, 8.0, 21.0, 7.7, C_CRITIC)

# Gate MLP
draw_box(14.5, 5.5, 4.0, 0.8, 'Gate MLP', C_CRITIC_LIGHT, edge_color=C_CRITIC, bold=True,
         sublabel='Linear(4→64)→ELU→Linear(64→4)→Softmax')
arrow(20.25, 13.8, 16.5, 6.3, C_CRITIC, connectionstyle='arc3,rad=0.15')
dim_label(15.5, 10.5, 'contact (B,4)')

# Gate output
draw_box(14.8, 4.5, 3.3, 0.6, 'weights', C_CRITIC_LIGHT, sublabel='(B, 4)', edge_color=C_CRITIC)
arrow(16.5, 5.5, 16.45, 5.1, C_CRITIC)

# Expert MLPs
draw_box(19.5, 5.0, 6.5, 1.5, '4 Expert MLPs', C_CRITIC_LIGHT, edge_color=C_CRITIC, bold=True,
         sublabel='each: Linear(320→256)→ELU→Linear(256→128)→ELU→Linear(128→1)')
arrow(19.5, 7.0, 22.75, 6.5, C_CRITIC)

# Expert output
draw_box(20.5, 4.0, 4.5, 0.6, 'expert_vals', C_CRITIC_LIGHT, sublabel='(B, 4, 1)', edge_color=C_CRITIC)
arrow(22.75, 5.0, 22.75, 4.6, C_CRITIC)

# Mixture
draw_box(17.5, 3.0, 7.0, 0.7, 'Weighted Mixture', C_CRITIC_LIGHT, sublabel='Σ(weights × expert_vals)', edge_color=C_CRITIC, bold=True)
arrow(16.45, 4.5, 19.5, 3.7, C_CRITIC, connectionstyle='arc3,rad=0.1')
arrow(22.75, 4.0, 21.5, 3.7, C_CRITIC)

# L-R Symmetry
draw_box(17.5, 2.0, 7.0, 0.7, 'L-R Symmetry Augmentation', C_CRITIC_LIGHT, sublabel='0.5 × (v_orig + v_flip)', edge_color=C_CRITIC, bold=True)
arrow(21.0, 3.0, 21.0, 2.7, C_CRITIC)

# Value output
draw_box(19.5, 1.0, 3.0, 0.7, 'Value', C_CRITIC_LIGHT, sublabel='(B, 1)', edge_color=C_CRITIC, bold=True, fontsize=9)
arrow(21.0, 2.0, 21.0, 1.7, C_CRITIC)

# ============================================================
# Legend
# ============================================================
legend_x, legend_y = 23.5, 11.5
ax.text(legend_x, legend_y + 0.9, 'Legend', fontsize=8, fontweight='bold', color=C_TEXT)
for i, (label, color) in enumerate([
    ('Environment / Observations', C_ENV),
    ('Mapping Pipeline', C_MAP),
    ('Actor Path', C_ACTOR),
    ('Critic Path', C_CRITIC),
]):
    box = FancyBboxPatch((legend_x, legend_y - i*0.5), 0.4, 0.3,
        boxstyle="round,pad=0.05", facecolor=color, edgecolor=color, linewidth=1, zorder=2)
    ax.add_patch(box)
    ax.text(legend_x + 0.6, legend_y - i*0.5 + 0.15, label, fontsize=6.5,
            color=C_TEXT, va='center', zorder=3)

plt.savefig('/home/user/ame2/ame2_architecture.png', dpi=150, bbox_inches='tight',
            facecolor=C_BG, edgecolor='none')
print("Image saved to /home/user/ame2/ame2_architecture.png")
