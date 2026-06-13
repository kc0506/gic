import os, re, glob
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/tmp2/b10401006/ev-project/gic'
SCENE = 'torus'
OUT = os.path.join(ROOT, 'figures_out')
os.makedirs(OUT, exist_ok=True)

# ---------- 1) GT video (cam0, frames 0..13) ----------
gt_dir = os.path.join(ROOT, f'data/pacnerf/{SCENE}/data')
gt_frames = []
for f in range(14):
    p = os.path.join(gt_dir, f'r_0_{f}.png')
    img = imageio.imread(p)
    if img.shape[-1] == 4:  # RGBA -> composite on white
        rgb = img[..., :3].astype(np.float32)
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
    gt_frames.append(img[..., :3])
imageio.mimsave(os.path.join(OUT, 'gt_cam0.mp4'), gt_frames, fps=8, quality=8)
print('saved gt_cam0.mp4', gt_frames[0].shape, 'x', len(gt_frames))

# ---------- 2) loss / param convergence curves from training log ----------
LOG = '/tmp/claude-70910/-tmp2-b10401006-PhysDreamer/994c6f9e-9d65-43c1-b6fc-3acb38ec4809/tasks/b8mb05dzb.output'
text = open(LOG, errors='ignore').read().replace('\r', '\n')

# Geometry loss / image loss per forward step (whole run, in order)
geo = [float(m) for m in re.findall(r'Geometry loss ([0-9.eE+-]+), image loss', text)]
img = [float(m) for m in re.findall(r'Geometry loss [0-9.eE+-]+, image loss ([0-9.eE+-]+)', text)]
# Physics-stage param trajectories
E   = [float(m) for m in re.findall(r'Youngs modulus: ([0-9.eE+-]+)', text)]
nu  = [float(m) for m in re.findall(r'Poisson ratio: ([0-9.eE+-]+)', text)]

# velocity stage = first len-after split; physics stage is where E/nu printed.
# Number of velocity-stage forward steps = total geo - physics steps. Physics prints E each iter.
n_phys = len(E)
n_vel = len(geo) - n_phys
print(f'forward steps: total={len(geo)} vel={n_vel} phys={n_phys}, E pts={len(E)} nu pts={len(nu)}')

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
# (a) velocity-stage geometry loss
axs[0].plot(range(1, n_vel+1), geo[:n_vel], color='tab:blue')
axs[0].set_title('Stage 1: velocity estimation\n(geometry loss)')
axs[0].set_xlabel('iteration'); axs[0].set_ylabel('geometry loss'); axs[0].grid(alpha=.3)
# (b) physics-stage total loss
gp = np.array(geo[n_vel:]); ip = np.array(img[n_vel:]) if len(img) >= len(geo) else np.zeros_like(gp)
axs[1].plot(range(1, len(gp)+1), gp, label='geometry', color='tab:blue')
if ip.any():
    axs[1].plot(range(1, len(ip)+1), ip, label='image', color='tab:orange')
    axs[1].plot(range(1, len(gp)+1), gp+ip, label='total', color='k', lw=1.5)
axs[1].set_title('Stage 2: physical-parameter estimation\n(loss)')
axs[1].set_xlabel('iteration'); axs[1].set_ylabel('loss'); axs[1].legend(); axs[1].grid(alpha=.3)
# (c) E and nu convergence (physics stage), with GIC-paper GT reference lines
GT_E, GT_NU = 1.0, 0.3  # GIC Table 1 GT for torus: E=1e6 Pa (=1.0 MPa), nu=0.3
ax = axs[2]; it = range(1, len(E)+1)
l1, = ax.plot(it, np.array(E)/1e6, color='tab:red', label=f"E est (final {E[-1]/1e6:.2f} MPa)")
g1 = ax.axhline(GT_E, ls='--', color='tab:red', alpha=.7, label=f"E GT = {GT_E:.1f} MPa")
ax.set_xlabel('iteration'); ax.set_ylabel("Young's modulus (MPa)", color='tab:red')
ax.tick_params(axis='y', labelcolor='tab:red'); ax.set_ylim(0, 1.7)
ax2 = ax.twinx()
l2, = ax2.plot(range(1, len(nu)+1), nu, color='tab:green', label=f"nu est (final {nu[-1]:.3f})")
g2 = ax2.axhline(GT_NU, ls='--', color='tab:green', alpha=.7, label=f"nu GT = {GT_NU:.1f}")
ax2.set_ylabel('Poisson ratio', color='tab:green'); ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.set_ylim(0, 0.4)
ax.set_title('Estimated parameters vs GT (GIC Table 1)')
ax.legend(handles=[l1, g1, l2, g2], loc='lower right', fontsize=8); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'convergence.png'), dpi=110)
print('saved convergence.png')
print(f'FINAL  E={E[-1]:.0f} Pa  nu={nu[-1]:.4f}')
