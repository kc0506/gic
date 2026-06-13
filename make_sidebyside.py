import os, glob
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = '/tmp2/b10401006/ev-project/gic'
OUT = os.path.join(ROOT, 'figures_out')
os.makedirs(OUT, exist_ok=True)

def load_white(p):
    img = imageio.imread(p)
    if img.ndim == 2:
        img = np.stack([img]*3, -1)
    if img.shape[-1] == 4:
        rgb = img[..., :3].astype(np.float32); a = img[..., 3:4].astype(np.float32)/255.
        img = (rgb*a + 255.*(1-a)).astype(np.uint8)
    return img[..., :3]

# GT cam0 frames 0..13
gt = [load_white(os.path.join(ROOT, f'data/pacnerf/torus/data/r_0_{f}.png')) for f in range(14)]
H, W = gt[0].shape[:2]
# Pred frames
pred_paths = sorted(glob.glob(os.path.join(ROOT, 'output/pacnerf/torus/render/*.png')))
pred = [np.array(Image.fromarray(load_white(p)).resize((W, H))) for p in pred_paths]
print(f'GT {len(gt)} frames @ {W}x{H}, Pred {len(pred)} frames')

def label(arr, text):
    im = Image.fromarray(arr.copy()); d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
    except Exception:
        try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', 34)
        except Exception: font = ImageFont.load_default()
    d.rectangle([0,0,W,46], fill=(0,0,0)); d.text((12,6), text, fill=(255,255,255), font=font)
    return np.array(im)

# (1) side-by-side over the overlapping 14-frame window
sb = []
for f in range(14):
    l = label(gt[f], f'GT  (cam0)  t={f}')
    r = label(pred[f], f"Pred  E=0.99MPa nu=0.31  t={f}")
    sb.append(np.concatenate([l, np.full((H,4,3),255,np.uint8), r], axis=1))
imageio.mimsave(os.path.join(OUT, 'gt_vs_pred_sidebyside.mp4'), sb, fps=6, quality=8)
imageio.mimsave(os.path.join(OUT, 'gt_vs_pred_sidebyside.gif'), [s[::2,::2] for s in sb], fps=6)
print('saved gt_vs_pred_sidebyside.mp4/.gif')

# (2) full predicted trajectory (48 frames, incl. extrapolation beyond GT)
full = [label(pred[f], f"Pred (re-sim, E=0.99MPa nu=0.31)  t={f}") for f in range(len(pred))]
imageio.mimsave(os.path.join(OUT, 'pred_full48.mp4'), full, fps=12, quality=8)
print('saved pred_full48.mp4')

# (3) static contact-sheet: GT row vs Pred row at t=0,3,6,9,13
sel = [0,3,6,9,13]
gt_row = np.concatenate([np.array(Image.fromarray(gt[f]).resize((256,256))) for f in sel], axis=1)
pr_row = np.concatenate([np.array(Image.fromarray(pred[f]).resize((256,256))) for f in sel], axis=1)
sheet = np.concatenate([label_row(gt_row,'GT'), label_row(pr_row,'Pred')], axis=0) if False else None
# simpler: stack with side captions via PIL
sheet_im = Image.new('RGB', (256*len(sel)+90, 256*2+10), (255,255,255))
for i,f in enumerate(sel):
    sheet_im.paste(Image.fromarray(gt[f]).resize((256,256)), (90+i*256, 0))
    sheet_im.paste(Image.fromarray(pred[f]).resize((256,256)), (90+i*256, 256+10))
d = ImageDraw.Draw(sheet_im)
try: f2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
except Exception: f2 = ImageFont.load_default()
d.text((6,110), 'GT', fill=(0,0,0), font=f2); d.text((6,376), 'Pred', fill=(0,0,0), font=f2)
for i,f in enumerate(sel):
    d.text((90+i*256+100, 262), f't={f}', fill=(0,0,0), font=f2)
sheet_im.save(os.path.join(OUT, 'gt_vs_pred_grid.png'))
print('saved gt_vs_pred_grid.png')
