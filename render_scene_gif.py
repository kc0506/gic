import os, torch, numpy as np
import imageio.v2 as imageio
from argparse import ArgumentParser
from gaussian_renderer import render, GaussianModel
from scene import Scene, DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state

OUT = '/tmp2/b10401006/ev-project/gic/figures_out'
os.makedirs(OUT, exist_ok=True)
SCALE = 1.0

parser = ArgumentParser()
model = ModelParams(parser)
pipeline = PipelineParams(parser)
op = OptimizationParams(parser)
parser.add_argument("--config_file", default='config/pacnerf/torus.json', type=str)
gs_args, phys_args = get_combined_args(parser)
safe_state(True)

dataset = model.extract(gs_args)
pipe = pipeline.extract(gs_args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=40000, shuffle=False, resolution_scales=[SCALE])
deform = DeformModel(dataset); deform.load_weights(dataset.model_path)
bg = torch.tensor([1,1,1], dtype=torch.float32, device='cuda')  # white bg

# test cameras = cam0 across frames (clean single-view sequence)
views = scene.getTestCameras(scale=SCALE)
if len(views) == 0:
    views = scene.getTrainCameras(scale=SCALE)
views = sorted(views, key=lambda v: v.fid.item())
fids = sorted(set(v.fid.item() for v in views))
print(f"Gaussians={gaussians.get_xyz.shape[0]}, test views={len(views)}, unique fids={len(fids)}")

xyz_canonical = gaussians.get_xyz.detach()

def render_at(view, fid_val, deformed=True):
    if deformed:
        fid = torch.tensor(fid_val, device='cuda')
        time_input = fid.unsqueeze(0).expand(1, -1)  # [1,1], deform broadcasts to N (mirrors prepare_gt)
        d_xyz, d_rot, d_scl = deform.step(xyz_canonical, time_input)
    else:
        d_xyz = d_rot = d_scl = 0.0
    out = render(view, gaussians, pipe, bg, d_xyz, d_rot, d_scl, False)
    img = out["render"].clamp(0,1).detach().cpu().numpy()
    return (np.transpose(img, (1,2,0)) * 255).astype(np.uint8)

# pick one camera (first), render its frames over time (dynamic GS reconstruction)
cam0_views = [v for v in views]
# group: one view per fid (test set is single camera)
frames = []
with torch.no_grad():
    for fid_val in fids:
        v = next(v for v in views if abs(v.fid.item()-fid_val) < 1e-6)
        frames.append(render_at(v, fid_val, deformed=True))
print(f"rendered {len(frames)} deformed frames, shape {frames[0].shape}")
imageio.mimsave(os.path.join(OUT,'gs_recon_cam0.gif'), [f[::2,::2] for f in frames], fps=6)
imageio.mimsave(os.path.join(OUT,'gs_recon_cam0.mp4'), frames, fps=6, quality=8)

# canonical rest-shape still (frame-0 camera, no deform)
with torch.no_grad():
    canon = render_at(views[0], fids[0], deformed=False)
imageio.imwrite(os.path.join(OUT,'gs_canonical_restshape.png'), canon)
print("saved gs_recon_cam0.gif/.mp4 + gs_canonical_restshape.png")
