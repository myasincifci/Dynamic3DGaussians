import numpy as np
import torch
from torch import nn
import json
import copy
from PIL import Image
from random import randint
from tqdm import tqdm
import wandb

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import o3d_knn, setup_camera


def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    return params, variables

def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params

def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params

def densify(params, variables, optimizer, i):
    if i <= 5000:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 500) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means'].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = 2  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            stds = torch.exp(params['scales'])[to_split].repeat(n, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['rotations'][to_split]).repeat(n, 1, 1)
            new_params['means'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['scales'] = torch.log(torch.exp(new_params['scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            remove_threshold = 0.25 if i == 5000 else 0.005
            to_remove = (torch.sigmoid(params['opacities']) < remove_threshold).squeeze()
            if i >= 3000:
                big_points_ws = torch.exp(params['scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        if i > 0 and i % 3000 == 0:
            new_params = {'opacities': inverse_sigmoid(torch.ones_like(params['opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables

def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def initialize_params(seq: str, md):
    pointcloud = np.load(f"./data/{seq}/init_pt_cld.npz")["data"]
    N = pointcloud.shape[0]

    sq_dist, _ = o3d_knn(pointcloud[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    
    params = {
        'means': pointcloud[:,:3],
        'colors': pointcloud[:,3:6],
        'scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'rotations': np.tile([1, 0, 0, 0], (N, 1)),
        'opacities': np.zeros((N, 1)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}

    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    vars = {
        'means2D_gradient_accum': torch.zeros(params['means'].shape[0]).cuda().float(),
        'denom': torch.zeros(params['means'].shape[0]).cuda().float(),
        'max_2D_radius': torch.zeros(params['means'].shape[0]).cuda().float(),
        'scene_radius': scene_radius,
    }
    
    return params, vars

def params2rendervar(params):
    rendervar = {
        'means3D': params['means'],
        'colors_precomp': params['colors'],
        'rotations': torch.nn.functional.normalize(params['rotations']),
        'opacities': torch.sigmoid(params['opacities']),
        'scales': torch.exp(params['scales']),
        'means2D': torch.zeros_like(params['means'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def get_loss(params, batch, vars):
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    im, radius, _, = Renderer(raster_settings=batch['cam'])(**rendervar)
    loss = torch.nn.functional.l1_loss(im, batch['im'])

    vars['means2D'] = rendervar['means2D']

    seen = radius > 0
    vars['max_2D_radius'][seen] = torch.max(radius[seen], vars['max_2D_radius'][seen])
    vars['seen'] = seen
    return loss, vars

def initialize_optimizer(params, variables):
    lrs = {
        'means': 0.00016 * variables['scene_radius'],
        'colors': 0.0025,
        'rotations': 0.001,
        'opacities': 0.05,
        'scales': 0.001,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, out_dim)


    def forward(self, x):
        x_ = x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x_ + x

def train(seq: str):
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))
    seq_len = len(md['fn'])
    params, vars = initialize_params(seq, md)

    # optimizer = torch.optim.Adam(params=[v for v in params.values()], lr=1e-4)
    optimizer = initialize_optimizer(params, vars)
    
    for t in range(seq_len):
        dataset = get_dataset(t, md, seq)
        dataset_queue = []

        if t == 0:
            for i in tqdm(range(30_000)):
                X = get_batch(dataset_queue, dataset)

                loss, vars = get_loss(params, X, vars)

                wandb.log({
                    f'loss-{t}': loss.item(),
                    'num_gaussians': params['means'].shape[0]
                })

                loss.backward()

                with torch.no_grad():
                    params, vars = densify(params, vars, optimizer, i)
                    optimizer.step()
                    optimizer.zero_grad()
        else:
            torch.save(params, 'params.pth')
            # mlp = MLP(7, 7).cuda()
            # mlp_optimizer = torch.optim.Adam(params=mlp.parameters(), lr=1e-4)

            # for i in tqdm(range(1_000)):
            #     X = get_batch(dataset_queue, dataset)

            #     loss, vars = get_loss(params, X, vars)

            #     wandb.log({
            #         f'loss-{t}': loss.item(),
            #     })

            #     loss.backward()

            #     with torch.no_grad():
            #         params, vars = densify(params, vars, optimizer, i)
            #         mlp_optimizer.step()
            #         mlp_optimizer.zero_grad()

def main():
    wandb.init(project="new-dynamic-gaussians")

    train('basketball')

if __name__ == '__main__':
    main()