import numpy as np
import torch
from torch import nn
from torch.optim import _functional as F
import json
import copy
from PIL import Image
from random import randint
from tqdm import tqdm
import wandb

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import o3d_knn, setup_camera

from torch.optim.lr_scheduler import LinearLR

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

def get_loss(params, batch):
    rendervar = params2rendervar(params)
    # rendervar['means2D'].retain_grad()

    im, _, _, = Renderer(raster_settings=batch['cam'])(**rendervar)
    loss = torch.nn.functional.l1_loss(im, batch['im'])

    return loss

def load_params(path: str):
    params = torch.load(path)

    for v in params.values():
        v.requires_grad = False

    return params

class MLP(nn.Module):
    def __init__(self, in_dim, seq_len) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim + 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, in_dim)

        self.relu = nn.ReLU()

        self.emb = nn.Embedding(seq_len, in_dim)

    def dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def forward(self, x, t):
        B, D = x.shape

        x_ = x

        # e = self.emb(t).repeat(B, 1)
        e = self.dec2bin(t, 3).repeat(B, 1)

        x = torch.cat((x, e), dim=1)

        x = x # + e
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x =           self.fc6(x)

        return x_ + x

def train(seq: str):
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))
    seq_len = len(md['fn'])
    params = load_params('params.pth')
    
    mlp = MLP(7, seq_len).cuda()
    mlp_optimizer = torch.optim.Adam(params=mlp.parameters(), lr=1e-3)

    for t in range(1, seq_len, 1):
        dataset = get_dataset(t, md, seq)
        dataset_queue = []

        for i in tqdm(range(10_000)):
            X = get_batch(dataset_queue, dataset)

            delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.tensor(t).cuda())
            delta_means = delta[:,:3]
            delta_rotations = delta[:,3:]

            l = 0.01
            updated_params = copy.deepcopy(params)
            updated_params['means'] = updated_params['means'].detach()
            updated_params['means'] += delta_means * l
            updated_params['rotations'] = updated_params['rotations'].detach()
            updated_params['rotations'] += delta_rotations * l

            loss = get_loss(updated_params, X)

            wandb.log({
                f'loss-{t}': loss.item(),
            })

            loss.backward()

            mlp_optimizer.step()
            mlp_optimizer.zero_grad()

        dataset_queue = dataset.copy()
        losses = []
        while dataset_queue:
            with torch.no_grad():
                X = get_batch(dataset_queue, dataset)

                loss = get_loss(updated_params, X)
                losses.append(loss.item())

        wandb.log({
            f'mean-losses': sum(losses) / len(losses)
        })

def main():
    wandb.init(project="new-dynamic-gaussians")

    train('basketball')

if __name__ == '__main__':
    main()