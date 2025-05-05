# relight_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T
import matplotlib.pyplot as plt
from typing import List
from PIL import Image
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics
import wandb


class PoseGTImageDataset(Dataset):
    def __init__(self, gt_list: List[Image.Image], pose_list: List[dict]):
        assert len(gt_list) == len(pose_list), "should keep the same length"
        self.transform = T.Compose([
            T.Resize((512, 512)),  # Resize to 512x512
            T.ToTensor()
        ])
        self.gts = [self.transform(gt.convert('RGB')) for gt in gt_list]
        self.poses = pose_list

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        return self.gts[idx], self.poses[idx]


def get_all_trainable_params(gs):
    params = []
    if gs._xyz is not None:
        gs._xyz = torch.nn.Parameter(gs._xyz, requires_grad=False)
        params.append(gs._xyz)

    if gs._features_dc is not None:
        gs._features_dc = torch.nn.Parameter(gs._features_dc, requires_grad=True)
        params.append(gs._features_dc)

    if gs._features_rest is not None:
        gs._features_rest = torch.nn.Parameter(gs._features_rest, requires_grad=True)
        params.append(gs._features_rest)

    if gs._opacity is not None:
        gs._opacity = torch.nn.Parameter(gs._opacity, requires_grad=False)
        params.append(gs._opacity)

    if gs._scaling is not None:
        gs._scaling = torch.nn.Parameter(gs._scaling, requires_grad=True)
        params.append(gs._scaling)

    if gs._rotation is not None:
        gs._rotation = torch.nn.Parameter(gs._rotation, requires_grad=True)
        params.append(gs._rotation)

    return params



class RelitTrainer:
    def __init__(self, gs, gt_images: List[Image.Image], poses: List[dict],
                 render_type='default',
                 epochs=5, lr=1e-3, batch_size=4, loss_type='mse'):
        self.gs = gs
        self.gt_images = gt_images
        self.poses = poses
        self.render_type = render_type
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = PoseGTImageDataset(gt_images, poses)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.criterion = nn.MSELoss() if self.loss_type == 'mse' else nn.L1Loss()
        # self.optimizer = optim.Adam(get_trainable_appearance_params(self.gs), lr=self.lr)
        # breakpoint()
        self.params = get_all_trainable_params(self.gs)
        self.optimizer = optim.Adam(self.params, lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)


    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            # breakpoint()
            for gt, pose in self.loader:
                gt = gt[0].to(self.device) # batch_size only = 1
                # print("type of gt:", type(gt))
                # print("type of pose:", type(pose))
                extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(pose['yaw'].item(), pose['pitch'].item(), pose['radius'].item(), pose['fov'].item())
                render = get_renderer(self.gs)
                pred = render.render(self.gs, extr, intr)['color']
                # print("type of pred:", type(pred))

                # pred = self.gs.render(x, render_type=self.render_type)
                # print("shape of pred:", pred.shape)
                # print("shape of gt:", gt.shape)
                loss = self.criterion(pred, gt)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {total_loss/len(self.loader):.4f}")

            # wandb record
            avg_loss = total_loss / len(self.loader)
            wandb.log({"loss": avg_loss, "epoch": epoch + 1})

            pred_img = pred.permute(1, 2, 0).detach().cpu().numpy()
            gt_img = gt.permute(1, 2, 0).detach().cpu().numpy()
            images = [
                wandb.Image(pred_img, caption="Rendered"),
                wandb.Image(gt_img, caption="GT")
            ]
            wandb.log({"Comparison": images, "epoch": epoch + 1})

            # Log the current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})

            # self._compare(pred, gt)

            self.scheduler.step()