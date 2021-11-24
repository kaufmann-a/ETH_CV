import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        )

        # TODO

    def forward(self, x):
        # x: [B,3,H,W]
        return self.layers(x)



class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        x = None #TODO Delete
        # TODO


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xy_hom = torch.stack((x, y, torch.ones_like(x)))
        xy_hom = torch.unsqueeze(xy_hom, 0).repeat(B, 1, 1).double()
        xy_hom_rot = torch.matmul(rot, xy_hom)
        rot_depth_xy_hom = xy_hom_rot.unsqueeze(2).repeat(1, 1, D, 1) * torch.unsqueeze(depth_values.view(B, 1, D), 3).repeat(1, 1, 1, H*W)
        p_xy_hom = rot_depth_xy_hom + trans.view(B, 3, 1, 1)
        negative_depth_mask = p_xy_hom[:, 2:] <= 1e-3
        p_xy_hom[:, 0:1][negative_depth_mask] = float(W)
        p_xy_hom[:, 1:2][negative_depth_mask] = float(H)
        p_xy_hom[:, 2:3][negative_depth_mask] = 1.0
        p_xy = p_xy_hom[:, :2, :, :] / p_xy_hom[:, 2:3, :, :]
        p_x_normalized = p_xy[:, 0, :, :] / ((W - 1) / 2) - 1
        p_y_normalized = p_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        p_xy_final = torch.stack((p_x_normalized, p_y_normalized), dim=3)


    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    warped_src_fea = F.grid_sample(
        src_fea.double(),
        p_xy_final.view(B, D, H * W, 2),
        align_corners=True
    ).view(B, C, D, H, W)

    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    w_B, w_C, w_D, w_H, w_W = warped_src_fea.shape
    warped_src_fea_v = warped_src_fea.view(w_B, G, w_C//G, w_D, w_H, w_W)

    r_B, r_C, r_H, r_W = ref_fea.shape
    ref_fea_v = ref_fea.view(r_B, G, r_C // G, 1, r_H, r_W)

    # out: [B,G,D,H,W]
    out = (warped_src_fea_v * ref_fea_v).mean(2)
    return out


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    p = None # TODO Delete
    # TODO

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    depth_est = None # TODO Delete
    # TODO
