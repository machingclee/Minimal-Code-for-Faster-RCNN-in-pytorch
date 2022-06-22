import numpy as np
import torch
from src import config
from src.device import device


class AnchorGenerator():
    anchors = None

    def __init__(self):
        self.anchors = None
        self.base_anchor = None  # centered at (0, 0)
        self.grid_nums = config.grid_nums

    def get_anchors(self):
        if AnchorGenerator.anchors is not None:
            return AnchorGenerator.anchors

        ratios = torch.as_tensor(config.anchor_ratios)
        scales = torch.as_tensor(config.anchor_scales)

        hs = scales[None, ...] * torch.sqrt(ratios)[..., None]
        hs = hs.view((len(ratios) * len(scales),))

        ws = scales[None, ...] / torch.sqrt(ratios)[..., None]
        ws = ws.view((len(ratios) * len(scales),))

        hs = hs[..., None]
        ws = ws[..., None]

        base_anchors_coor_shift = torch.cat(
            [-ws / 2, -hs / 2, ws / 2, hs / 2],
            dim=-1
        )
        base_anchors_coor_shift = base_anchors_coor_shift.view((-1, 4))

        for grid_num in self.grid_nums:
            stride = config.input_width // grid_num
            center_ys = torch.arange(1, (grid_num + 1)) * stride - stride / 2
            center_xs = torch.arange(1, (grid_num + 1)) * stride - stride / 2
            center_ys, center_xs = torch.meshgrid(center_ys, center_xs)
            center_ys = center_ys[..., None]
            center_xs = center_xs[..., None]
            centers = torch.cat(
                [center_xs, center_ys, center_xs, center_ys],
                dim=-1
            )
            centers = centers.view((-1, 4))

            # (64*64, 1, 4) + (1, 9, 4)
            anchors_curr_grid_scale = centers[:, None, :] + base_anchors_coor_shift[None, ...]
            anchors_curr_grid_scale = anchors_curr_grid_scale.view((-1, 4))
            mask1 = anchors_curr_grid_scale[..., 0] >= -5
            mask2 = anchors_curr_grid_scale[..., 1] >= -5
            mask3 = anchors_curr_grid_scale[..., 2] <= config.input_width + 5
            mask4 = anchors_curr_grid_scale[..., 3] <= config.input_height + 5
            # screen cross boundary anchors:
            # mask[i] = 1 if and only if maskj[i] = 1 for j = 1,2,3,4
            mask = mask1 * mask2 * mask3 * mask4
            anchors_curr_grid_scale = anchors_curr_grid_scale[mask]

        AnchorGenerator.anchors = anchors_curr_grid_scale

        return anchors_curr_grid_scale


if __name__ == "__main__":
    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.get_anchors()
    print(anchors.shape)
