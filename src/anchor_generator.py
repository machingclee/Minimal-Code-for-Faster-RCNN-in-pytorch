from operator import index
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
            return AnchorGenerator.anchors.to(device)

        ratios = torch.as_tensor(config.anchor_ratios)
        scales = torch.as_tensor(config.anchor_scales)

        hs = scales[None, ...] * torch.sqrt(ratios)[..., None]
        hs = hs.reshape((len(ratios) * len(scales),))

        ws = scales[None, ...] / torch.sqrt(ratios)[..., None]
        ws = ws.reshape((len(ratios) * len(scales),))

        hs = hs[..., None]
        ws = ws[..., None]

        base_anchors_coor_shift = torch.cat(
            [-ws / 2, -hs / 2, ws / 2, hs / 2],
            dim=-1
        )
        base_anchors_coor_shift = base_anchors_coor_shift.reshape((-1, 4))

        for (grid_num_y, grid_num_x) in self.grid_nums:
            stride_y = config.input_height // grid_num_y
            stride_x = config.input_width // grid_num_x
            center_ys = torch.arange(1, (grid_num_y + 1)) * stride_y - stride_y / 2
            center_xs = torch.arange(1, (grid_num_x + 1)) * stride_x - stride_x / 2
            center_ys, center_xs = torch.meshgrid(center_ys, center_xs, indexing="ij")
            center_ys = center_ys[..., None]
            center_xs = center_xs[..., None]
            centers = torch.cat(
                [center_xs, center_ys, center_xs, center_ys],
                dim=-1
            )
            centers = centers.reshape((-1, 4))

            # (64*64, 1, 4) + (1, 9, 4)
            anchors_curr_grid_scale = centers[:, None, :] + base_anchors_coor_shift[None, ...]
            anchors_curr_grid_scale = anchors_curr_grid_scale.reshape((-1, 4))
            # mask1 = anchors_curr_grid_scale[..., 0] >= 0
            # mask2 = anchors_curr_grid_scale[..., 1] >= 0
            # mask3 = anchors_curr_grid_scale[..., 2] <= config.input_width
            # mask4 = anchors_curr_grid_scale[..., 3] <= config.input_height
            # # screen cross boundary anchors:
            # # mask[i] = 1 if and only if maskj[i] = 1 for j = 1,2,3,4
            # mask = mask1 * mask2 * mask3 * mask4
            # internal_index = torch.where(mask == 1)[0]
            # anchors_curr_grid_scale = anchors_curr_grid_scale[internal_index]

        AnchorGenerator.anchors = anchors_curr_grid_scale

        return anchors_curr_grid_scale


if __name__ == "__main__":
    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.get_anchors()
    print(anchors.shape)
