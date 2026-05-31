"""
Gradient (Sobolev) loss term for flow training.

Adds a derivative-domain penalty ``||grad(pred) - grad(gt)||`` on top of the
usual end-point loss, so the network is optimised for accurate *spatial
derivatives* (vorticity / strain), not just velocity. This up-weights
high-frequency error -- the dominant source of noise when differentiating a
dense flow field -- while staying ground-truth-anchored, so true sharp
structures are preserved rather than smoothed away (unlike a post-hoc Gaussian
filter). No incompressibility assumption, so it is valid on 3D-projected data.

Rationale and validation plan: see the dpivsoft CLAUDE.md "Design-direction
exploration" notes.
"""
import torch
import torch.nn.functional as F


def flow_gradients(flow):
    """Central-difference spatial gradients of a flow (or any C-channel) field.

    flow: [B, C, H, W]  ->  (d/dx, d/dy), each [B, C, H, W] (replicate-padded
    so the output keeps full resolution). Pixel spacing is 1; any physical
    scale folds into the loss weight.
    """
    b, c, h, w = flow.shape
    f = flow.reshape(b * c, 1, h, w)
    kx = torch.tensor([-0.5, 0.0, 0.5], dtype=flow.dtype, device=flow.device).view(1, 1, 1, 3)
    ky = kx.view(1, 1, 3, 1)
    fx = F.conv2d(F.pad(f, (1, 1, 0, 0), mode="replicate"), kx).reshape(b, c, h, w)
    fy = F.conv2d(F.pad(f, (0, 0, 1, 1), mode="replicate"), ky).reshape(b, c, h, w)
    return fx, fy


def erode_mask(mask, border=0):
    """Shrink a validity mask for the gradient loss.

    Removes (a) any pixel with an invalid neighbour (1 px morphological erosion,
    since a finite-difference stencil touching an invalid pixel is meaningless),
    and (b) a ``border``-px frame around the image edge. At the edge, CNN/warp
    padding and out-of-frame particle loss make both the prediction and the
    GT-gradient target unreliable, so imposing a derivative target there is
    ill-posed and produces the characteristic border "disaster" -- excluding a
    margin removes it. ``mask`` may be bool or float; returns float in {0, 1}.
    """
    m = -F.max_pool2d(-mask.float(), kernel_size=3, stride=1, padding=1)
    if border > 0:
        m[..., :border, :] = 0.0
        m[..., -border:, :] = 0.0
        m[..., :, :border] = 0.0
        m[..., :, -border:] = 0.0
    return m


def flow_gradient_loss(pred, gt_grads, mask, mode="jacobian"):
    """L1 derivative-domain loss between a predicted flow and the ground truth.

    pred:     [B, 2, H, W] predicted flow at full resolution (channel 0 = u,
              channel 1 = v).
    gt_grads: (gfx, gfy) ground-truth gradients from ``flow_gradients(gt)``,
              precomputed once per batch and reused across iterations/levels.
    mask:     [B, 1, H, W] eroded validity mask (bool or float).
    mode:     'jacobian' -> all four components (du/dx, du/dy, dv/dx, dv/dy);
              'vorticity' -> only dv/dx - du/dy (leaves strain unconstrained).
    """
    gfx, gfy = gt_grads
    pfx, pfy = flow_gradients(pred)
    if mode == "vorticity":
        diff = ((pfx[:, 1:2] - pfy[:, 0:1]) - (gfx[:, 1:2] - gfy[:, 0:1])).abs()
    else:
        diff = ((pfx - gfx).abs() + (pfy - gfy).abs()).sum(dim=1, keepdim=True)
    return (mask.float() * diff).mean()
