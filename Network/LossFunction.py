import torch

def temporal_loss(pred_box, prev_box, prev_prev_box, visibility):
    """
    pred_box, prev_box, prev_prev_box: [B,4]
    visibility: [B]
    """
    delta = pred_box - prev_box
    delta_loss = (delta.abs().sum(dim=1) * visibility).mean()

    vel_t = pred_box - prev_box
    vel_tm1 = prev_box - prev_prev_box
    smooth_loss = (vel_t - vel_tm1).abs().sum(dim=1).mean()

    scale_loss = (torch.log(pred_box[:, 2:] / prev_box[:, 2:]).abs().sum(dim=1)).mean()

    return delta_loss + smooth_loss + 0.3 * scale_loss
