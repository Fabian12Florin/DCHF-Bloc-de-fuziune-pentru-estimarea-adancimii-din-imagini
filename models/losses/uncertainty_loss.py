import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import platform_manager

@platform_manager.LOSSES.add_module
class UncertaintyLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 target_n,
                 uncertainty_n,
                 scales=[0, 1, 2, 3],
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.target_n = target_n
        self.uncertainty_n = uncertainty_n
        self.scales = scales
        self.device = device

    def forward(self, outputs, side):
        total_loss = 0
        
        for scale in self.scales:
            # Get predictions and uncertainty for current scale
            pred = outputs[self.pred_n.format(side)]
            target = outputs[self.target_n.format(side)]
            uncertainty = outputs['{}_uncertainty_{}_{}'.format(self.uncertainty_n, scale, side)]
            
            # Resize predictions and targets to match uncertainty scale
            if scale > 0:
                pred = F.interpolate(pred, scale_factor=1/2**scale, mode='bilinear', align_corners=False)
                target = F.interpolate(target, scale_factor=1/2**scale, mode='bilinear', align_corners=False)
            
            # Compute uncertainty loss (original ProDepth implementation)
            error = torch.abs(pred - target)
            uncertainty_loss = error / (uncertainty + 1e-6) + torch.log(uncertainty + 1e-6)
            uncertainty_loss = uncertainty_loss.mean()
            
            total_loss += uncertainty_loss
            
        return total_loss / len(self.scales) 