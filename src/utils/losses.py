import torch
import torch.nn as nn

def total_variation_loss(y_true, y_pred):
    # Implementation of total variation loss to smooth the prediction map
    diff_i = torch.mean(torch.abs(y_pred[:, 1:, :] - y_pred[:, :-1, :]))
    diff_j = torch.mean(torch.abs(y_pred[:, :, 1:] - y_pred[:, :, :-1]))
    return diff_i + diff_j


class CustomBCELoss(nn.Module):
    def __init__(self, alpha=10.0, beta=1.0):
        super(CustomBCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss(reduction='none')  # Binary Cross-Entropy without reduction
        
    def forward(self, predictions, probabilities):
        # Binary Cross Entropy
        bce = self.bce_loss(predictions, probabilities)
        
        # Custom weighting
        weights = self.alpha * (1 - probabilities) + self.beta * probabilities
        
        # Apply weights to BCE loss
        weighted_bce = weights * bce
        return weighted_bce.mean()