import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

def compute_class_weights(dataloader):
    """
    Compute class weights from a dataset.
    """
    class_counts_binary = np.zeros(2, dtype=np.float64)

    # Iterate through dataloader
    for _, _, lab in tqdm(dataloader):
        
        labels_lab = lab#.cpu().numpy().flatten()
        #print(lab)
        for cls in range(2):
            class_counts_binary[cls] += np.sum((labels_lab==cls).cpu().numpy())

    weights_binary = 1.0/(class_counts_binary)
    weights_binary /= weights_binary.sum()
    return torch.tensor(weights_binary, dtype=torch.float32).cuda()
    
