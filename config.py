import torch
CONFIG = {
    'epochs_train' : 5,
    'optimizer_lr' : 3e-4, 
    'batch_size'   : 64, 
    
    'device'       : 'cuda' if torch.cuda.is_available() else 'cpu', 
}