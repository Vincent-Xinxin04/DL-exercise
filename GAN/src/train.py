import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os


#设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)