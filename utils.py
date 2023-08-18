import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
import random 
import faiss
from tool import *

class ContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.temp = config.temp
        self.batch_size = config.batch_size
        self.maxlen = config.maxlen
        self.eps = 1e-12

# the complete code of ContrastiveLoss will be released after acceptance

class InterestProtos(nn.Module):
    def __init__(self, config):
        super(InterestProtos, self).__init__()
        self.seed = config.random_seed
        self.proto_num = config.proto_num
        self.gpu_id = config.gpu_id
        self.proto_topk = config.proto_topk
        self.sim_type = config.sim_type
        self.temp = config.pcl_temp
        self.item_temp = config.item_temp
        self.bias = config.bias
        self.eps = 1e-7
        self.proto_embs = torch.zeros(size = (self.proto_num, config.hidden_size))

# the complete code of InterestProtos will be released after acceptance 