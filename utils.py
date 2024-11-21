import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
import random 
import faiss
from tool import *

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


    def run_kmeans_update_proto(self, x):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(self.proto_num)
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = self.seed
        clus.max_points_per_centroid = 10000
        clus.min_points_per_centroid = 5

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0 

        # spherical K-means clustering
        clus.spherical = True
        index = faiss.GpuIndexFlatIP(res, d, cfg)
        clus.train(x, index)   

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        # convert to cuda Tensors for broadcast
        centroids = trans_to_cuda(torch.Tensor(centroids))
        centroids = F.normalize(centroids, p=2, dim=-1)  
        
        self.proto_embs = centroids
        
        _, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        id2cluster = [int(n[0]) for n in I]
        self.id2cluster = trans_to_cuda(torch.LongTensor(id2cluster))

        return centroids


    def get_main_protos(self, support_sets):
        # support set [B, h, emb]
        # protos [num, emb]
        support_sets = support_sets.detach()
        support_sets_sim = torch.matmul(support_sets, self.proto_embs.T) #[B, h, num]

        self.ave_seq_sim = support_sets_sim.mean(dim=1).detach() #[B, num]

        _ , main_protos_ids = torch.topk(support_sets_sim, k=self.proto_topk, dim=-1) #[B, h, k]
        main_protos_bool = torch.zeros_like(support_sets_sim, dtype=torch.bool)  # [B, h, num]
        main_protos_bool.scatter_(-1, main_protos_ids, True)  # [B, h, num]
        main_protos_bool = torch.all(main_protos_bool, dim=-2)  # [B, num]
        # 235 20
        main_protos_sim = support_sets_sim.mean(dim=1) #[B, num]
        main_protos_sim = main_protos_sim * main_protos_bool - (main_protos_bool == False) * 1e7
        main_protos_dist = F.softmax(main_protos_sim, dim=-1)

        main_protos_embs = torch.matmul(main_protos_dist, self.proto_embs.detach()) #[B, emb]
        # main_protos_embs = (main_protos_dist.unsqueeze(-1) * self.protos.unsqueeze(0)).sum(1) #[B, emb]
        main_protos_embs = F.normalize(main_protos_embs, p=2, dim=-1) # [B, emb]

        return main_protos_bool, main_protos_embs

    def cal_multi_proto_loss(self, pos_embeds, main_protos_bool, tau = 2.0):
        batch_size = pos_embeds.shape[0]
        main_protos_cnt = main_protos_bool.sum(dim=0, keepdim=True) #[1, proto_num]
        
        # dynamic temperature
        main_protos_tau = 1 / torch.exp(tau * main_protos_cnt / batch_size) #[1, proto_num]
        

        proto_mask_pos = main_protos_bool

        pcl_temp = self.proto_embs.clone().detach()

        score = torch.matmul(pos_embeds, pcl_temp.T) / (self.temp * main_protos_tau) #[batch_size, proto_num]

        loss = self.cal_sup_loss(proto_mask_pos, score)

        return loss


    def cal_item_cluster_scl(self, item_ids, item_embs, all_item_embs):
        item_clusters = self.id2cluster[item_ids] # [B]


        item_mask = item_clusters.unsqueeze(-1) == self.id2cluster.unsqueeze(0) # [B, items]
        mask_self = torch.ones_like(item_mask, dtype=torch.bool, device=item_mask.device)
        mask_self.scatter_(-1, item_ids.unsqueeze(-1), False)

        logits = torch.matmul(item_embs, all_item_embs.T) / self.item_temp # [B, items]

        logits = logits[mask_self].view(item_embs.shape[0], all_item_embs.shape[0] - 1)
        item_mask = item_mask[mask_self].view(item_embs.shape[0], all_item_embs.shape[0] - 1)
        loss = self.cal_sup_loss(item_mask, logits)

        return loss

    def cal_sup_loss(self, mask_pos, score):
        # for stabilty
        max_norm, _ = score.max(dim=1, keepdim=True)
        max_norm = max_norm.expand_as(score)

        score_norm = score - max_norm
        # [batch_size, hidden]
        exp_score = torch.exp(score_norm)
        exp_logit = torch.log(exp_score + self.eps) - torch.log(exp_score.sum(dim=-1, keepdim=True) + self.eps)
        
        loss = -(exp_logit * mask_pos).sum(dim=-1) / (mask_pos.sum(dim=-1) + self.eps)

        return loss.mean()



    def kmeans_gumbel_softmax_drop(self, ori_items_embs, main_protos_embs, padding_embs, tau = 0.2):
        # ori_items_embs [B, L, embs]
        # seqs_embs [B, embs]
        # main_protos [B, k, embs]
        norm_items_embs = F.normalize(ori_items_embs, dim=-1)
        main_protos_embs = F.normalize(main_protos_embs, dim=-1)
        item_intent_sim = (main_protos_embs.unsqueeze(1) * norm_items_embs).sum(-1).unsqueeze(-1) #B, L
        ave_item_intent_sim = (norm_items_embs.matmul(self.proto_embs.T)).mean(dim=-1).unsqueeze(-1)
        mask = ave_item_intent_sim - item_intent_sim + self.bias
        
        mask = torch.sigmoid(mask / tau)
        retain = 1 - mask
        drop_prob = torch.concat((mask, retain), dim=-1)

        y = torch.rand_like(drop_prob)
        g = self.inverse_gumbel_cdf(y, mu=0, beta=1)

        drop = torch.softmax((torch.log(drop_prob) + g) / 0.01, dim=-1)
        drop_mask = drop[:, :, 0] < 0.8
        dropped_items_embs = ori_items_embs * drop_mask.unsqueeze(-1) + padding_embs * (drop_mask.unsqueeze(-1)  == False)
        
        
        return dropped_items_embs
    
    def inverse_gumbel_cdf(self, y, mu, beta):
        return mu - beta * torch.log(-torch.log(y))


class ContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.temp = config.temp
        self.batch_size = config.batch_size
        self.maxlen = config.maxlen
        self.eps = 1e-12

    def forward(self, pos_embeds1, pos_embeds2):
        batch_size = pos_embeds1.shape[0]
        pos_diff = torch.sum(pos_embeds1 * pos_embeds2, dim=-1).view(-1, 1) / self.temp
        score11 = torch.matmul(pos_embeds1, pos_embeds1.transpose(0, 1)) / self.temp
        score22 = torch.matmul(pos_embeds2, pos_embeds2.transpose(0, 1)) / self.temp
        score12 = torch.matmul(pos_embeds1, pos_embeds2.transpose(0, 1)) / self.temp

        mask = (-torch.eye(batch_size).long() + 1).bool()
        mask = trans_to_cuda(mask)
        score11 = score11[mask].view(batch_size, -1)
        score22 = score22[mask].view(batch_size, -1)
        score12 = score12[mask].view(batch_size, -1)


        score1 = torch.cat((pos_diff, score11, score12), dim=1) # [B, 2B - 2]
        score2 = torch.cat((pos_diff, score22, score12), dim=1)
        score = torch.cat((score1, score2), dim=0)

        labels = torch.zeros(batch_size * 2).long()
        score = trans_to_cuda(score)
        labels = trans_to_cuda(labels)
        assert score.shape[-1] == 2 * batch_size - 1
        return self.ce_loss(score, labels)
    
    def cal_sup_loss(self, mask_pos, score):
        # score = torch.matmul(pos_embeds1, self.proto.T) / self.temp #[batch_size, proto_num]
        # for stabilty
        max_norm, _ = score.max(dim=1, keepdim=True)
        max_norm = max_norm.expand_as(score)

        score_norm = score - max_norm
        # [batch_size, hidden]
        exp_score = torch.exp(score_norm)
        exp_logit = torch.log(exp_score + self.eps) - torch.log(exp_score.sum(dim=-1, keepdim=True) + self.eps)
        
        loss = -(exp_logit * mask_pos).sum(dim=-1) / (mask_pos.sum(dim=-1) + self.eps)

        return loss.mean()

