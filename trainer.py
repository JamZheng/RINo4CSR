from re import L
import torch
import os
import numpy as np

from tqdm import tqdm

from tool import *

class Trainer(object):
    def __init__(self) -> None:
        super().__init__()

    def train(self, epoch, writer, model, train_data, config):
        logger = config.logger
        model.train()

        step = 0
        loss_sum = 0

        if not os.path.exists('runs'):
            os.mkdir('runs')


        if epoch > config.warm_up_epoch:
            if epoch == config.warm_up_epoch + 1: 
                logger.info('warming up ends ...')
            logger.info('performing spherical kmeans to update proto...')
            item_embs = F.normalize(model.item_embeddings.weight, dim=-1).detach().cpu().numpy()
            model.protos.run_kmeans_update_proto(item_embs)

        for seqs, poss, negs, lens, aug_seqs, aug_lens in train_data:
            # torch.cuda.empty_cache()
            seqs = trans_to_cuda(torch.LongTensor(seqs))
            poss = trans_to_cuda(torch.LongTensor(poss))
            negs = trans_to_cuda(torch.LongTensor(negs))
            lens = trans_to_cuda(torch.LongTensor(lens))

            model.optimizer.zero_grad()

            loss = 0
            cl_loss_record = 0
            proto_loss_record = 0
            item_loss_record = 0
            step += 1


            # calculate the Next item loss           
            loss += model.cal_loss(seqs, poss, negs, lens)
                    
            # calculate the contrastive loss
            aug_seqs1, aug_seqs2 = trans_to_cuda(torch.LongTensor(aug_seqs[0])), trans_to_cuda(torch.LongTensor(aug_seqs[1]))
            aug_lens1, aug_lens2 = trans_to_cuda(torch.LongTensor(aug_lens[0])), trans_to_cuda(torch.LongTensor(aug_lens[1]))
            
            aug_embs1 = model(aug_seqs1, aug_lens1, phase=config.cl_embs)
            aug_embs2 = model(aug_seqs2, aug_lens2, phase=config.cl_embs)
            batch_size = aug_embs1.shape[0]
            if epoch <= config.warm_up_epoch:
                # warm up with the sequence-level contrastive loss
                cl_loss_record += model.cl_loss(aug_embs1.view(batch_size, -1), aug_embs2.view(batch_size, -1)) * config.w_clloss

            
            elif epoch > config.warm_up_epoch:
                    seq_embs = model(seqs, lens)
                    if config.cl_embs == 'concat':
                        aug_embs1_predict = model.gather_indexes(aug_embs1, lens - 1)
                        aug_embs2_predict = model.gather_indexes(aug_embs2, lens - 1)
                    else:
                        aug_embs1_predict = aug_embs1
                        aug_embs2_predict = aug_embs2
                    
                    
                    # contruct support set
                    support_set = torch.concat([seq_embs.unsqueeze(1), aug_embs1_predict.unsqueeze(1), aug_embs2_predict.unsqueeze(1)], dim=1)
                    # extract user robust interest
                    main_protos_bool, main_protos_embs = model.protos.get_main_protos(support_set)
                    
                    # robust interest guided denoising layer
                    aug_embs1 = model.kmeans_aug_forward(aug_seqs1, aug_lens1, main_protos_embs, config.cl_embs)
                    aug_embs2 = model.kmeans_aug_forward(aug_seqs2, aug_lens2, main_protos_embs, config.cl_embs)
                    
                    if config.cl_embs == 'concat':
                        aug_embs1_predict = model.gather_indexes(aug_embs1, lens - 1)
                        aug_embs2_predict = model.gather_indexes(aug_embs2, lens - 1)
                    else:
                        aug_embs1_predict = aug_embs1
                        aug_embs2_predict = aug_embs2

                    # filter out the noisy seqs before calculating the cl loss
                    noise_seqs_mask = main_protos_bool.sum(-1) > 0 # [B]
                    aug_embs1 = aug_embs1[noise_seqs_mask]
                    aug_embs2 = aug_embs2[noise_seqs_mask]
                    aug_embs1_predict = aug_embs1_predict[noise_seqs_mask]
                    aug_embs2_predict = aug_embs2_predict[noise_seqs_mask]
                    main_protos_bool = main_protos_bool[noise_seqs_mask]

                    

                    batch_size = aug_embs1.shape[0]

                    # sequence-level contrastive loss
                    cl_loss_record += model.cl_loss(aug_embs1.view(batch_size, -1), aug_embs2.view(batch_size, -1)) * config.w_clloss

                    # interest-level contrastive loss
                    if config.pcl_embs == 'mean':
                        proto_loss_record += model.protos.cal_multi_proto_loss(aug_embs1.mean(-2), main_protos_bool) * config.w_pcl_loss / 2
                        proto_loss_record += model.protos.cal_multi_proto_loss(aug_embs2.mean(-2), main_protos_bool) * config.w_pcl_loss / 2
                    else:
                        proto_loss_record += model.protos.cal_multi_proto_loss(aug_embs1_predict, main_protos_bool) * config.w_pcl_loss / 2
                        proto_loss_record += model.protos.cal_multi_proto_loss(aug_embs2_predict, main_protos_bool) * config.w_pcl_loss / 2

                    # item-level contrastive loss
                    anchor_items = poss[torch.arange(0, poss.shape[0]), lens - 1]
                    item_loss_record += model.kmeans_item_scl_loss(anchor_items) * config.w_item_clloss



            loss += cl_loss_record
            loss += proto_loss_record
            loss += item_loss_record
            loss.backward()
            model.optimizer.step()
            loss_sum += loss.item()
            writer.add_scalar("loss", loss.item(), step)
            # writer.add_scalar("cl_loss", cl_loss_record.item(), step)


        logger.info('Epoch(by epoch):{:d}\tloss:{:4f}'\
            .format(epoch, loss_sum / train_data.n_batch / config.test_epoch))


    def eval(self, epoch, model, config, test_data, ks, phase='valid'):
        logger = config.logger

        recall, ndcg = [0] * len(ks), [0] * len(ks)
        num = 0
        model.eval()

        test_data_iter = tqdm(test_data, total=test_data.n_batch)
        with torch.no_grad():
            for seqs, tars, lens in test_data_iter:
                seqs = trans_to_cuda(torch.LongTensor(seqs))
                lens = trans_to_cuda(torch.LongTensor(lens))
                item_scores = model.full_sort_predict(seqs, lens)
                nrecall = int(ks[-1])
                item_scores[:, 0] -= 1e9
                if config.repeat_rec == False:
                    for seq, item_score in zip(seqs, item_scores):
                        item_score[seq] -= 1e9
                _, items = torch.topk(item_scores, nrecall, sorted=True)
                items = trans_to_cpu(items).detach().numpy()

                batch_recall, batch_ndcg = [0] * len(ks), [0] * len(ks)

                for item, tar in zip(items, tars):
                    for k, kk in enumerate(ks):
                        if tar in set(item[:kk]):
                            batch_recall[k] += 1
                            item_idx = {i:idx + 1 for idx, i in enumerate(item[:kk])}
                            batch_ndcg[k] += (1 / np.log2(item_idx[tar] + 1))
                
                recall = [r + br for r, br in zip(recall, batch_recall)]
                ndcg = [n + bn for n, bn in zip(ndcg, batch_ndcg)]
                num += seqs.shape[0] 
                
        if phase == 'valid':
            log_str = 'Valid: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'Recall@{:2d}:\t{:.4f}\t'.format(kk, recall[nbr_k] / num)
            logger.info(log_str)
            log_str = 'Valid: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'NDCG@{:2d}:\t{:.4f}\t'.format(kk, ndcg[nbr_k] / num)
            logger.info(log_str)
        else:
            log_str = 'Test: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'Recall@{:2d}:\t{:.4f}\t'.format(kk, recall[nbr_k] / num)
            logger.info(log_str)
            log_str = 'Test: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'NDCG@{:2d}:\t{:.4f}\t'.format(kk, ndcg[nbr_k] / num)
            logger.info(log_str)

        if ks is None:
            return [recall / num]
        else:
            return [r / num for r in recall]
