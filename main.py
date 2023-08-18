import torch
import os
import argparse
import logging
import sys
from datetime import datetime
from time import time

sys.path.append('..')
from tensorboardX import SummaryWriter
from trainer import Trainer
from model import RINoCSR
from tool import *
from data_iterator import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='Beauty')
    parser.add_argument('--model', type=str, default='RINoCSR')

    parser.add_argument('--sim_type', type=str, default='cos', help='dot | cos')
    parser.add_argument('--cos_tau', type=float, default=0.05)
    parser.add_argument('--warm_up_epoch', type=int, default=1)


    # preprocessing
    parser.add_argument('--pre_seq_aug', type=str, default='none', help='none | pre_seq_epoch | pre_seq_dataset')
    parser.add_argument('--min_pre_aug_len', type=int, default=2, help='the minimum length after prefix augumentation')
    
    # user robust interest modeling

    parser.add_argument('--proto_topk', type=int, default=8)
    parser.add_argument('--proto_num', type=int, default=512)
    parser.add_argument('--proto_type', type=str, default='topk', help='topk')
    parser.add_argument('--aug_type', type=str, default= 'mcr',
                        help='mask | crop | reorder')
    # sequence-level contrastive learning 
    parser.add_argument('--cl_embs', type=str, default='concat', help='concat | last')
    parser.add_argument('--w_clloss', type=float, default=0.01, help='the weight of cl loss')
    parser.add_argument('--temp', type=float, default=0.1)

    # interest-level contrastive learning
    parser.add_argument('--pcl_embs', type=str, default='mean', help = 'mean')
    parser.add_argument('--pcl_temp', type=float, default=0.1)
    parser.add_argument('--w_pcl_loss', type=float, default=0.01)
    parser.add_argument('--bias', type=float, default=0)

    # item-level contrastive learning
    parser.add_argument('--w_item_clloss', type=float, default=0.01, help='the weight of cl loss')
    parser.add_argument('--item_temp', type=float, default=0.1)
    

    parser.add_argument('--repeat_rec', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.25)
    parser.add_argument('--filename', type=str, default='debug', help='post filename')
    parser.add_argument('--random_seed', type=int, default=11)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--inner_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.25)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.25)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--hidden_act', type=str, default='relu')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.000, help='')
    parser.add_argument('--max_iter', type=int, default=100, help='(k)')
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--best_ckpt_path', type=str, default='runs/',
                        help='the direction to save ckpt')
    parser.add_argument('--log_dir', type=str, default='log_debug', help='the direction of log')
    parser.add_argument('--loss_type', type=str, default='BCE', help='CE | BCE')

    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--max_epoch', type=int, default=500)

    parser.add_argument('--is_hard', type=bool, default=True)
    parser.add_argument('--mask_p', type=float, default=0.5)
    parser.add_argument('--crop_p', type=float, default=0.6)
    parser.add_argument('--reorder_p', type=float, default=0.2)
    return parser.parse_args()



def main():
    # initial config and seed
    config = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    SEED = config.random_seed
    setup_seed(SEED)

    config.log_dir += '/' + datetime.now().strftime('%m%d')
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    config.log_dir += '/' + config.loss_type
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    # config.log_dir = os.path.join(config.log_dir, datetime.now().strftime('%m%d'))

    if config.aug_type == 'None':
        filename = '{}_{}_{}_clNone_{}_{}_{}'.format(config.dataset, config.model,config.batch_size,
                                                               config.loss_type,config.sim_type, 
                                                             datetime.fromtimestamp(time()).strftime('%m%d%H%M'))
    else:
        filename = '{}_{}_{}_wcl{}_cl_{}_t{}_wp{}_wi{}_{}_{}_{}_warm{}'.format(config.dataset, config.model, config.batch_size, config.w_clloss,
                                                             config.aug_type, config.temp, config.w_pcl_loss, config.w_item_clloss, config.loss_type, config.sim_type, 
                                                             datetime.fromtimestamp(time()).strftime('%m%d%H%M'),
                                                            config.warm_up_epoch)


    filename += '__' + config.filename
    if os.path.exists('{}/{}.log'.format(config.log_dir, filename)): filename += '_2'


    if config.filename == '':
        fileflag = input("Please input the title of the checkpoint: ")
        filename += fileflag
    config.best_ckpt_path += filename
    if not os.path.exists('runs_tensorboard'): os.mkdir('runs_tensorboard')

    writer = SummaryWriter('runs_tensorboard/{}'.format(filename))

    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists('runs'):
        os.mkdir('runs')

    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        filename='{}/{}.log'.format(config.log_dir, filename),
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    config.logger = logger

    # initial dataset
    train_data = TrainData(config)
    valid_data = TestData(config, is_valid=True)
    test_data = TestData(config, is_valid=False)
    config.n_item, config.n_user = train_data.num_items + 1, train_data.num_users + 1


    logger.info("--------------------Configure Info:------------")
    for arg in vars(config):
        logger.info(f"{arg} : {getattr(config, arg)}")

    # initial model
    if config.model == 'RINoCSR':
        model = trans_to_cuda(RINoCSR(config))
    
    # initial trainer
    trainer = Trainer()

# ------------------train------------------------------
    best_metrics = [0]
    trials = 0
    best_epoch = 0

    for i in range(config.max_epoch):
        epoch = i + 1
        trainer.train(epoch, writer, model, train_data, config)
        
        if epoch % config.test_epoch == 0:
            metrics = trainer.eval(epoch, model, config, valid_data, [20, 50], phase='valid')
            
            
            if metrics[0] > best_metrics[0]:
                best_epoch = epoch
                torch.save(model.state_dict(), config.best_ckpt_path)
                # torch.save(model, config.best_ckpt_path + '_whole')
                best_metrics = metrics
                trials = 0
            else:
                trials += 1
                # early stopping
                if trials > config.patience:
                    break

# ------------------test------------------------------
    model.load_state_dict(torch.load(config.best_ckpt_path))
    logger.info('-------------best valid in epoch {:d}-------------'.format(best_epoch))
    trainer.eval(epoch, model, config, valid_data, ks = [20, 50, 100], phase='valid')
    logger.info('------------test-----------------')
    trainer.eval(epoch, model, config, test_data, ks=[20, 50, 100], phase='test')

if __name__ == "__main__":
    main()
