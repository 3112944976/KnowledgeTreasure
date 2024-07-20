'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''
import numpy as np
import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time


if __name__ == '__main__':
    # 参数设置
    args.device = torch.device('cuda:' + str(args.gpu_id))
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    # 1. 加载熟数据
    # plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    n_users = data_generator.n_users
    n_items = data_generator.n_items
    n_train = data_generator.n_train
    n_test = data_generator.n_test
    print('总用户数n_users=%d, 总物品数n_items=%d' % (n_users, n_items))
    print('总交互数n_interactions=%d' % (n_train + n_test))
    print('训练交互数n_train=%d, 测试交互数n_test=%d, 数据稀疏度sparsity=%.5f' % (n_train, n_test, (n_train + n_test) / (n_users * n_items)))

    norm_adj = data_generator.get_adj_mat()
    # 2. 模型实例化
    model = NGCF(data_generator.n_users, data_generator.n_items, norm_adj, args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 3. 模型训练与测试
    t0 = time()  # 记录开始时间
    cur_best_pre_0, stopping_step = 0, 0  # 初始化最佳预测准确率和停止步数
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        for idx in range(n_batch):
            # 采样批次用户和正负样本
            users, pos_items, neg_items = data_generator.sample()
            # 获取批次(u,i,j)所对应的嵌入表征
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, pos_items, neg_items, drop_flag=args.node_dropout_flag)
            # 计算bpr损失batch_loss=batch_mf_loss+batch_emb_loss
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
        # 在每10次周期之外，仅需打印训练信息并直接进入下一个训练周期即可，无需后续测试。
        if (epoch + 1) % 1 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                print('Epoch %d [%.1fs]: train_loss==%.4f' % (epoch, time() - t1, loss))
            continue
        # 1. 测试评估
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)
        t3 = time()
        # 2. 记录当前周期的损失和评估指标记录ret
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        # 3. 在每10次周期之时，打印训练和测试信息
        if args.verbose > 0:
            # 测试信息中对于每个指标，返回其top-n=Ks[0]和top-n=Ks[-1]下的值。
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train_loss==%.4f, recall=[%.4f, %.4f], ' \
                       'precision=[%.4f, %.4f], hit=[%.4f, %.4f], ndcg=[%.4f, %.4f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
        # 4. 根据当前epoch的recall和之前的最佳召回率，来决定是否提前停止训练。若累计flag_step次测试的recall[0]值都没超过历史最佳，则早停。
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=5)
        if should_stop == True:
            break
        # 5. 若当前周期top-n=Ks[0]下的recall测试度量值为历史最佳，且save_flag==1时，则保存当前模型的参数到指定路径，并打印保存路径信息。
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
    # 将记录下来的评估指标转换为NumPy数组，以便后续分析
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    # 获取最佳召回率及其对应的epoch索引
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)
    # 6. 打印最佳性能指标结果，包括最佳epoch的索引
    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)