# -*- coding: utf-8 -*-
# @File  : KSCD_train_gan.py
# @Author: scy
# @Date  : 2022/11/5
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from data_loader_gan import TrainDataLoader, TRAINDataLoader, TestDataLoader
from model_KSCD_DACD_gan import Net
import warnings
from base_ways import *
from utils import CommonArgParser
#
warnings.filterwarnings('ignore')
# can be changed according to command parameter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_n = 50


def model_train():
    data_loader = TrainDataLoader(args.knowledge_n)
    data_loader1 = TRAINDataLoader(args.knowledge_n)
    net = Net(args.student_n, args.exer_n, args.knowledge_n, args.low_dim)
    # load_snapshot(net, 'model/model_epoch10')
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('training model...')

    loss_function = nn.NLLLoss()
    loss_domain = nn.CrossEntropyLoss()


    for epoch in range(epoch_n):
        data_loader.reset()
        data_loader1.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            alpha = 1
            t1 = torch.zeros(128).long().to(device)
            t2 = torch.ones(128).long().to(device)


            input_stu_id_s, input_exer_id_s, input_knowledge_emb_s, label_s = data_loader.next_batch()
            input_stu_id_t, input_exer_id_t, input_knowledge_emb_t, label_t = data_loader1.next_batch()

            input_stu_id_s, input_exer_id_s, input_knowledge_emb_s, label_s = input_stu_id_s.to(device), input_exer_id_s.to(device), input_knowledge_emb_s.to(device), label_s.to(device)
            input_stu_id_t, input_exer_id_t, input_knowledge_emb_t, label_t = input_stu_id_t.to(device), input_exer_id_t.to(device), input_knowledge_emb_t.to(device), label_t.to(device)

            optimizer.zero_grad()
            output_1, domain_s_output, domain_t_output, domain_s_output_A, domain_t_output_A = net.forward(input_stu_id_s, input_exer_id_s, input_knowledge_emb_s, input_stu_id_t, input_exer_id_t, input_knowledge_emb_t, alpha)

            err_s_domain1 = loss_domain(domain_s_output, t1)
            err_s_domain2 = loss_domain(domain_s_output_A, t1)
            # err_s_domain3 = loss_domain(domain_s_output_B, t1)
            # err_s_domain4 = loss_domain(domain_s_output_D, t1)
            # err_s_domain5 = loss_domain(domain_s_output_Q, t1)

            err_t_domain1 = loss_domain(domain_t_output, t2)
            err_t_domain2 = loss_domain(domain_t_output_A, t2)
            # err_t_domain3 = loss_domain(domain_t_output_B, t2)
            # err_t_domain4 = loss_domain(domain_t_output_D, t2)
            # err_t_domain5 = loss_domain(domain_t_output_Q, t2)

            # err_s_domain = err_s_domain1 + err_s_domain3 + err_s_domain4
            # err_t_domain = err_t_domain1 + err_t_domain3 + err_t_domain4
            err_diag_domain = err_s_domain1 + err_t_domain1
            # err_embe_domain = err_s_domain3 + err_t_domain3 + err_s_domain4 + err_t_domain4
            err_embe_domain = err_s_domain2 + err_t_domain2
            # err_embe_domain = err_s_domain5 + err_t_domain5
            # err_embe_domain = err_s_domain3 + err_t_domain3 + err_s_domain4 + err_t_domain4 + err_s_domain5 + err_t_domain5

            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            # loss = loss_function(torch.log(output+1e-10), label_s) + 0.5 * err_s_domain + 0.5 * err_t_domain
            loss = loss_function(torch.log(output), label_s) + 0.5 * err_diag_domain + 0.5 * err_embe_domain  # diag和embe的loss权重参数
            # loss = loss_function(torch.log(output), label_s) + 0.5 * err_diag_domain  # diag和embe的loss权重参数

            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
        acc, rmse, auc = test(net, epoch)
        save_snapshot(net, 'model1/model_epoch' + str(epoch + 1))


def test(model, epoch):
    data_loader = TestDataLoader(args.knowledge_n)
    net = Net(args.student_n, args.exer_n, args.knowledge_n, args.low_dim)
    print('testing model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count = 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        alpha = 1

        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output, _, _, _, _ = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, input_stu_ids, input_exer_ids, input_knowledge_embs, alpha)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))

    with open('result/gan.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))
    return accuracy, rmse, auc

if __name__ == '__main__':
    args = CommonArgParser().parse_args()

    # 获取习题、知识点、人数
    exer_n, knowledge_n, student_n = get_data_statistic('assisst')

    # 模型训练
    model_train()



