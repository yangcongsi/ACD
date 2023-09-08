# -*- coding: utf-8 -*-
# @File  : KSCD_train_NCD.py
# @Author: scy
# @Date  : 2022/11/11
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, TestDataLoader
from model_KSCD_NCD import Net
import warnings
from base_ways import *
from utils_1 import CommonArgParser
#
warnings.filterwarnings('ignore')
# can be changed according to command parameter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_n = 50


def model_train():
    data_loader = TrainDataLoader(args.knowledge_n)
    net = Net(args.student_n, args.exer_n, args.knowledge_n, args.low_dim)
    # load_snapshot(net, 'model1/model_epoch13')
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('training model...')
    loss_function = nn.NLLLoss()
    # rmse_min = 1.0
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output+1e-10), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
        acc, rmse, auc = test(net, epoch)
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))

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
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
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
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))
    return accuracy, rmse, auc


if __name__ == '__main__':
    args = CommonArgParser().parse_args()

    # 获取习题、知识点、人数
    exer_n, knowledge_n, student_n = get_data_statistic('math')

    # 模型训练
    model_train()



