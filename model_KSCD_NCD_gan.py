# -*- coding: utf-8 -*-
# @File  : model_KSCD_NCD_gan.py
# @Author: scy
# @Date  : 2022/11/10
import torch
import torch.nn as nn
from functions import ReverseLayerF
from torch.nn import functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, low_dim):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.lowdim = low_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.lowdim)  # 学生的低维表示
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.lowdim)  # 知识点矩阵的低维表示
        self.k_difficulty = nn.Embedding(self.exer_n, self.lowdim)  # 习题的低维表示

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self.layer1 = nn.Linear(self.lowdim, 1)

        self.dc1 = nn.Linear(256, 10)
        self.dc2 = nn.Linear(10, 2)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def domain_classifier(self, h):
        h = self.dc1(h)
        # h = self.dc_drop(h)
        h = F.relu(h)
        h = self.dc2(h)
        return h

    def forward(self, stu_id_s, exer_id_s, kn_emb_s, stu_id_t, exer_id_t, kn_emb_t, alpha):
        """
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        """
        # before prednet
        # 学生能力表示
        stu_low_emb_s = self.student_emb(stu_id_s)
        stu_low_emb_t = self.student_emb(stu_id_t)
        knowledge_low_emb = self.knowledge_emb(torch.arange(self.knowledge_dim).to(device))
        stu_emb_s = torch.sigmoid(torch.mm(stu_low_emb_s, knowledge_low_emb.T))  # 得到表示学生能力
        stu_emb_t = torch.sigmoid(torch.mm(stu_low_emb_t, knowledge_low_emb.T))  # 得到表示学生能力


        # 习题难度表示
        exe_low_emb_s = self.k_difficulty(exer_id_s)
        exe_low_emb_t = self.k_difficulty(exer_id_t)
        k_difficulty_s = torch.sigmoid(torch.mm(exe_low_emb_s, knowledge_low_emb.T))  # 得到表示学生能力
        k_difficulty_t = torch.sigmoid(torch.mm(exe_low_emb_t, knowledge_low_emb.T))  # 得到表示学生能力
        e_discrimination_s = torch.sigmoid(self.layer1(exe_low_emb_s)) * 10
        e_discrimination_t = torch.sigmoid(self.layer1(exe_low_emb_t)) * 10

        # prednet
        input_x_s = e_discrimination_s * (stu_emb_s - k_difficulty_s) * kn_emb_s
        input_x_s = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_s)))
        input_x_s = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_s)))

        reverse_s = ReverseLayerF.apply(input_x_s, alpha)
        domain_s_output = self.domain_classifier(reverse_s)

        input_x_t = e_discrimination_t * (stu_emb_t - k_difficulty_t) * kn_emb_t
        input_x_t = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_t)))
        input_x_t = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_t)))

        reverse_t = ReverseLayerF.apply(input_x_t, alpha)
        domain_t_output = self.domain_classifier(reverse_t)

        output = torch.sigmoid(self.prednet_full3(input_x_s))
        return output, domain_s_output, domain_t_output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stat_idx):  # stat_emb是一个1×k的维度，(k个知识点)，表示对知识点的熟练程度
        stu_low_emb = self.student_emb(stat_idx)
        knowledge_low_emb = self.knowledge_emb(torch.arange(self.knowledge_dim).to(device))
        stu_emb = torch.sigmoid(torch.mm(stu_low_emb, knowledge_low_emb.T))  # 得到表示学生能力
        return stu_emb.data

    def get_knowledge_embed(self):
        knowledge_low_emb = self.knowledge_emb(torch.arange(self.knowledge_dim).to(device))
        return knowledge_low_emb.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
