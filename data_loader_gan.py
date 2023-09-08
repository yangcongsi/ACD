# encoding: utf-8
"""
  @author: scy
  @file: data_loader_gan.py
  数据加载
  @time: 2022/11/10
"""
import json
import torch
import random


class TrainDataLoader(object):
    """
    assist_set loader for training
    """
    def __init__(self, kn_num):
        self.batch_size = 128
        self.ptr = 0
        self.data = []
        self.knowledge_dim = int(kn_num)

        data_file = 'data/ASSISTment09/stu/40%/dataA_train.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        random.shuffle(self.data)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

class TRAINDataLoader(object):
    """
    assist_set loader for training
    """
    def __init__(self, kn_num):
        self.batch_size = 128
        self.ptr = 0
        self.data = []
        self.knowledge_dim = int(kn_num)

        data_file = 'data/ASSISTment09/stu/40%/dataB_train.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        random.shuffle(self.data)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class TestDataLoader(object):
    def __init__(self, kn_num):
        self.ptr = 0
        self.data = []
        self.knowledge_dim = int(kn_num)

        data_file = 'data/ASSISTment09/stu/40%/dataB_test.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        random.shuffle(self.data)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
