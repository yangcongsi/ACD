# -*- coding: utf-8 -*-
# @File  : base_ways.py
# @Author: mw
# @Date  : 2021/10/25
# 存储一些模型通用的方法
import torch
import random
import json


# 根据输入的数据集名称，得到该数据集中学生、习题、知识点数
def get_data_statistic(data_type):
    e_num, k_num, s_num = 0, 0, 0
    if data_type == 'bbk':
        e_num = 1686
        k_num = 61
        # k_num = 31
        s_num = 1967
    elif data_type == 'assisst':
        e_num = 17751
        k_num = 123
        s_num = 4163
    elif data_type == 'junyi':
        e_num = 712
        k_num = 39
        s_num = 1000
    elif data_type == 'mooper':
        e_num = 314
        k_num = 288
        s_num = 5000
    elif data_type == 'math':
        e_num = 20
        k_num = 11
        s_num = 4209
    elif data_type == 'NIPS':
        e_num = 232
        k_num = 71
        s_num = 6918
    else:
        print("该数据集不存在")
    return e_num, k_num, s_num


# 加载模型参数文件
def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


# 保存模型参数文件
def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


# 随机选择一份作为验证集，其他四份作为训练集
# 同时记录每个学生训练集中出现的知识点
def get_Train_set(data_name, data_num):
    train_set, val_set = [], []
    stu_kno_dict = {}
    for i in range(1, 6):
        if i != data_num:  # 表明该数据集为训练集数据
            with open(data_name + '/data' + str(i) + '.json', encoding='utf8') as i_f:
                stus = json.load(i_f)
            for stu in stus:
                user_id = stu['user_id']
                temp = []
                for log in stu['logs']:
                    train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                                      'knowledge_code': log['knowledge_code']})
                    temp += log['knowledge_code']
                temp = list(set(temp))
                if user_id in stu_kno_dict:
                    temp1 = stu_kno_dict[user_id] + temp
                    temp1 = list(set(temp1))
                    stu_kno_dict[user_id] = temp1
                else:
                    stu_kno_dict[user_id] = temp
        else:
            with open(data_name + '/data' + str(i) + '.json', encoding='utf8') as i_f:
                stus = json.load(i_f)
            for stu in stus:
                val_set.append(stu)
    random.shuffle(train_set)
    with open(data_name + '/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(data_name + '/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)
    # with open('train_kno.json', 'w', encoding='utf8') as output_file:
    #     json.dump(stu_kno_dict, output_file, indent=4, ensure_ascii=False)
    print("数据集已构成")
    # 根据训练集中的知识点，将测试集分成三类
    with open(data_name + '/test_set_all.json', encoding='utf8') as i_f:
        test_data = json.load(i_f)
    test_sparse, test_normal, test_dense = [], [], []
    for stu in test_data:
        logs = stu['logs']
        log_dense, log_normal, log_sparse = [], [], []
        for log in logs:
            knowledge_test = list(set(log['knowledge_code']))
            knowledge_train = stu_kno_dict[stu['user_id']]
            inter = [j for j in knowledge_test if j not in knowledge_train]     # 该题的新知识点
            s_rate = len(inter) / len(knowledge_test)
            if s_rate == 1:     # 表示该题涉及的知识点全是新知识点(sparse)
                log_sparse.append(log)
            elif 0.4 <= s_rate <= 0.6:
                log_normal.append(log)
            elif s_rate == 0:
                log_dense.append(log)
        if len(log_sparse) > 0:
            test_sparse.append({'user_id': stu['user_id'], 'logs': log_sparse})
        if len(log_normal) > 0:
            test_normal.append({'user_id': stu['user_id'], 'logs': log_normal})
        if len(log_dense) > 0:
            test_dense.append({'user_id': stu['user_id'], 'logs': log_dense})

    with open(data_name + '/test_sparse.json', 'w', encoding='utf8') as output_file:
        json.dump(test_sparse, output_file, indent=4, ensure_ascii=False)
    with open(data_name + '/test_normal.json', 'w', encoding='utf8') as output_file:
        json.dump(test_normal, output_file, indent=4, ensure_ascii=False)
    with open(data_name + '/test_dense.json', 'w', encoding='utf8') as output_file:
        json.dump(test_dense, output_file, indent=4, ensure_ascii=False)

# get_Train_set('Dataset_5_5/assist_set', 3)
# # 统计三类测试集一般的数据量多少
# with open('Dataset_5_5/assist_set/test_sparse.json', encoding='utf8') as i_f:
#     test_s = json.load(i_f)
# with open('Dataset_5_5/assist_set/test_normal.json', encoding='utf8') as i_f:
#     test_n = json.load(i_f)
# with open('Dataset_5_5/assist_set/test_dense.json', encoding='utf8') as i_f:
#     test_d = json.load(i_f)
# num_d, num_n, num_s = 0, 0, 0
# for stu in test_s:
#     num_s += len(stu['logs'])
# for stu in test_n:
#     num_n += len(stu['logs'])
# for stu in test_d:
#     num_d += len(stu['logs'])
# print("稀疏测试题数：", num_s)
# print("普通测试题数：", num_n)
# print("稠密测试题数：", num_d)
