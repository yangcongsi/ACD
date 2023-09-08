# -*- coding: utf-8 -*-
# @Author: 你相信光吗
# @Datatime: 2022 
# @File: utils.py
import argparse


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
#         self.add_argument('--exer_n', type=int, default=1686,
#                           help='The number for exercise.')
#         self.add_argument('--knowledge_n', type=int, default=61,
#                           help='The number for knowledge concept.')
#         self.add_argument('--student_n', type=int, default=1967,
#                           help='The number for student.')
#         self.add_argument('--epoch_n', type=int, default=10,
#                           help='The epoch number of training')
#         self.add_argument('--lr', type=float, default=0.0002,
#                           help='Learning rate')
#         self.add_argument('--low_dim', type=int, default=20,
#                           help='embedding dim')

        # # junyi
        # self.add_argument('--exer_n', type=int, default=1686,
        #                   help='The number for exercise.')
        # self.add_argument('--knowledge_n', type=int, default=61,
        #                   help='The number for knowledge concept.')
        # self.add_argument('--student_n', type=int, default=1967,
        #                   help='The number for student.')
        # self.add_argument('--epoch_n', type=int, default=10,
        #                   help='The epoch number of training')
        # self.add_argument('--lr', type=float, default=0.001,
        #                   help='Learning rate')
        # self.add_argument('--low_dim', type=int, default=20,
        #                   help='embedding dim')
        # self.add_argument('--dropout', type=int, default=0,
        #                   help='dropout rate')
        # self.add_argument('--net1', type=int, default=61,
        #           help='表示学生在特定知识点掌握的表征维度')

        # # assisst
        # self.add_argument('--exer_n', type=int, default=17751,
        #                   help='The number for exercise.')
        # self.add_argument('--knowledge_n', type=int, default=123,
        #                   help='The number for knowledge concept.')
        # self.add_argument('--student_n', type=int, default=4163,
        #                   help='The number for student.')
        # self.add_argument('--epoch_n', type=int, default=50,
        #                   help='The epoch number of training')
        # self.add_argument('--lr', type=float, default=0.0002,
        #                   help='Learning rate')
        # self.add_argument('--low_dim', type=int, default=40,
        #                   help='embedding dim')
        # self.add_argument('--dropout', type=int, default=0.5,
        #                   help='dropout rate')
        # self.add_argument('--net1', type=int, default=123,
        #           help='表示学生在特定知识点掌握的表征维度')

        # mooper
        # self.add_argument('--exer_n', type=int, default=314,
        #                   help='The number for exercise.')
        # self.add_argument('--knowledge_n', type=int, default=288,
        #                   help='The number for knowledge concept.')
        # self.add_argument('--student_n', type=int, default=5000,
        #                   help='The number for student.')
        # self.add_argument('--epoch_n', type=int, default=50,
        #                   help='The epoch number of training')
        # self.add_argument('--lr', type=float, default=0.0002,
        #                   help='Learning rate')
        # self.add_argument('--low_dim', type=int, default=90,
        #                   help='embedding dim')
        # self.add_argument('--dropout', type=int, default=0.5,
        #                   help='dropout rate')

        # # math
        self.add_argument('--exer_n', type=int, default=20,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=11,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=4209,
                          help='The number for student.')
        self.add_argument('--epoch_n', type=int, default=100,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0002,
                          help='Learning rate')
        self.add_argument('--low_dim', type=int, default=4,
                          help='embedding dim')
        self.add_argument('--dropout', type=int, default=0.5,
                          help='dropout rate')
        #
        #
        # # NIPS
        # self.add_argument('--exer_n', type=int, default=232,
        #                   help='The number for exercise.')
        # self.add_argument('--knowledge_n', type=int, default=71,
        #                   help='The number for knowledge concept.')
        # self.add_argument('--student_n', type=int, default=6918,
        #                   help='The number for student.')
        # self.add_argument('--epoch_n', type=int, default=50,
        #                   help='The epoch number of training')
        # self.add_argument('--lr', type=float, default=0.0002,
        #                   help='Learning rate')
        # self.add_argument('--low_dim', type=int, default=20,
        #                   help='embedding dim')
        # self.add_argument('--dropout', type=int, default=0.5,
        #                   help='dropout rate')
