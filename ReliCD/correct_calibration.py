#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zhangyunfei
Date: 2023-01-11 00:18:40
LastEditTime: 2023-01-14 16:15:27
LastEditors: zhangyunfei09@baidu.com
Description: 
FilePath: /zhangyunfei/confidence_work1/32-thirtieth-second/correct_calibration.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from utils import main

params = main()
device = params.device


# correctness history class
class History(object):
    def __init__(self, student_n, exer_n, knowledge_n):
        self.correctness = np.zeros((student_n, knowledge_n))
        self.max_response_num = np.ones((student_n, knowledge_n))

    # correctness update
    def correctness_update(self, input_stu_ids, correct_concept):

        # 第一种写法，如果一个batch里面没有一个学生和同个知识概念有大于一次的交互，则可以这样写
        # non_indx = torch.nonzero(correct_concept)
        # non_indx = non_indx.to(torch.device('cpu')).tolist()
        # non_indx = np.array(non_indx)
        # input_stu_ids = input_stu_ids.to(torch.device('cpu')).tolist()
        # input_stu_ids = np.array(input_stu_ids)
        # index = np.stack((input_stu_ids[non_indx[:,0]], non_indx[:,1]), axis=1)
        # self.correctness[index[:,0],index[:,1]] += 1
        # self.correctness[input_stu_ids[non_indx[:,0]]][non_indx[:,1]] += 1

        # 第二种写法，适用于任何情况，但是速度较慢
        non_indx = torch.nonzero(correct_concept)
        for i in range(len(non_indx)):
            self.correctness[input_stu_ids[non_indx[i][0]]][non_indx[i][1]] += 1
        

    # max num update
    def max_response_num_update(self, input_stu_ids, kn_id):

        non_indx = torch.nonzero(kn_id)
        # 第一种写法，如果一个batch里面没有一个学生和同个知识概念有大于一次的交互，则可以这样写
        # non_indx = non_indx.to(torch.device('cpu')).tolist()
        # non_indx = np.array(non_indx)
        # input_stu_ids = input_stu_ids.to(torch.device('cpu')).tolist()
        # input_stu_ids = np.array(input_stu_ids)
        # index = np.stack((input_stu_ids[non_indx[:,0]], non_indx[:,1]), axis=1)

        # 第二种写法，适用于任何情况，但是速度较慢
        for i in range(len(non_indx)):
            self.max_response_num[input_stu_ids[non_indx[i][0]]][non_indx[i][1]] += 1


    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, cum_correctness, stu_idx, kno_idx):

        data_min = self.correctness.min()
        data_max = float(self.max_response_num.max())


        return (cum_correctness - data_min) / (data_max - data_min)


    def get_target_margin(self, stu_idx1, stu_idx2, kno_idx1, kno_idx2):

        stu_idx1 = stu_idx1.to(torch.device('cpu')).tolist()
        stu_idx2 = stu_idx2.to(torch.device('cpu')).tolist()
        kno_idx1 = kno_idx1.to(torch.device('cpu')).tolist()
        kno_idx2 = kno_idx2.to(torch.device('cpu')).tolist()

        
        # 在这里直接进行归一化了  correct_num / response_num
        cum_correctness1 = np.sum(self.correctness[stu_idx1] * kno_idx1, axis=1) / np.sum(self.max_response_num[stu_idx1] * kno_idx1, axis=1)
        cum_correctness2 = np.sum(self.correctness[stu_idx2] * kno_idx2, axis=1) / np.sum(self.max_response_num[stu_idx2] * kno_idx2, axis=1)

        # cum_correctness1 = np.array(cum_correctness1)
        # cum_correctness2 = np.array(cum_correctness2)
        # cum_correctness1 = self.correctness_normalize(cum_correctness1, stu_idx1, kno_idx1)
        # cum_correctness2 = self.correctness_normalize(cum_correctness2, stu_idx2, kno_idx2)

        # make target pair
        n_pair = len(cum_correctness1)
        target1 = (cum_correctness1[:n_pair])
        target2 = (cum_correctness2[:n_pair])
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.tensor(target).to(device)
        # calc margin
        margin = np.abs(target1 - target2)
        margin = torch.tensor(margin).to(device)

        return target, margin

    
    def out_put(self, ):

        return self.correctness, self.max_response_num

