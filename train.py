#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dataset.py 
@File    ：train.py
@Author  ：Polaris
@Date    ：2022-05-13 20:42
'''

import torch
from torch.optim import Adam
import numpy as np


def train(gpt_model, trainloader, testloader, device, learning_rate=0.01, num_epoch=10):
    optimizer = Adam(gpt_model.parameters(), learning_rate)  # 使用Adam优化器
    epoch = 1
    gpt_model.to(device)
    gpt_model.train()
    for epoch in range(1, num_epoch + 1):
        # 记录当前epoch的总loss
        total_loss = 0
        # tqdm用以观察训练进度，在console中会打印出进度条

        for step, batch in enumerate(trainloader):
            inputs = batch.to(device)
            optimizer.zero_grad()
            # 清除现有的梯度
            gpt_output = gpt_model(**inputs, labels=inputs['input_ids'])
            loss = gpt_output.loss
            loss.backward()
            optimizer.step()
            # 统计总的损失
            total_loss += loss.item()
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss.item()), end="")
        with torch.no_grad():
            testloss = []
            for step, batch in enumerate(testloader):
                inputs = batch.to(device)
                gpt_output = gpt_model(**inputs, labels=inputs['input_ids'])
                loss = gpt_output.loss
                testloss.append(loss.item())
                rate = (step + 1) / len(testloader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtest loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss.item()), end="")
            print("Test loss: {:.3f}".format(np.mean(testloss)))

