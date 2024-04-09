#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dataset.py
@File    ：main.py
@Author  ：Polaris
@Date    ：2022-05-13 20:16
'''

from dataload import loadset
from train import train
import torch
import random
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline


def predict(text,text_generator):
    """

    @param text: 对联上联 str or [str,str,...]
    @param text_generator: TextGenerationPipeline
    @return: 对联下联 [[{}],[{}]...]
    """
    outs = []
    for input in text:
        out = text_generator(input, max_length=len(input), do_sample=True)
        out = out[0]['generated_text']
        outs.append(out)
    return outs

def main():
    # Train Model
    trainloader, testloader = loadset(filename, batchsize)
    train(model, trainloader, testloader, device, learning_rate, num_epoch)

    # Predict
    # model = torch.load('./gpt2_model_4.pth', map_location=torch.device('cpu'))
    # text_generator = TextGenerationPipeline(model, tokenizer)
    # texts = []
    # inputs = []
    # res = []
    # length = 1000
    # with open(r'.\couplet\test\in.txt', encoding='utf-8') as f:
    #     for ind, i in enumerate(f):
    #         rands = random.randint(0, 1000)
    #         if rands >= 970:
    #             texts.append(i[0:-1])
    #             inputs.append('[CLS]' + i[0:-1] + '。')
    #         if ind == length:
    #             break
    # res = predict(inputs, text_generator)
    #
    # for i in range(len(texts)):
    #     print(texts[i], end=':->    ')
    #     out = res[i]
    #     print(out[len(texts[i]) + 6:len(out)])


if __name__ == 'main':
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem")

    batchsize = 16
    filename=r'.\couplet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    num_epoch = 5
    main()
