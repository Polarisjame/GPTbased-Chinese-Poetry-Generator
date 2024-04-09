#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dataset.py 
@File    ：model.py
@Author  ：Polaris
@Date    ：2022-05-13 19:35
'''
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem")