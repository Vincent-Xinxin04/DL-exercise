import json
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.utils.data as DataLoader, Dataset
import random
import numpy as np
from transformers import AdamW,BertForQuestionAnswering,BertTokenizerFast
import json


#设置种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class QAdataset(Dataset):
    def __init__(self, questions, paragraphs, tokenizer):
        self.questions = questions
        self.paragraphs = paragraphs
        self.tokenizer = tokenizer

#加载tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese-v1.0')

#设置种子
set_seed(42)

#读取数据
train_dataset = json.load(open('..data/hw7/train.json'))
dev_dataset = json.load(open('..data/hw7/dev.json'))
test_dataset = json.load(open('..data/hw7/test.json'))
train_questions , train_paragraphs = train_dataset['questions'], train_dataset['paragraphs']
dev_questions , dev_paragraphs = dev_dataset['questions'], dev_dataset['paragraphs']
test_questions , test_paragraphs = test_dataset['questions'], test_dataset['paragraphs']

#token化所有数据
train_question_tokens = [tokenizer(question) for question in train_questions['question_text']]
dev_question_tokens = [tokenizer(question) for question in dev_questions['question_text']]
test_question_tokens = [tokenizer(question) for question in test_questions['question_text']]
train_paragraph_tokens = tokenizer(train_paragraphs)
dev_paragraph_tokens = tokenizer(dev_paragraphs)
test_paragraph_tokens = tokenizer(test_paragraphs)
