import json
import os
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from torch.optim import AdamW
from transformers import BertForQuestionAnswering, BertTokenizerFast


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


class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_window_len = 150
        self.doc_stride = 150

        self.max_input_len = 1 + self.max_question_len + 1 + self.max_window_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question['paragraph_id']]

        if self.split == 'train':
            answer_start_token = tokenized_paragraph.char_to_token(question['answer_start'])
            answer_end_token = tokenized_paragraph.char_to_token(question['answer_end'])

            mid = (answer_start_token + answer_end_token) // 2
            window_start = max(0, min(mid - self.max_window_len // 2,
                                       len(tokenized_paragraph['input_ids']) - self.max_window_len))
            window_end = window_start + self.max_window_len

            input_ids_question = [101] + tokenized_question['input_ids'][:self.max_question_len] + [102]
            input_ids_window = tokenized_paragraph['input_ids'][window_start:window_end] + [102]

            answer_token_start = answer_start_token + len(input_ids_question) - window_start
            answer_token_end = answer_end_token + len(input_ids_question) - window_start

            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_window)
            return (torch.tensor(input_ids), torch.tensor(token_type_ids),
                    torch.tensor(attention_mask), answer_token_start, answer_token_end)

        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []

            for i in range(0, len(tokenized_paragraph['input_ids']), self.doc_stride):
                input_ids_question = [101] + tokenized_question['input_ids'][:self.max_question_len] + [102]
                input_ids_window = tokenized_paragraph['input_ids'][i:i + self.max_window_len] + [102]

                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_window)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return (torch.tensor(input_ids_list), torch.tensor(token_type_ids_list),
                    torch.tensor(attention_mask_list))

    def padding(self, input_ids_question, input_ids_window):
        padding_len = self.max_input_len - len(input_ids_question) - len(input_ids_window)
        input_ids = input_ids_question + input_ids_window + [0] * padding_len
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_window) + [0] * padding_len
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_window)) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask

#设置超参数
config = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'num_epochs': 5,
    'patience': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


#加载tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese', cache_dir='../pretrained')

#设置种子
set_seed(42)


def evaluate(data, output):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        prob = start_prob + end_prob
        if prob > max_prob:
            max_prob = prob
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])
    return answer.replace(' ', '')


def collate_train(batch):
    input_ids = torch.stack([x[0] for x in batch]).long()
    token_type_ids = torch.stack([x[1] for x in batch]).long()
    attention_mask = torch.stack([x[2] for x in batch]).long()
    start = torch.tensor([x[3] for x in batch]).long()
    end = torch.tensor([x[4] for x in batch]).long()
    return input_ids, token_type_ids, attention_mask, start, end


if __name__ == '__main__':
    #读取数据
    train_data = json.load(open('../data/hw7_train.json', encoding='utf-8'))
    dev_data = json.load(open('../data/hw7_dev.json', encoding='utf-8'))
    test_data = json.load(open('../data/hw7_test.json', encoding='utf-8'))
    train_questions, train_paragraphs = train_data['questions'], train_data['paragraphs']
    dev_questions, dev_paragraphs = dev_data['questions'], dev_data['paragraphs']
    test_questions, test_paragraphs = test_data['questions'], test_data['paragraphs']

    #逐条 tokenize（返回 Encoding，有 .ids / .char_to_token）
    train_question_tokens = [tokenizer(q['question_text'], add_special_tokens=False)
                             for q in train_questions]
    dev_question_tokens = [tokenizer(q['question_text'], add_special_tokens=False)
                           for q in dev_questions]
    test_question_tokens = [tokenizer(q['question_text'], add_special_tokens=False)
                            for q in test_questions]
    train_paragraph_tokens = [tokenizer(p, add_special_tokens=False) for p in train_paragraphs]
    dev_paragraph_tokens = [tokenizer(p, add_special_tokens=False) for p in dev_paragraphs]
    test_paragraph_tokens = [tokenizer(p, add_special_tokens=False) for p in test_paragraphs]

    train_dataset = QA_Dataset('train', train_questions, train_question_tokens, train_paragraph_tokens)
    dev_dataset = QA_Dataset('dev', dev_questions, dev_question_tokens, dev_paragraph_tokens)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                               collate_fn=collate_train)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    device = config['device']
    model = BertForQuestionAnswering.from_pretrained('bert-base-chinese', cache_dir='../pretrained').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda')

    best_em = 0.0
    early_stop = 0

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]'):
            input_ids, token_type_ids, attention_mask, start, end = [b.to(device) for b in batch]
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                start_positions=start, end_positions=end)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        mean_train_loss = train_loss / len(train_loader)

        model.eval()
        dev_acc = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(dev_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Valid]')):
                output = model(
                    input_ids=data[0].squeeze(dim=0).to(device),
                    token_type_ids=data[1].squeeze(dim=0).to(device),
                    attention_mask=data[2].squeeze(dim=0).to(device)
                )
                dev_acc += evaluate(data, output) == dev_questions[i]['answer_text']

        mean_em = dev_acc / len(dev_loader)
        print(f'Epoch {epoch+1}/{config["num_epochs"]} | Train Loss: {mean_train_loss:.4f} | Valid EM: {mean_em:.4f}')

        if mean_em > best_em:
            best_em = mean_em
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), '../models/bert_qa_best.pth')
            print(f'  -> Best model saved (EM: {best_em:.4f})')
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= config['patience']:
                print('Early stopping!')
                break

    print(f'Training completed! Best EM: {best_em:.4f}')

