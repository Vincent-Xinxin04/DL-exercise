from train import QA_Dataset, evaluate, set_seed
import json
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)
print(f'Using device: {device}')

#加载tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese', cache_dir='../pretrained')

#加载数据
print('Loading test data...')
test_data = json.load(open('../data/hw7_test.json', encoding='utf-8'))
test_questions = test_data['questions']
test_paragraphs = test_data['paragraphs']

test_q_tokens = [tokenizer(q['question_text'], add_special_tokens=False) for q in test_questions]
test_p_tokens = [tokenizer(p, add_special_tokens=False) for p in test_paragraphs]

test_dataset = QA_Dataset('test', test_questions, test_q_tokens, test_p_tokens)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#加载模型
model = BertForQuestionAnswering.from_pretrained('bert-base-chinese', cache_dir='../pretrained')
model_path = '../models/bert_qa_best.pth'
if not os.path.exists(model_path):
    print(f'Model not found: {model_path}')
    print('Please run train.py first to train the model.')
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print('Model loaded successfully!')

#推理
print('Running inference...')
results = []

with torch.no_grad():
    for batch_idx, data in enumerate(tqdm(test_loader)):
        output = model(
            input_ids=data[0].squeeze(dim=0).to(device),
            token_type_ids=data[1].squeeze(dim=0).to(device),
            attention_mask=data[2].squeeze(dim=0).to(device)
        )
        results.append({
            'id': test_questions[batch_idx]['id'],
            'answer': evaluate(data, output)
        })

os.makedirs('../data', exist_ok=True)
with open('../data/hw7_test_result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'Saved {len(results)} predictions to ../data/hw7_test_result.json')
