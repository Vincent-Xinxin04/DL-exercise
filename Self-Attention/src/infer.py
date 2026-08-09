from train import TransformerModel, device
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
import os
from tqdm import tqdm

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=-20)


class Testdataset(Dataset):
    def __init__(self, data_path):
        test_path = Path(data_path)
        self.test_data = json.load(open(test_path / 'testdata.json', 'r'))['utterances']
        self.data_path = data_path

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        feature_path = self.test_data[idx]['feature_path']
        feature = torch.load(os.path.join(self.data_path, feature_path))
        feature = feature[:256]          # 截取前 256 帧，和训练一致
        return feature


print(f'Device: {device}')
print('Loading test data...')
test_set = Testdataset('../data/Dataset')
# 先把所有 feature_path 取出来，顺序和 DataLoader 一致
test_paths = [item['feature_path'] for item in test_set.test_data]
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collate_fn)
print(f'Test samples: {len(test_set)}')

model = TransformerModel().to(device)
model.load_state_dict(torch.load('../models/best_model.pth', map_location=device))
model.eval()

print('Running inference...')
predictions = []
with torch.no_grad():
    for x in tqdm(test_loader):
        x = x.to(device)
        output = model(x)
        _, pred = torch.max(output, dim=1)
        predictions.append(pred)
predictions = torch.cat(predictions).cpu()

with open('../data/submission.csv', 'w') as f:
    f.write('Id,Category\n')
    for i in range(len(predictions)):
        f.write(f'{test_paths[i]},{predictions[i].item()}\n')
