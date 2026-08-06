import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import mymodel
from train import Covid19Dataset
from train import config
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#读取测试数据（全部特征列）
raw_x_test = pd.read_csv('../data/test.csv').values

test_dataset = Covid19Dataset(raw_x_test,y=None)
test_loder = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,pin_memory=True)

model = mymodel(raw_x_test.shape[1]).to(device)
model.load_state_dict(torch.load('../models/best_model.pth'))

model.eval()
preds = []
for x in tqdm(test_loder):
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
        preds.append(pred.detach().cpu())
preds = torch.cat(preds,dim=0).numpy()

with open('../data/submission.csv','w') as f:
    f.write('id,tested_positive\n')
    for i in range(len(preds)):
        f.write(f'{i},{preds[i]}\n')