from train import FoodModel,FoodDataset,test_tfm,batch_size,device
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

print(f'Device: {device}')
print('Loading test data...')
test_set = FoodDataset('../data/food11/test', transform=test_tfm,mode='test')
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print(f'Test samples: {len(test_set)}')

model = FoodModel()
model.load_state_dict(torch.load('../models/best_model.pth', map_location=device))
model.to(device)
model.eval()
print('Running inference...')
predictions = []

with torch.no_grad():
    for x in tqdm(test_loader):
        x = x.to(device)
        output = model(x)
        _, pred = torch.max(output, 1)
        predictions.append(pred)
    predictions = torch.cat(predictions, dim=0)
with open('../data/submission.csv', 'w') as f:
    f.write('id,label\n')
    f.write('\n'.join([f'{i},{pred.item()}' for i, pred in enumerate(predictions)]))
print(f'Done! Saved {len(predictions)} predictions to ../data/submission.csv')
