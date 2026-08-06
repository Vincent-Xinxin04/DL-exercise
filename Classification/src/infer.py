import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from train import device, model_path, concat_nframes,preprocess_data,VoiceDataset,VoiceModel,batch_size

test_dataset = preprocess_data(split='test',feat_dir='../data/libriphone/feat',phone_path='../data/libriphone',concat_nframes=3)
test_set = VoiceDataset(test_dataset)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model = VoiceModel(39*concat_nframes, 41).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    test_pred = []
    for x in test_loader:
        x = x.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)  #选取概率最大的那个
        test_pred.append(predicted)
test_pred = torch.cat(test_pred, dim=0)
with open('../data/submission.csv','w') as f:
    f.write('Id,Class\n')
    for i in range(len(test_set)):
        f.write(f'{i},{test_pred[i].item()}\n')
