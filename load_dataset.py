import pandas as pd 
import torch 

class ZINCDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        
    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        inputs = self.tokenizer(smiles, return_tensors='pt', max_length=self.max_length, 
                                padding='max_length', truncation=True)
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()
    

def load_ZINCdataset(tokenizer, path_to_dataset, batch_size, train=True):
  raw_data = pd.read_csv(path_to_dataset)
  raw_data = raw_data['smiles']
  data = []
  for mol in raw_data:
      new_mol = mol.rstrip()
      new_mol = new_mol + '<|endoftext|>'
      data.append(new_mol)
      
  start, end = 0, len(data)
  if train: end = int(len(data) * 0.9)
  else: start = int(len(data) * 0.9)
      
  dataset = ZINCDataset(tokenizer, data[start:end])
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
  return dataset, dataloader 