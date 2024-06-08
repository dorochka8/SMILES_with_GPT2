import datetime
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from config import CONFIG
from generating_molecules import generate_molecules
from load_dataset import load_ZINCdataset
from num_params import num_params_with_description
from train import trainer 

model_name = 'gpt2'
model_fine_tuning = GPT2LMHeadModel.from_pretrained(model_name).to(CONFIG['device'])
print(model_fine_tuning)
num_params_with_description(model_fine_tuning)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]', 
                            })
model_fine_tuning.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(model_fine_tuning.parameters(), lr=CONFIG['optimizer_lr'])

path_to_dataset = '/Users/dorochka/Desktop/250k_rndm_zinc_drugs_clean_3.csv'
traindata, trainloader = load_ZINCdataset(tokenizer, path_to_dataset, batch_size=CONFIG['batch_size'], train=True)
testdata,  testloader  = load_ZINCdataset(tokenizer, path_to_dataset, batch_size=CONFIG['batch_size'], train=False)

start = datetime.datetime.now()
total_loss = trainer(model_fine_tuning, 
        optimizer, 
        trainloader,
        testloader,
        num_epochs=CONFIG['epochs_train'], 
        device=CONFIG['device']
      )
end = datetime.datetime.now()
print(f'Training proccess took: {end - start}')


lossess = []
for i in total_loss:
    for j in i:
        lossess.append(j)

plt.plot(losses)
plt.show()
