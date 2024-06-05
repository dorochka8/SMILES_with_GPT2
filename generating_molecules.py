import torch


def generate_molecules(model, 
                       tokenizer, 
                       config, 
                       temp, 
                       prompts, 
                       ):

    model.eval()
    model.to(config['device'])
    smiels_collection = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer.encode(prompt, max_length=256, return_tensors='pt').to(config['device'])
            outputs = model.generate(inputs,
                                     max_length=256, 
                                     do_sample=True, 
                                     temperature=temp, 
                                     early_stopping=True, 
                                     pad_token_id=tokenizer.pad_token_id,
                                     num_return_sequences=32,
                                     eos_token_id=tokenizer.eos_token_id,
                                      )
            smiles = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            smiels_collection.append(smiles)
    return smiels_collection