import matplotlib.pyplot as plt 
import torch

def trainer(model, optimizer, trainloader, testloader, num_epochs, device='cpu'):
    total_loss = []
    
    print(f'Model training ...')
    model.train()
    
    for epoch in range(num_epochs):
        print(f'EPOCH: {epoch + 1}')
        loss_on_epoch = []
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            inputs, masks = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs, 
                            attention_mask=masks, 
                            labels=inputs,
                            token_type_ids=None,
                           )
            loss, logits = outputs[0], outputs[1]
            
            loss_on_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            
        loss_on_epoch_mean = sum(loss_on_epoch) / len(loss_on_epoch)
        total_loss.append(loss_on_epoch)
        print(f'\tloss (mean): {loss_on_epoch_mean:.3f}')
        plt.plot(loss_on_epoch)
            
        print(f'Model evaluating ...')
        model.eval()
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(testloader):
                inputs, masks = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs, 
                                attention_mask=masks, 
                                labels=inputs, 
                                token_type_ids=None,                                
                               )
                loss, _ = outputs[0], outputs[1]
                losses.append(loss.item())
            print(f'\tloss: {sum(losses) / len(losses):.3f}')
    return total_loss