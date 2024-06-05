def parameters_counter(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


def num_params_with_description(model, num_layers_to_unfreeze=2):
  parameters_before = parameters_counter(model)
  print(f'Number of parameters before: {parameters_before}')

  num_layers = len(list(model.parameters()))

  for p in model.parameters():
      p.requires_grad = False

  for p in list(model.parameters())[-num_layers_to_unfreeze:]:  
      p.requires_grad = True

  parameters_after = parameters_counter(model)
  print(f'Number of layer unfreezed: {num_layers_to_unfreeze}')
  print(f'Number of parameters after : {parameters_after}')

  return 