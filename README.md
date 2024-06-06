# SMILESwithGPT2

## Overview

This repository focuses on the generation of molecules using Large Language Models (LLMs). Primary objective is to fine-tune the GPT-2 model on the ZINC250 dataset, aiming to generate valid SMILES (Simplified Molecular Input Line Entry System) strings that can be visualized using the `rdkit.Chem.MolFromSmiles()` tool. Correctly generated molecules will appear as graphs.

The generation of molecules is performed using the built-in `.generate(max_length=256, early_stopping=True, num_return_sequences=32)` function, and experiment with different temperature settings: `[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]`.

Common issues in generating SMILES include unpaired parentheses, incomplete rings, and invalid symbols. 

### Input Size Dependancy 
When generating molecules using the pre-trained GPT-2 model, the output is highly dependent on the input size. If the input length is less than 8-11 characters (including brackets), the model frequently returns the input itself.

### Validation 
The validity of generated molecules is determined by the number of graphs successfully produced using `rdkit.Chem.MolFromSmiles()` compared to the total number of generated molecules at each temperature setting. An output is considered "not changed (no\ch)" if all 32 generated SMILES strings are identical to the prompt.


## Zero-shot GPT2Head model results
|           prompt       | length |        temperature           |   valid, %  |
|:----------------------:|:------:|:----------------------------:|:-----------:|
|         C              |    1   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]| not changed |
|         CC             |    2   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]| not changed |
|        CCO             |    3   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]| not changed |
|      CCOC(=O)          |    8   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]|      0      |
|     CCCCC(=O)NC        |   11   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]| not changed |
|    CCN(CC)C(=O)C       |   13   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]|      0      |
|   C[C@@H](NC(=O)COC    |   17   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]|      0      |
| O=c1n(CCO)c2ccccc2n1CC |   22   |[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]|      0      |

Generated examples: \
`CCOC(=O) :: single`\
`CCN(CC)C(=O)C THE G G H the`\
`C[C@@H](NC(=O)COC F F 39 39 39 39 39 39`\
`O=c1n(CCO)c2ccccc2n1CC Windows -- -- --` \
`'CCN(CC)C(=O)C May`



## Fine-tuned GPT2Head model results
The model trained last two layers (1536 params.) on ZINC250 dataset for 5 epochs with `torch.optim.Adam(lr=3e-4)` on batch 64.

<!---|           prompt       | length |temperature|valid, %|temperature|valid, %|temperature|valid, %|temperature|valid, %|temperature|valid, %|temperature|valid, %|
|:----------------------:|:------:|:---------:|:------:|:---------:|:------:|:---------:|:------:|:---------:|:------:|:---------:|:------:|:---------:|:------:|
|         C              |    1   |   1.1     |  no\ch |   1.2     |  no\ch |   1.3     |  no\ch |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch |
|         CC             |    2   |   1.1     |  no\ch |   1.2     |  no\ch |   1.3     |  no\ch |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch |
|        CCO             |    3   |   1.1     |  no\ch |   1.2     |  no\ch |   1.3     |  no\ch |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch |
|      CCOC(=O)          |    8   |   1.1     |    0   |   1.2     |    0   |   1.3     |    0   |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch |
|     CCCCC(=O)NC        |   11   |   1.1     |  no\ch |   1.2     |  no\ch |   1.3     |  no\ch |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch |
|    CCN(CC)C(=O)C       |   13   |   1.1     |    0   |   1.2     |    0   |   1.3     |  no\ch |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch |
|   C[C@@H](NC(=O)COC    |   17   |   1.1     |    0   |   1.2     |    0   |   1.3     |  no\ch |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch |
| O=c1n(CCO)c2ccccc2n1CC |   22   |   1.1     |    0   |   1.2     |    0   |   1.3     |  no\ch |   1.4     |  no\ch |   1.5     |  no\ch |   1.6     |  no\ch | -->



## Usage
To use this repository, follow these steps:
1. Clone the repository:
```
git clone https://github.com/dorochka8/SMILESwithGPT2.git
```

2. Install the required dependencies:
```
pip install transformers rdkit 
```

3. Run the main.py script 
