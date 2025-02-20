# Mechanistic Interpretability Project: Activation Patching Implementation

## Overview
This project implements and explores a key finding from the paper "How To Use And Interpret Activation Patching" by Stefan Heimersheim and Neel Nanda (April 2024). The implementation focuses on the Activation Patching technique to analyze the Indirect Object Identification (IOI) task by examining residual stream activations across different layers and positions.

## Project Goals
- Implement the Activation Patching technique from Neel Nanda's notebook
- Analyze the IOI task through residual stream activation patching
- Visualize and interpret the results of activation patching across different layers
- Compare findings with the original paper's conclusions

## Setup Instructions

### Prerequisites
The project requires several Python packages which are listed in `requirements.txt`. You can choose either Michael's or Neel's setup process:

#### Michael's Setup (Simplified)
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Import necessary libraries:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import torch
import einops
from functools import partial
import tqdm.auto as tqdm
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
```

#### Neel's Setup (Original)
1. Install specific packages:
```python
pip install transformer_lens
pip install circuitsvis
```

2. Additional setup for Colab (if needed):
```python
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## Implementation Steps

1. **Model Loading**
```python
device = utils.get_device()
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
torch.set_grad_enabled(False)
```

2. **IOI Task Setup**
- Implements clean and corrupted prompts for testing
- Calculates logit differences between correct and incorrect answers

3. **Activation Patching**
- Implements residual stream patching hook
- Iterates over layers and positions
- Calculates normalized logit differences

4. **Visualization**
- Creates heatmap of normalized logit differences
- Visualizes impact across layers and positions

## Key Findings

1. **High-Impact Positions**
- Critical tokens identified around "gave", "medal", and "to"
- Later tokens show strongest influence on indirect object identification

2. **Critical Layers**
- Layers 9-11 show highest impact
- Upper layers crucial for integrating contextual information
- Lower layers (0-8) show minimal impact

3. **Activation Patterns**
- Successfully restored logit differences in critical areas
- Clear pattern of layer specialization observed

## Future Extensions
- Test implementation on larger models (e.g., GPT-3)
- Combine with other techniques like Head Ablation
- Explore additional prompt variations

## Requirements
See `notebooks/requirements.txt` for complete list of dependencies.

## References
- [Heimersheim, S., & Nanda, N. (2024). How To Use And Interpret Activation Patching](https://arxiv.org/abs/2404.15255)
- 
- [Nanda, N. (2025). Attribution Patching: Activation Patching At Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- 
- [Nanda, N. Quickstart Guide for Mechanistic Interpretability](https://alignmentjam.com/post/quickstart-guide-for-mechanistic-interpretability)
- 
- [Bereska, L., & Gavves, E. Mechanistic Interpretability for AI Safety — A Review](https://leonardbereska.github.io/blog/2024/mechinterpreview/)
- 
- [Neel Nanda's *Main Demo* Google Colab Notebook on Mechanistic Probability](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=h8lPgmajfJuM)

- [ChatGPT](https://chatgpt.com/)

- [GeminiAI](https://gemini.google.com/app)

- [ClaudeAI](https://claude.ai/new)
   
---

📚 **Author of Notebook:** Michael Dankwah Agyeman-Prempeh [MEng. DTI '25]