# LLM-Playground-From-Scratch

**LLM-Playground-From-Scratch** is an educational, end-to-end project demonstrating how to build a miniature Large Language Model (LLM) entirely from scratch using Python and PyTorch.  
The repository processes dozens of real LLM technical reports (GPT-3, GPT-4, LLaMA, PaLM, Gemini, Qwen, DeepSeek, and more), converts them into a structured dataset, tokenizes them, trains a small GPT-style model, and fine-tunes it on a custom instruction-following dataset of 2000+ pairs.

This project is inspired by the book: **Build a Large Language Model (From Scratch)** https://amzn.to/4fqvn0D, and includes components adapted from: https://github.com/rasbt/LLMs-from-scratch

> **Note:** This project is for **learning purposes only**.  
> The resulting models are not intended to be competitive.

## Project Overview
This repository demonstrates the complete workflow of building a GPT-style LLM, including:

- Downloading and managing a large set of LLM technical reports  
- PDF text extraction  
- Cleaning & preprocessing of noisy scientific text  
- Custom regex-based tokenizer implementation  
- Byte Pair Encoding (BPE) tokenization with *tiktoken*  
- Vocabulary construction  
- Sliding-window dataset creation for autoregressive modeling  
- PyTorch `Dataset` and `DataLoader`  
- Training a GPT-2-like model and a Regex model **from scratch**  
- Fine-tuning on custom instruction-following data  
- Evaluation using:
  - BLEU, ROUGE, METEOR, and F1 Score  
  - Model-as-a-Judge (e.g., Phi-3-Mini)

## Tokenizers and Model Variants
This project uses **two different tokenizers**, and therefore trains **two separate models** from scratch:
1. **Custom Regex Tokenizer**
   - Vocabulary built from scratch using regex-based tokenization.
   - Model: `regex_pretrained.pth` → fine-tuned → `regex_finetuned.pth`

2. **GPT-2 Tokenizer (tiktoken)**
   - Uses OpenAI's GPT-2 BPE vocabulary.
   - Model: `gpt2_pretrained.pth` → fine-tuned → `gpt2_finetuned.pth`

Additionally, we download the official **pretrained GPT-2 model** (referred to here as **gpt-2o**) → fine-tuned → `gpt_finetuned.pth`

This allows us to compare:
- Two models trained from scratch (Regex vs GPT-2 tokenizer)
- One pretrained baseline (GPT-2 original weights)

## Dataset Sources
This project uses **38 public LLM technical reports** from major research labs. A detailed list of these reports is available at `data/raw_pdf/pdfs_list.txt`, all of them can be downloaded online.

## Usage
Navigate to the main project directory, install all dependencies (tested on Python 3.10, PyTorch 2.8, and CUDA 12.6), and run the preprocessing workflow and the end-to-end training using the notebook `main.ipynb`.

```bash
cd LLM_Playground_From_Scratch
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Instruction Dataset for Fine-Tuning (Optional)
The instruction-following dataset used to fine-tune the models can be **generated directly from the notebook** (automatic summarization, explanation, and Q&A generation), or can also be found in `data/instructions_dataset.json`.

Model responses on the validation set (used for scoring and evaluation) can be **generated directly from the notebook**, or can also be found in `data/model_responses/`.

These files contain all Alpaca-style prompts and the corresponding model outputs.

## Pretrained & Fine-Tuned Models (Optional)
You can train your own models directly inside the notebook (`main.ipynb`). However, to save time, all pretrained and fine-tuned models are provided via Hugging Face.
After downloading, place all model files into `models/`.

| Model Description | Download Link |
|------------------|----------------|
| Regex Model — Pretrained | https://huggingface.co/AlShurbaji/LLM-Playground-From-Scratch/resolve/main/regex_pretrained.pth |
| Regex Model — Fine-tuned | https://huggingface.co/AlShurbaji/LLM-Playground-From-Scratch/resolve/main/regex_finetuned.pth |
| GPT-2 Model — Pretrained | https://huggingface.co/AlShurbaji/LLM-Playground-From-Scratch/resolve/main/gpt2_pretrained.pth |
| GPT-2 Model — Fine-tuned | https://huggingface.co/AlShurbaji/LLM-Playground-From-Scratch/resolve/main/gpt2_finetuned.pth |
| GPT-2o Model — Fine-tuned | https://huggingface.co/AlShurbaji/LLM-Playground-From-Scratch/resolve/main/gpt2o_finetuned.pth |

## My Results
Below are the evaluation metrics for the three models:

| Metric | Regex | GPT-2 (124M) | GPT-2 (124M) - Pretrained Weights |
| ----------------- | -------------------------------- | --------------------------- | ---------------------------------- |
| **BLEU-4** | 4.99 (BP=1.00, P1=24.68, P2=7.71, P3=2.84, P4=1.14) | 9.18 (BP=1.00, P1=35.78, P2=12.66, P3=5.61, P4=2.79) | **20.26** (BP=1.00, P1=44.91, P2=23.15, P3=15.05, P4=10.77) |
| **ROUGE-1 / ROUGE-2 / ROUGE-L-F1** | 0.383 / 0.114 / 0.198   | 0.400 / 0.120 / 0.212  | **0.510 / 0.244 / 0.312**      |
| **METEOR**         | 0.237    | 0.235       | **0.357**                                                   |
| **Token-F1**   | 0.299 (P=0.250, R=0.373)   | 0.324 (P=0.306, R=0.345)  | **0.426** (P=0.394, R=0.464)                |
| **BERTScore**   | −0.104 (P=−0.239, R=0.039)  | 0.061 (P=0.009, R=0.115)    | **0.268** (P=0.227, R=0.311)               |
| **Judge Model Score**       | 43.3                | 49.9            | **68.6**      |


Think you can beat these results? Go ahead, and keep me in touch ;)

## What You Can Tweak to Improve Results
This project is intentionally designed so you can experiment with many components and see how they influence model quality. Here are the main knobs you can adjust:

### 1. PDF → Text Cleaning Pipeline
- Modify thresholds for removing headers/footers.
- Change `min_words` in subsection splitting (affects dataset size).
- Adjust how aggressively captions, junk, or numeric artifacts are removed.

### 2. Tokenization
- Switch between:
  - Regex tokenizer (small vocabulary, simpler behaviour)
  - GPT-2 BPE tokenizer (larger vocabulary, better generalization)
- Experiment with different regex splits or custom vocab pruning.

### 3. GPT Model Architecture
- `emb_dim`, `n_layers`, `n_heads`, `context_length`
- Dropout rate (`drop_rate`)
- Whether to use `qkv_bias=True/False`

### 4. Dataset Windowing
- `context_size`
- `stride`
- Sliding window overlap ratio
- Train/val split proportions

### 5. Training Hyperparameters
- Learning rate (`lr`)
- Weight decay
- Batch size
- Number of epochs
- Scheduling or warmup (you can add your own)

### 6. Sampling Parameters (Generation Quality)
- `temperature`
- `top_k`
- `max_new_tokens`
Different values create very different outputs—from deterministic to creative.

### 7. Instruction–Response Dataset Generation
- Number of synthetic pairs generated
- Types of tasks (summarization, explanation, Q&A…)
- Max character limits per chunk
- Choice of LLM used for generating responses (Qwen, Mistral, etc.)

### 8. Evaluation Methods
- Add more LLM-Judge scoring with different judge models

By adjusting these components (and many more), you can improve or degrade the model—so feel free to experiment and push your own custom LLMs to their limits!
> Note that you are not restricted with the data given, and you can use your own data to train your LLM.

