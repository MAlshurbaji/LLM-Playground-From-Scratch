# LLM-Playground-From-Scratch

**LLM-Playground-From-Scratch** is an educational, end-to-end project demonstrating how to build a miniature Large Language Model (LLM) entirely from scratch using Python and PyTorch.  
The repository processes dozens of real LLM technical reports (GPT-3, GPT-4, LLaMA, PaLM, Gemini, Qwen, DeepSeek, and more), converts them into a structured dataset, tokenizes them, trains a small GPT-style model, and fine-tunes it on a custom instruction-following dataset of 2000+ pairs.

> **Note:** This project is for **learning purposes only**.  
> The resulting models are not intended to be competitive.

---
## Inspiration & References
This project is inspired by the book: **Build a Large Language Model (From Scratch)** https://amzn.to/4fqvn0D, and includes components adapted from: https://github.com/rasbt/LLMs-from-scratch

---

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
- Training a GPT-2-like model **from scratch**  
- Fine-tuning on custom instruction-following data  
- Evaluation using:
  - BLEU  
  - ROUGE  
  - METEOR  
  - F1 Score  
  - Model-as-a-Judge (e.g., Phi-3-Mini)

---

## Dataset Sources

This project uses **38 public LLM technical reports** from major research labs:

- OpenAI (GPT-3, GPT-4)  
- Google DeepMind (PaLM, PaLM-2, Gemini series, Gemma)  
- Meta (LLaMA series)  
- Alibaba (Qwen series)  
- DeepSeek  
- Mistral  
- NVIDIA  
- Microsoft  
- 01.AI (Yi)  
- Shanghai AI Lab (InternLM, InternVL)  
- Others

A detailed list is available at:  
`report/project_overview.txt`

---

## Usage

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
