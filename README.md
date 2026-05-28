# 📚 Question Answering with DistilBERT

A complete implementation of a question answering system using fine-tuned DistilBERT on the bAbI dataset.

## 📋 Overview

This project implements a **question answering system** using DistilBERT, a distilled version of BERT that is smaller and faster while maintaining high accuracy. The model is fine-tuned on the bAbI dataset to answer questions based on given context passages.

### Key Features:
- ✅ **Extractive QA** - Extracts answer spans from context text
- ✅ **DistilBERT architecture** - Smaller, faster, and 95% of BERT's performance
- ✅ **Dual framework support** - Both TensorFlow and PyTorch implementations
- ✅ **bAbI dataset** - 20 tasks for reasoning and comprehension
- ✅ **Token alignment** - Maps character-level answers to token positions

## 🧠 How It Works
Input: Context + Question
↓
DistilBERT Tokenizer
↓
Token IDs + Attention Mask
↓
DistilBERT Model
↓
Start Logits + End Logits
↓
Argmax over logits
↓
Extract Answer Span
↓
Output: Answer Text


## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- PyTorch 1.x
- Hugging Face Transformers

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/question-answering-distilbert.git
cd question-answering-distilbert

pip install -r requirements.txt