# Day 1 — First Pretrained Models

## What I did
Ran two pretrained models from Hugging Face using the `transformers` library.

## Key Concepts

### Transformers (the library)
A library by Hugging Face that makes it easy to use transformer-based models. Not the same as transformer architecture — that is the neural network design. This is the tool.

### Pipeline
A shortcut that handles everything internally — loading the model, tokenizing input, running inference, decoding output. Without pipeline, the same task would take 20-30 lines of code.

### Hugging Face Hub
A public repository of models — like GitHub for code but for models. Anyone can upload or download models. Models are referenced by name, for example `gpt2` or `distilbert-base-uncased-finetuned-sst-2-english`.

### Progress Bars
When a model loads for the first time, it downloads from HF Hub. `model.safetensors` contains the actual learned weights. `vocab.json` and `tokenizer.json` contain the vocabulary. These download every new Colab session.

## Model 1 — DistilBERT
- **Task:** Text Classification
- **Input:** "This news article seems very suspicious and emotionally charged."
- **Output:** POSITIVE, 0.81 confidence
- **Observation:** Called a suspicious sentence POSITIVE because it was trained on movie reviews, not misinformation data. This is the difference between a general model and a fine-tuned model — same architecture, different training data, completely different behavior.

## Model 2 — GPT2
- **Task:** Text Generation
- **Input:** "Misinformation spreads because"
- **Output:** A hallucinated news article with fake people and fake organizations.
- **Why hallucination happened:** GPT2 has no discriminator, no fact-checking mechanism. It is a language model — its only job is to predict the next probable word based on patterns in training data. It does not know what is true. It knows what is probable.

## Two Fundamentally Different Tasks
| Task | What it does | Example model |
|------|-------------|---------------|
| Understanding | Classify, analyze existing content | DistilBERT |
| Generation | Create new content | GPT2 |

Mirra needs both — understand the content first, then generate the analysis.

## GPU Note
Both models ran on CPU. For inference this is fine, just slower. GPU will be needed when fine-tuning begins.

## Mirra Connection
Currently Gemini handles both understanding and generation inside Mirra. After fine-tuning, a local model will do this — trained specifically on Mirra's philosophy of trust analysis, not general text patterns.

## Open Question
Why does the same architecture behave so differently with different training data? This is what fine-tuning is about — same structure, different knowledge baked in.