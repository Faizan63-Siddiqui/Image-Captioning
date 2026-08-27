# Image Captioning with CNN + LSTM

A deep learning system that generates natural language captions for images, built using a CNN encoder (ResNet50) and an LSTM decoder, trained from scratch on the Flickr8k dataset.

## Demo

Upload any image and the model generates a descriptive caption in real time via a Gradio web interface.

![Demo Screenshot](sample_outputs/demo_screenshot.png)

## Overview

This project implements an encoder-decoder architecture for automatic image captioning:

- **Encoder (CNN):** A pretrained ResNet50 (ImageNet weights, frozen) extracts a 2048-dimensional feature vector summarizing the visual content of an image.
- **Decoder (LSTM):** A custom LSTM network takes the image feature vector as its initial hidden state and generates a caption word-by-word, conditioned on the words generated so far.

This is the classic "Show and Tell" style architecture — a strong baseline for image captioning, without an attention mechanism.

## Architecture

```
Input Image (224x224x3)
        |
   ResNet50 (frozen, pretrained on ImageNet)
        |
   2048-dim feature vector
        |
   Linear projection -> LSTM initial hidden state
        |
   LSTM decoder (word embedding -> LSTM -> vocabulary softmax)
        |
   Generated caption, word by word
```

**Model details:**
- Embedding dimension: 256
- LSTM hidden dimension: 512
- Vocabulary size: 2,988 words (frequency threshold: min 5 occurrences)
- Total trainable parameters: ~4.9M
- Max caption length: 35 tokens

## Dataset

**Flickr8k** — 8,091 images, each paired with 5 human-written reference captions (40,455 total captions).

| Split | Images | Captions |
|---|---|---|
| Train | 6,877 (85%) | 34,385 |
| Validation | 809akes the image feature vector as its initial hidden state and generates a caption word-by-word, conditioned on the words generated so
