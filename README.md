# Image Captioning with CNN + LSTM

A deep learning system that generates natural language captions for images, built using a CNN encoder (ResNet50) and an LSTM decoder, trained from scratch on the Flickr8k dataset.

## Demo

Upload any image and the model generates a descriptive caption in real time via a Gradio web interface.

*(Add your demo screenshot here — e.g. `sample_outputs/demo_screenshot.png`)*

```markdown
![Demo Screenshot](sample_outputs/demo_screenshot.png)
```

## Overview

This project implements an encoder-decoder architecture for automatic image captioning:

- **Encoder (CNN):** A pretrained ResNet50 (ImageNet weights, frozen) extracts a 2048-dimensional feature vector summarizing the visual content of an image.
- **Decoder (LSTM):** A custom LSTM network takes the image feature vector as its initial hidden state and generates a caption word-by-word, conditioned on the words generated so
