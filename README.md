# Stylometry: Keystroke Dynamics Classification

This repository contains the code and research for **Stylometry**, a project that identifies users based on their typing behavior (keystroke dynamics) using a deep Recurrent Neural Network (Bi-LSTM).

## Overview

Stylometry captures user keystroke data—specifically **Hold Time**, **Flight Time**, and **Key Codes**—to build a unique "neural signature" for each user. The system consists of:

- **Frontend:** A Svelte-based web interface for data collection and real-time inference.
- **Backend:** A Python/PyTorch API that handles model training, data processing, and inference.

## Features

- **Data Collection:** Web interface to capture keystroke timing with millisecond precision.
- **Bi-LSTM Model:** A Bidirectional Long Short-Term Memory network trained to classify users based on temporal sequences.
- **Real-time Inference:** Identify typists on the fly using trained models.
- **Visualizations:** Confusion matrices and typing pattern analysis.

## Architecture

The core model is a **2-layer Bidirectional LSTM** with:

- **Input:** Sequence of 43 keystrokes (Hold Time, Flight Time, Key Code).
- **Hidden Size:** 64 units.
- **Output:** User classification logits.
