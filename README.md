# MiniNN

**MiniNN** is a lightweight neural network library written in C++ from scratch.

This project was created primarily for **practice and educational purposes**, with the goal of understanding how neural networks work at a low level — without relying on external ML frameworks.

## Features

- Fully custom implementation of:
  - Matrix operations
  - Dense (fully-connected) layers
  - Activation functions (ReLU, Sigmoid, Softmax)
  - Loss functions (MSE, CrossEntropy)
- Support for:
  - Feedforward (MLP) networks
  - Backpropagation
  - Mini-batch training
- Simple model serialization (save/load)
- Optional CPU parallelization (OpenMP)

## Goals

- Learn and implement core ML concepts manually
- Understand data flow, gradients, and optimization
- Build a minimal but functional neural network framework

## Status

The library is capable of training simple models (e.g. MNIST digit classification) and serves as a foundation for further experimentation.

---

> This is a learning project, not a production-ready ML framework.
