# rust-text-gen-from-scratch

**This README is a work in progress!**

A gated recurrent unit (GRU)-based recurrent neural network to generate text. Written in Rust, from scratch!

## Overview

If you're familiar with recurrent neural networks (RNNs), you've probably seem [something like this](https://www.tensorflow.org/text/tutorials/text_generation)—an RNN that trains on some corpus of text, and is then used to _generate_ new text.

While I thought this was fascinating, I felt as if the copy-and-paste nature of many of these tutorials left something out; I wanted to do the same thing, but from scratch.

So I did! This project implements:

- Forward and backward passes (as you'd expect!)
- Support for using BLAS for matrix algebra
- A whopping two optimizers: SGD and Adam
- Support for saving parameters and resuming training later
- Deterministic pseudo-random parameter initialization

## Layers

1. An input layer with each node corresponding to each possible character in the corpus.
2. A gated recurrent unit (GRU) layer—nothing special.
3. An output layer, producing the usual normalized character probability distribution.

## Text Generation

To generate text, we begin by feeding the network one-hot encodings of each character of the prompt. Once the prompt has been exhausted, we begin to generate new text by randomly choosing a character from the probability distribution created by the output layer.

## Differentiation

This was *by far* that hardest part to get right. I had originally tried to do backprop using a [forward accumulation](https://math.stackexchange.com/questions/2195377/reverse-mode-differentiation-vs-forward-mode-differentiation-where-are-the-be) like strategy where the derivative of the loss was calculated *during* the forward pass. While this was probably simpler when in implementation, it was just way too slow since we ended up having to perform several-dimension by several-dimension matrix multiplications.

Instead, I went with a more traditional reverse accumulation like strategy where the values produced during forward propagation that are required during back propagation are simply saved in memory, returned from the forward prop routine, and used when needed during backprop.