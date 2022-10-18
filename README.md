# rust-text-gen-from-scratch

**This README is a work in progress!**

A gated recurrent unit (GRU)-based recurrent neural network to generate text. Written in Rust, from scratch!

This project implements:

- Forward and backward passes (as you'd expect!)
- Support for using BLAS for matrix algebra
- A whopping two optimizers: SGD and Adam
- Support for saving parameters and resuming training later