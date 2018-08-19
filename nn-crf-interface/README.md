## StatNLP Neural Network Interface

This document describes how to integrate the neural network into usual StatNLP graphical model implementations. You may want to use continuous feature value instead of binary feature value, or a neural network component such as Long short-term memory (LSTM) to improve the model performance. We will go through the procedure step by step based on previous StatNLP implementations.


### Table of Contents
- [Requirements](#requirements)

### Requirements
* Torch 7 (installed with Lua 5.2)
* [Element-Research/rnn](https://github.com/Element-Research/rnn)
* JNLua (follow the instructions under `install_jnlua` folder)
