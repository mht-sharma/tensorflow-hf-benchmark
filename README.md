# tensorflow-hf-benchmark

This repository contains benchmarks for evaluating the performance of TensorFlow models from the Hugging Face Transformers library.It provides comprehensive comparison of model performance with plots. Currently the repository only support for sequence classification tasks.

Supported Features of TensorFlow Hugging Face Benchmarks Repository

* Benchmarks for **TensorFlow eager execution**
* Benchmarks for **TensorFlow XLA**
* Benchmarks for **PyTorch**
* TFLite export support
* Benchmarks for **TensorFlow Lite with XNNPACK**
* Benchmarks for **TensorFlow Lite without XNNPACK**

## Installation

TBA

## Results
The results presented are computed on **AWS C6I.X8large** EC2 instance

### Bert base

| Description | Latency (ms) |
| --- | --- |
| Tensorflow Eager  | 115.92 |
| Tensorflow Graph (XLA) | 17.89 |
| TFLite (Python) | 69.86 |
| TFLite wout XNNPack Bench. Tool | 78.83 |
| TFLite XNNPack (Bench. Tool) Threads (1) | 50.74|
| TFLite XNNPack (Bench. Tool) Threads (8) | 12.59 |
| TFLite XNNPack (Bench. Tool) Threads (16 | 10.98 |
| Pytorch | 10.25 |

<div style="text-align:center;">
![Models Latency Plot](https://github.com/mht-sharma/tensorflow-hf-benchmark/blob/main/plots/bert-base-cased_benchmark.png)
</div>

### Bert large

| Description | Latency (ms) |
| --- | --- |
| Tensorflow Eager  | 234.37 |
| Tensorflow Graph (XLA) | 45.63 |
| TFLite (Python) | 303.57 |
| TFLite wout XNNPack Bench. Tool | 312.27 |
| TFLite XNNPack (Bench. Tool) Threads (1) | 191.22|
| TFLite XNNPack (Bench. Tool) Threads (8) | 39.53 |
| TFLite XNNPack (Bench. Tool) Threads (16 | 31.04 |
| Pytorch | 27.35 |

<div style="text-align:center;">
![Models Latency Plot](https://github.com/mht-sharma/tensorflow-hf-benchmark/blob/main/plots/bert-large-cased_benchmark.png)
</div>

## Tensorflow Model Optimization Limitations
The section mentions the limitations of the features mentioned in [https://www.tensorflow.org/model_optimization](https://www.tensorflow.org/model_optimization)

* [**Pruning**](https://www.tensorflow.org/model_optimization/guide/pruning) - 