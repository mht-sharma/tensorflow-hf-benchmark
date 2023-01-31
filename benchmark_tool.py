import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from collections import OrderedDict
from pathlib import Path

from transformers import TFAutoModelForSequenceClassification

from utils import (convert_model_to_tflite, generate_input, plot_latency,
                   run_tflite_benchmark)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="bert-base-cased", type=str, help="Model to benchmark"
    )
    parser.add_argument(
        "--num_runs", type=int, default=100, help="Number of runs for benchmark"
    )
    parser.add_argument(
        "--warmup_runs", type=int, default=10, help="Warmup iterations for benchmark"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=8, help="Sequence length for benchmark"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch Size for benchmark"
    )
    parser.add_argument(
        "--use_xnnpack", help="Use xnnpack for benchmarking", action="store_true"
    )
    parser.add_argument(
        "--quantize_weights", help="Quantize tflite model weights", action="store_true"
    )
    parser.add_argument(
        "--tflite_path",
        type=Path,
        default="/tmp/",
        help="Path to save tflite model",
    )
    parser.add_argument(
        "--tf_source_path",
        type=Path,
        default=None,
        help="Path to tensorflow source model",
    )
    parser.add_argument(
        "--num_threads",
        nargs="+",
        type=int,
        default=[1],
        help="Number of threads for tflite benchmark tool",
    )

    parser.add_argument("--plot", help="Plot latencies", action="store_true")

    args = parser.parse_args()

    return args


def benchmark(
    model_name,
    num_runs=100,
    warmup_runs=10,
    sequence_length=8,
    batch_size=1,
    tflite_path="/tmp/",
    tf_source_path=None,
    quantize_weights=False,
    num_threads=[1, 2],
    use_xnnpack=False,
    plot=False,
):
    latencies = OrderedDict()

    input_names = ["input_ids", "attention_mask"]
    input_shape = [batch_size, sequence_length]

    model_tf = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    vocab_size = model_tf.config.vocab_size
    input_tf = {
        input_name: generate_input(input_name, "tf", input_shape, vocab_size)
        for input_name in input_names
    }

    tflite_model_path = tflite_path.joinpath("model.tflite")
    convert_model_to_tflite(
        model_tf, input_shape, tflite_model_path, quantize_weights=quantize_weights
    )

    benchmark_binary_path = f"{tf_source_path.as_posix()}/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"
    if not os.path.exists(benchmark_binary_path):
        raise OSError(
            "The tensorflow lite benchmark path does not exist, please build the binary to for benchmarking with multiple threads!"
        )

    for num_thread in num_threads:
        time_avg_ms = run_tflite_benchmark(
            benchmark_binary_path,
            tflite_model_path,
            num_thread,
            warmup_runs,
            num_runs,
            "true" if use_xnnpack else "false",
        )

        if len(num_threads) == 1:
            latencies[f"TFLite (Bench. Tool)"] = time_avg_ms
        else:
            latencies[f"TFLite (Bench. Tool) Threads[{num_thread}]: "] = time_avg_ms

    print(f"Average Latency (ms) - ")
    for desc, value in latencies.items():
        print(f"\t{desc}-{value:.2f}")

    if plot:
        plot_latency(model_name, latencies)


if __name__ == "__main__":
    args = argparser()
    benchmark(
        args.model_name,
        args.num_runs,
        args.warmup_runs,
        args.sequence_length,
        args.batch_size,
        args.tflite_path,
        args.tf_source_path,
        args.quantize_weights,
        args.num_threads,
        args.use_xnnpack,
        args.plot,
    )
