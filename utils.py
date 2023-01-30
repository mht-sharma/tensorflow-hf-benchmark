import random
import re
import subprocess
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)


set_seed()


def measure_latency(test_model, encoded_input, warmup_runs=10, num_runs=100):
    latencies = []
    # warm up
    for _ in range(warmup_runs):
        _ = test_model(**encoded_input)
    # Timed run
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = test_model(**encoded_input)
        latency = time.perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    return time_avg_ms, time_std_ms


def random_int_tensor(shape, max_value: int, min_value: int = 0, framework: str = "tf"):
    if framework == "pt":
        return torch.randint(low=min_value, high=max_value, size=shape)
    if framework == "tf":
        tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tf.int32)
    return np.random.randint(low=min_value, high=max_value, size=shape, dtype=np.int32)


def generate_input(
    input_name: str, framework: str = "tf", shape=[1, 8], vocab_size=100
):
    min_value = 1
    max_value = 2 if input_name != "input_ids" else vocab_size
    return random_int_tensor(shape, max_value, min_value=min_value, framework=framework)


def model_to_signatures(model, input_shape=[1, 8], **model_kwargs):
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    def forward(*args):
        if len(args) != len(input_names):
            raise (
                f"The number of inputs provided ({len(args)} do not match the number of expected inputs: "
                f"{', '.join(input_names)}."
            )
        kwargs = dict(zip(input_names, args))
        outputs = model.call(**kwargs, **model_kwargs)
        return {key: value for key, value in outputs.items() if key in output_names}

    vocab_size = model.config.vocab_size
    dummy_inputs = {
        input_name: generate_input(input_name, "np", input_shape, vocab_size)
        for input_name in input_names
    }
    inputs_specs = [
        tf.TensorSpec(dummy_input.shape, dtype=dummy_input.dtype, name=input_name)
        for input_name, dummy_input in dummy_inputs.items()
    ]

    function = tf.function(
        forward, input_signature=inputs_specs
    ).get_concrete_function()

    return {"model": function}


def convert_model_to_tflite(
    model,
    input_shape=[1, 8],
    output_path=None,
    quantize_fp16=False,
    quantize_weights=False,
):

    signatures = model_to_signatures(model, input_shape=input_shape)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tf.saved_model.save(model, tmp_dir_name, signatures)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir_name)

        if quantize_weights:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantize_fp16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

    if output_path is not None:
        print("Writing tensorflow lite model to: %s" % output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "wb") as fp:
            fp.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    runner = interpreter.get_signature_runner("model")

    return runner


def run_tflite_benchmark(
    benchmark_binary_path,
    tflite_model_path,
    num_threads,
    warmup_runs,
    num_runs,
    use_xnnpack,
):
    cmd = (
        f"{benchmark_binary_path} --graph={tflite_model_path} "
        f"--num_threads={num_threads} --warmup_runs={warmup_runs} "
        f"--num_runs={num_runs} --use_xnnpack={use_xnnpack}"
    )

    result = subprocess.run(
        cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout.decode("utf-8")

    result = result.split("\n")[-4]

    time_avg_ms = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", result)[-1]) / 1000

    return time_avg_ms


def plot_latency(model_name, latencies):
    desc = list(latencies.keys())
    values = list(latencies.values())

    x = 1
    for i in range(len(values)):
        plt.bar(x, values[i], width=0.4)
        x += 0.5

    plt.ylabel("Latency (ms)")
    plt.title(f"{model_name.replace('/', '-')} CPU benchmark")
    plt.legend(desc)
    plt.savefig(f"{model_name.replace('/', '-')}_benchmark.png")
