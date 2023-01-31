import argparse
import subprocess
from collections import OrderedDict
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


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
        "--sequence_length",
        nargs="+",
        type=int,
        default=[8],
        help="Sequence length for benchmark",
    )
    parser.add_argument(
        "--batch_size",
        nargs="+",
        type=int,
        default=[1],
        help="Batch Size for benchmark",
    )
    parser.add_argument(
        "--pt", help="Include Pytorch model in benchmark", action="store_true"
    )
    parser.add_argument(
        "--ignore_tflite", help="Ignore tflite inference", action="store_true"
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
    sequence_length=[8],
    batch_size=[1],
    pt=False,
    ignore_tflite=False,
    tflite_path="/tmp/",
    tf_source_path=None,
    quantize_weights=False,
    num_threads=[1, 2],
    plot=False,
):
    latencies_bs = OrderedDict()
    for bs in batch_size:
        latencies_sl = OrderedDict()
        for sl in sequence_length:
            latencies_thread = OrderedDict()
            for i, num_thread in enumerate(num_threads):
                inter_latency = OrderedDict()
                print("-" * 100)
                print(f"Running, [BS-{bs}|SL-{sl}|NT-{num_thread}] ")

                cmd = (
                    f"numactl -C 0-{num_thread-1} "
                    f"python benchmark_python.py "
                    f"--model_name={model_name} "
                    f"--num_runs={num_runs} --warmup_runs={warmup_runs} "
                    f"--sequence_length={sl} --batch_size={bs} --tflite_path={tflite_path}"
                )

                cmd += " --ignore_tflite" if ignore_tflite else ""
                cmd += " --pt" if pt else ""

                print(cmd)

                result = subprocess.run(
                    cmd.split(" "),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ).stdout.decode("utf-8")

                result = result.split("\n")[:-1]
                for idx in range(len(result) - 1, 0, -1):
                    if "Average Latency (ms)" in result[idx]:
                        break
                    desc, value = result[idx].split("-")
                    inter_latency[desc[1:]] = float(value)

                if not ignore_tflite:
                    quantize_values = [False, True] if quantize_weights else [False]
                    for quantize_value in quantize_values:
                        for use_xnnpack in [False, True]:
                            cmd = (
                                f"python benchmark_tool.py --model_name {model_name} "
                                f"--num_runs {num_runs} --warmup_runs {warmup_runs} "
                                f"--sequence_length {sl} --batch_size {bs} --tflite_path {tflite_path} "
                                f"--tf_source_path {tf_source_path} --num_threads {num_thread}"
                            )

                            cmd += " --quantize_weights" if quantize_value else ""
                            cmd += " --use_xnnpack" if use_xnnpack else ""

                            print(cmd)

                            result = subprocess.run(
                                cmd.split(" "),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                            ).stdout.decode("utf-8")

                            result = result.split("\n")[:-1]
                            for idx in range(len(result) - 1, 0, -1):
                                if "Average Latency (ms)" in result[idx]:
                                    break
                                desc, value = result[idx].split("-")
                                inter_latency[
                                    desc[1:]
                                    + f" XNNPack[{use_xnnpack}]"
                                    + f" Quantize[{quantize_value}]"
                                ] = float(value)

                latencies_thread[num_thread] = inter_latency

                # print(f"Average Latency (ms) - ")
                # for desc, value in inter_latency.items():
                #     print(f"\t{desc} - {value:.2f}")

            latencies_sl[sl] = latencies_thread

        latencies_bs[bs] = latencies_sl

    for bs in batch_size:
        for sl in sequence_length:
            for num_thread in num_threads:
                print("-" * 100)
                print(f"Average Latency [BS-{bs}|SL-{sl}|NT-{num_thread}] (ms) - ")
                inter_latency = latencies_bs[bs][sl][num_thread]
                for desc, value in inter_latency.items():
                    print(f"\t{desc} - {value:.2f}")

    if plot:
        color = [colors.to_hex(np.random.rand(4)) for i in range(20)]

        # BS SL
        desc = None
        for bs in batch_size:
            fig, axes = plt.subplots(len(sequence_length), len(num_threads))
            for i, sl in enumerate(sequence_length):
                for j, num_thread in enumerate(num_threads):
                    inter_latency = latencies_bs[bs][sl][num_thread]
                    desc = list(inter_latency.keys())
                    values = list(inter_latency.values())

                    x = 1
                    for k in range(len(values)):
                        bars = axes[i, j].bar(x, values[k], width=0.4, color=color[k])
                        for l, bar in enumerate(bars):
                            height = bar.get_height()
                            axes[i, j].text(
                                bar.get_x() + bar.get_width() / 2,
                                height / 2,
                                str(int(height)),
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )

                        x += 0.5

                    axes[i, j].set_title(
                        f"BS-{bs}, SL-{sl}, NT-{num_thread}", fontsize=6
                    )
                    axes[i, j].set_xticks([])
            fig.text(
                0.05, 0.5, "Latency (ms)", ha="center", va="center", rotation="vertical"
            )
            fig.suptitle(
                f"{model_name.replace('/', '-')} CPU benchmark - BS {bs}",
                fontsize=14,
                fontweight="bold",
                y=0.05,
            )
            fig.legend(desc, fontsize=6)
            plt.savefig(f"{model_name.replace('/', '-')} CPU benchmark - BS {bs}")
            plt.clf()


if __name__ == "__main__":
    args = argparser()
    benchmark(
        args.model_name,
        args.num_runs,
        args.warmup_runs,
        args.sequence_length,
        args.batch_size,
        args.pt,
        args.ignore_tflite,
        args.tflite_path,
        args.tf_source_path,
        args.quantize_weights,
        args.num_threads,
        args.plot,
    )
