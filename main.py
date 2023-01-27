import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TFAutoModelForSequenceClassification)

from utils import generate_input


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model to benchmark")
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
        "--pt", help="Include Pytorch model in benchmark", action="store_true"
    )
    parser.add_argument(
        "--ignore_tflite", help="Ignore tflite inference", action="store_true"
    )
    parser.add_argument(
        "--tflite_save_path",
        type=Path,
        default="/tmp/",
        help="Path to save tflite model",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = argparser()

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    input_shape = [args.batch_size, args.sequence_length]

    model_tf = TFAutoModelForSequenceClassification.from_pretrained(args.model_name)
    vocab_size = model_tf.config.vocab_size
    input_tf = {
        input_name: generate_input(input_name, "tf", input_shape, vocab_size)
        for input_name in input_names
    }

    if args.pt:
        model_pt = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        input_pt = {
            input_name: generate_input(input_name, "pt", input_shape, vocab_size)
            for input_name in input_names
        }

    if not args.ignore_tflite:
        tflite_models_dir = args.tflite_save_path
