"""Benchmark precomputed-pair loading against eager dynamic construction.

The benchmark exercises the production eager loader against legacy pair files,
validates equivalent samples, and runs real GPU optimizer steps through three
loader variants.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config_cesm
from src.experiment_configs import load_config
from src.models.losses import WeightedMSELoss
from src.models.models import UNetRes3
from src.models.models_util import CESM_Dataset, EagerDynamicCESMDataset
from src.models.optim import build_optimizer
from src.utils import util_cesm


def current_rss_gib() -> float:
    """Return this process's current resident memory in GiB."""
    with open("/proc/self/status", encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def process_io() -> dict[str, int]:
    """Read Linux per-process I/O counters."""
    counters = {}
    with open("/proc/self/io", encoding="utf-8") as io_file:
        for line in io_file:
            key, value = line.split(":", maxsplit=1)
            counters[key] = int(value)
    return counters



def validate_equivalence(current_dataset, eager_dataset) -> dict[str, float]:
    """Check representative samples before benchmarking speed."""
    if len(current_dataset) != len(eager_dataset):
        raise AssertionError(
            f"Dataset lengths differ: {len(current_dataset)} != {len(eager_dataset)}"
        )

    max_input_difference = 0.0
    max_target_difference = 0.0
    for index in (0, len(current_dataset) // 2, len(current_dataset) - 1):
        current_sample = current_dataset[index]
        eager_sample = eager_dataset[index]
        torch.testing.assert_close(
            eager_sample["input"], current_sample["input"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            eager_sample["target"], current_sample["target"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            eager_sample["start_prediction_month"],
            current_sample["start_prediction_month"],
        )
        max_input_difference = max(
            max_input_difference,
            float((eager_sample["input"] - current_sample["input"]).abs().max()),
        )
        max_target_difference = max(
            max_target_difference,
            float((eager_sample["target"] - current_sample["target"]).abs().max()),
        )
    return {
        "max_input_abs_difference": max_input_difference,
        "max_target_abs_difference": max_target_difference,
    }


def make_loader(dataset, config, *, workers: int, prefetch: bool):
    generator = torch.Generator().manual_seed(1234)
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "generator": generator,
        "num_workers": workers,
        "pin_memory": prefetch,
    }
    if workers:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context="fork",
        )
    return DataLoader(dataset, **kwargs)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def benchmark_arm(
    name,
    loader,
    iterator,
    config,
    initial_state,
    loss_fn,
    device,
    *,
    warmup_steps,
    timed_steps,
    non_blocking,
):
    model = UNetRes3(
        in_channels=util_cesm.get_num_input_channels(config.input_config),
        out_channels=util_cesm.get_num_output_channels(
            config.max_lead_months, config.target_config
        ),
        predict_anomalies=config.target_config["predict_anom"],
        **config.model_args,
    ).to(device)
    model.load_state_dict(initial_state)
    optimizer = build_optimizer(config, model)
    model.train()

    if iterator is None:
        iterator = iter(loader)

    def update(batch):
        inputs = batch["input"].to(device, non_blocking=non_blocking)
        targets = batch["target"].to(device, non_blocking=non_blocking)
        target_months = batch["start_prediction_month"][:, :, 1].to(
            device, non_blocking=non_blocking
        )
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets, target_months)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    for _ in range(warmup_steps):
        update(next(iterator))

    torch.cuda.reset_peak_memory_stats(device)
    io_before = process_io()
    batch_wait_seconds = []
    update_seconds = []
    total_started = time.perf_counter()
    last_loss = float("nan")
    for _ in range(timed_steps):
        step_started = time.perf_counter()
        batch = next(iterator)
        batch_ready = time.perf_counter()
        last_loss = update(batch)
        step_finished = time.perf_counter()
        batch_wait_seconds.append(batch_ready - step_started)
        update_seconds.append(step_finished - batch_ready)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - total_started
    io_after = process_io()

    result = {
        "name": name,
        "steps": timed_steps,
        "samples_per_second": timed_steps * config.batch_size / elapsed,
        "seconds_per_step": elapsed / timed_steps,
        "batch_wait_seconds": summarize(batch_wait_seconds),
        "gpu_update_seconds": summarize(update_seconds),
        "last_loss": last_loss,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "parent_process_io_delta": {
            key: io_after.get(key, 0) - io_before.get(key, 0)
            for key in ("rchar", "read_bytes", "syscr")
        },
    }

    del optimizer, model, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp1_inputs:input4a")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a GPU")

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = load_config(args.config)
    if args.steps + args.warmup_steps > (
        len(config.data_split["train"]) * len(config.data_split["time_range"])
        // config.batch_size
    ):
        raise ValueError("Requested more benchmark steps than one epoch contains")

    eager_started = time.perf_counter()
    eager_dataset = EagerDynamicCESMDataset("train", config)
    eager_initialization_seconds = time.perf_counter() - eager_started

    current_started = time.perf_counter()
    current_dataset = CESM_Dataset("train", config)
    current_initialization_seconds = time.perf_counter() - current_started
    equivalence = validate_equivalence(current_dataset, eager_dataset)

    report = {
        "config": args.config,
        "batch_size": config.batch_size,
        "dataset_samples": len(eager_dataset),
        "eager_initialization_seconds": eager_initialization_seconds,
        "eager_variable_load_seconds": eager_dataset.store.load_seconds,
        "eager_resident_data_gib": eager_dataset.store.resident_data_gib,
        "process_rss_after_eager_load_gib": current_rss_gib(),
        "current_initialization_seconds": current_initialization_seconds,
        "equivalence": equivalence,
        "arms": [],
    }
    print("INITIALIZATION " + json.dumps(report, indent=2), flush=True)

    prefetched_loader = make_loader(
        eager_dataset, config, workers=args.workers, prefetch=True
    )
    # Fork workers while the parent contains only CPU state. The resident NumPy
    # arrays are then shared copy-on-write rather than serialized or reloaded.
    prefetched_iterator = iter(prefetched_loader)

    torch.manual_seed(0)
    template_model = UNetRes3(
        in_channels=util_cesm.get_num_input_channels(config.input_config),
        out_channels=util_cesm.get_num_output_channels(
            config.max_lead_months, config.target_config
        ),
        predict_anomalies=config.target_config["predict_anom"],
        **config.model_args,
    )
    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in template_model.state_dict().items()
    }
    del template_model

    device = torch.device("cuda")
    normalized_dir = os.path.join(
        config_cesm.PROCESSED_DATA_DIRECTORY,
        "normalized_inputs",
        config.data_split["name"],
    )
    with open(os.path.join(normalized_dir, "month_weights.pkl"), "rb") as file:
        month_weights = pickle.load(file)
    loss_fn = WeightedMSELoss(
        device, util_cesm.calculate_area_weights(), month_weights
    )
    report["gpu"] = torch.cuda.get_device_name(device)

    arms = [
        (
            "eager_dynamic_prefetch",
            prefetched_loader,
            prefetched_iterator,
            True,
        ),
        (
            "eager_dynamic_single_process",
            make_loader(eager_dataset, config, workers=0, prefetch=False),
            None,
            False,
        ),
        (
            "current_precomputed_pairs",
            make_loader(current_dataset, config, workers=0, prefetch=False),
            None,
            False,
        ),
    ]
    for name, loader, iterator, non_blocking in arms:
        print(f"Starting {name}", flush=True)
        try:
            result = benchmark_arm(
                name,
                loader,
                iterator,
                config,
                initial_state,
                loss_fn,
                device,
                warmup_steps=args.warmup_steps,
                timed_steps=args.steps,
                non_blocking=non_blocking,
            )
        except Exception as exc:
            result = {"name": name, "error": repr(exc)}
        report["arms"].append(result)
        print("RESULT " + json.dumps(result, indent=2), flush=True)

    print("FINAL_REPORT " + json.dumps(report, indent=2), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
