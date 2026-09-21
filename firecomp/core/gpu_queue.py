"""
core/gpu_queue.py — Multi-GPU subprocess queue.

Dispatches a list of configs across available GPUs as independent
subprocesses. Each config is serialized to JSON and launched via
``python -m <module> <command> --config <path>``, pinned to one GPU with
CUDA_VISIBLE_DEVICES.  The command name defaults to ``train`` but can be
any Cli subcommand.

Used by grid_search.py and transfer.py — anywhere you have N configs
and M GPUs and want to keep all GPUs busy.

Usage:
    from firecomp.core.gpu_queue import GpuQueue

    results = GpuQueue.run(
        configs=my_configs,
        out_dir=Path("data/runs/experiment"),
        train_module="firecomp.next_day.implementations.dl_2d",
        num_gpus=4,
    )
"""

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# GpuQueue — static-method namespace
# ---------------------------------------------------------------------------

class GpuQueue:
    """
    Multi-GPU subprocess queue for training runs.

    Namespace class (static methods only). Each config is a dataclass with
    at least ``tag`` and ``run_dir`` fields. The queue serializes it to JSON,
    launches ``python -m <module> <command> --config <json>`` pinned to a
    GPU via CUDA_VISIBLE_DEVICES, and reads the ``result.json`` each process
    writes on completion.
    """

    MAX_RETRIES = 2

    @staticmethod
    def run(
        configs: list,
        out_dir: Path,
        train_module: str,
        num_gpus: int = 1,
        label: str = "Experiment",
        command: str = "train",
    ) -> list[dict]:
        """
        Dispatch configs as subprocesses across GPUs, return results.

        Failed runs are retried up to MAX_RETRIES times before being
        recorded as failures.

        Args:
            configs:      list of dataclass configs (must have .tag, .run_dir).
            out_dir:      root directory for logs.
            train_module: dotted module path (e.g. "firecomp.next_day.implementations.dl_2d").
            num_gpus:     GPUs for queue dispatch (1 = sequential).
            label:        human label for progress output.
            command:      Cli subcommand name (default "train").

        Returns:
            list of dicts — one per completed run (from result.json), plus
            failure dicts for crashed runs.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = len(configs)

        pending = list(configs)              # queue
        active  = {}                         # gpu_id -> (Popen, cfg, log_fh, attempt)
        free    = list(range(num_gpus))      # available GPU ids
        retries: dict[str, int] = {}         # tag -> attempt count
        rows    = []

        print(f"\n{label}: {total} configs, {num_gpus} GPU(s)")
        print(f"Output dir:  {out_dir}")
        print(f"Logs:        {out_dir}/<tag>.log\n")

        # Fill GPUs with initial batch.
        while pending and free:
            GpuQueue._dispatch(pending, free, active, out_dir,
                               train_module, command, retries)

        # Poll until everything is done.
        while active:
            for gpu_id in list(active):
                proc, cfg, log_f, attempt = active[gpu_id]
                ret = proc.poll()
                if ret is None:
                    continue

                # Process finished — close log.
                log_f.close()
                del active[gpu_id]
                free.append(gpu_id)

                if ret == 0:
                    row = GpuQueue._read_result(cfg)
                    rows.append(row)
                    done = len(rows)
                    f1 = row.get("val_f1", 0)
                    print(f"[{done}/{total}]  DONE   {cfg.tag:<45s}  "
                          f"val_F1={f1:.4f}  GPU {gpu_id}")
                elif attempt < GpuQueue.MAX_RETRIES:
                    # Retry — push back to front of queue.
                    pending.insert(0, cfg)
                    print(f"  RETRY  {cfg.tag:<45s}  "
                          f"exit={ret}  attempt {attempt+1}/{GpuQueue.MAX_RETRIES}  "
                          f"see {out_dir / f'{cfg.tag}.log'}")
                else:
                    rows.append({"tag": cfg.tag, "status": "failed",
                                 "exit_code": ret, "val_f1": 0.0})
                    done = len(rows)
                    print(f"[{done}/{total}]  FAIL   {cfg.tag:<45s}  "
                          f"exit={ret}  GPU {gpu_id}  "
                          f"(gave up after {attempt} retries)  "
                          f"see {out_dir / f'{cfg.tag}.log'}")

                # Dispatch next from queue.
                if pending and free:
                    GpuQueue._dispatch(pending, free, active, out_dir,
                                       train_module, command, retries)

            time.sleep(2)

        return rows

    @staticmethod
    def _dispatch(pending, free, active, out_dir, train_module, command,
                  retries=None):
        """Pop next config from queue, launch on a free GPU."""
        cfg    = pending.pop(0)
        gpu_id = free.pop(0)
        if retries is None:
            retries = {}

        attempt = retries.get(cfg.tag, 0) + 1
        retries[cfg.tag] = attempt

        # Write config JSON for this run.
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)

        # Launch subprocess.
        cmd = [
            sys.executable, "-m", train_module,
            command, "--config", str(config_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Append on retry so we keep the original log.
        log_path = out_dir / f"{cfg.tag}.log"
        mode = "a" if attempt > 1 else "w"
        log_f = open(log_path, mode)
        if attempt > 1:
            log_f.write(f"\n--- RETRY attempt {attempt} ---\n")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f,
                                stderr=subprocess.STDOUT)

        active[gpu_id] = (proc, cfg, log_f, attempt)
        retry_tag = f" (retry {attempt})" if attempt > 1 else ""
        print(f"  START  {cfg.tag:<45s}  GPU {gpu_id}  pid {proc.pid}"
              f"{retry_tag}")

    @staticmethod
    def _read_result(cfg) -> dict:
        """Read result.json from a completed run directory."""
        result_path = Path(cfg.run_dir) / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                return json.load(f)
        # Subprocess finished OK but no result.json — shouldn't happen.
        return {"tag": cfg.tag, "status": "no_result", "val_f1": 0.0}

    @staticmethod
    def resolve_num_gpus(n: int) -> int:
        """Resolve GPU count: 0 means auto-detect.

        Uses nvidia-smi instead of torch.cuda.device_count() to avoid
        initializing CUDA in the parent process.  If CUDA is initialized
        here, subprocesses inherit a stale context and
        CUDA_VISIBLE_DEVICES changes are ignored — causing silent
        fallback to CPU.
        """
        if n >= 1:
            return n
        import subprocess as sp
        try:
            out = sp.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=10,
            )
            count = len(out.strip().splitlines())
        except (FileNotFoundError, sp.SubprocessError):
            count = 0
        print(f"Auto-detected {count} GPU(s)")
        return max(count, 1)
