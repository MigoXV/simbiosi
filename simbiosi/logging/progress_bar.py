# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

import json
import logging
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from numbers import Number
from typing import Optional

import matplotlib
import numpy as np
import seaborn as sns
import torch
from fairseq.logging.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.logging.progress_bar import (
    JsonProgressBar,
    NoopProgressBar,
    SimpleProgressBar,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def progress_bar(
    iterator,
    log_format: Optional[str] = None,
    log_interval: int = 100,
    log_file: Optional[str] = None,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    aim_repo: Optional[str] = None,
    aim_run_hash: Optional[str] = None,
    aim_param_checkpoint_dir: Optional[str] = None,
    tensorboard_logdir: Optional[str] = None,
    default_log_format: str = "tqdm",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    azureml_logging: Optional[bool] = False,
):
    if log_format is None:
        log_format = default_log_format
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file)
        logger.addHandler(handler)

    if log_format == "tqdm" and not sys.stderr.isatty():
        log_format = "simple"

    if log_format == "json":
        bar = JsonProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == "none":
        bar = NoopProgressBar(iterator, epoch, prefix)
    elif log_format == "simple":
        bar = SimpleProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == "tqdm":
        bar = TqdmProgressBar(iterator, epoch, prefix)
    else:
        raise ValueError("Unknown log format: {}".format(log_format))

    if aim_repo:
        bar = AimProgressBarWrapper(
            bar,
            aim_repo=aim_repo,
            aim_run_hash=aim_run_hash,
            aim_param_checkpoint_dir=aim_param_checkpoint_dir,
        )

    if wandb_project:
        bar = WandBProgressBarWrapper(bar, wandb_project, run_name=wandb_run_name)

    return bar


def build_progress_bar(
    args,
    iterator,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    default: str = "tqdm",
    no_progress_bar: str = "none",
):
    """Legacy wrapper that takes an argparse.Namespace."""
    if getattr(args, "no_progress_bar", False):
        default = no_progress_bar
    if getattr(args, "distributed_rank", 0) == 0:
        tensorboard_logdir = getattr(args, "tensorboard_logdir", None)
    else:
        tensorboard_logdir = None
    return progress_bar(
        iterator,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch,
        prefix=prefix,
        tensorboard_logdir=tensorboard_logdir,
        default_log_format=default,
    )


def format_stat(stat):
    if isinstance(stat, Number):
        stat = "{:g}".format(stat)
    elif isinstance(stat, AverageMeter):
        stat = "{:.3f}".format(stat.avg)
    elif isinstance(stat, TimeMeter):
        stat = "{:g}".format(round(stat.avg))
    elif isinstance(stat, StopwatchMeter):
        stat = "{:g}".format(round(stat.sum))
    elif torch.is_tensor(stat):
        stat = stat.tolist()
    return stat


class BaseProgressBar(object):
    """Abstract class for progress bars."""

    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable
        self.n = getattr(iterable, "n", 0)
        self.epoch = epoch
        self.prefix = ""
        if epoch is not None:
            self.prefix += "epoch {:03d}".format(epoch)
        if prefix is not None:
            self.prefix += (" | " if self.prefix != "" else "") + prefix

    def __len__(self):
        return len(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def update_config(self, config):
        """Log latest configuration."""
        pass

    def _str_commas(self, stats):
        return ", ".join(key + "=" + stats[key].strip() for key in stats.keys())

    def _str_pipes(self, stats):
        return " | ".join(key + " " + stats[key].strip() for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name


class TqdmProgressBar(BaseProgressBar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)
        from tqdm import tqdm

        self.tqdm = tqdm(
            iterable,
            self.prefix,
            leave=False,
            disable=(logger.getEffectiveLevel() > logging.INFO),
        )

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""

        postfix = self._str_pipes(self._format_stats(stats))
        with rename_logger(logger, tag):
            logger.info("{} | {}".format(self.prefix, postfix))

    def _format_stats(self, stats):
        postfix = OrderedDict(
            (k, v) for k, v in stats.items() if not hasattr(v, "shape")
        )
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


try:
    import functools

    from aim import Repo as AimRepo

    @functools.lru_cache()
    def get_aim_run(repo, run_hash):
        from aim import Run

        return Run(run_hash=run_hash, repo=repo)

except ImportError:
    get_aim_run = None
    AimRepo = None


class AimProgressBarWrapper(BaseProgressBar):
    """Log to Aim."""

    def __init__(self, wrapped_bar, aim_repo, aim_run_hash, aim_param_checkpoint_dir):
        self.wrapped_bar = wrapped_bar

        if get_aim_run is None:
            self.run = None
            logger.warning("Aim not found, please install with: pip install aim")
        else:
            logger.info(f"Storing logs at Aim repo: {aim_repo}")

            if not aim_run_hash:
                # Find run based on save_dir parameter
                query = f"run.checkpoint.save_dir == '{aim_param_checkpoint_dir}'"
                try:
                    runs_generator = AimRepo(aim_repo).query_runs(query)
                    run = next(runs_generator.iter_runs())
                    aim_run_hash = run.run.hash
                except Exception:
                    pass

            if aim_run_hash:
                logger.info(f"Appending to run: {aim_run_hash}")

            self.run = get_aim_run(aim_repo, aim_run_hash)

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to Aim."""
        self._log_to_aim(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_aim(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        if self.run is not None:
            for key in config:
                self.run.set(key, config[key], strict=False)
        self.wrapped_bar.update_config(config)

    def _log_to_aim(self, stats, tag=None, step=None):
        if self.run is None:
            return

        if step is None:
            step = stats["num_updates"]

        if "train" in tag:
            context = {"tag": tag, "subset": "train"}
        elif "val" in tag:
            context = {"tag": tag, "subset": "val"}
        else:
            context = {"tag": tag}

        for key in stats.keys() - {"num_updates"}:
            self.run.track(stats[key], name=key, step=step, context=context)


try:
    from wandb.plot.custom_chart import CustomChart, plot_table

    import wandb
except ImportError:
    wandb = None


class WandBProgressBarWrapper(BaseProgressBar):
    """Log to Weights & Biases."""

    def __init__(self, wrapped_bar, wandb_project, run_name=None):
        self.wrapped_bar = wrapped_bar
        if wandb is None:
            logger.warning("wandb not found, pip install wandb")
            return

        # reinit=False to ensure if wandb.init() is called multiple times
        # within one process it still references the same run
        wandb.init(project=wandb_project, reinit=False, name=run_name)

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        if wandb is not None:
            wandb.config.update(config)
        self.wrapped_bar.update_config(config)

    def _log_to_wandb(self, stats, tag=None, step=None):
        if wandb is None:
            return
        if step is None:
            step = stats["num_updates"]

        prefix = "" if tag is None else tag + "/"

        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                wandb.log({prefix + key: stats[key].val}, step=step)
            elif isinstance(stats[key], Number):
                wandb.log({prefix + key: stats[key]}, step=step)
            # elif isinstance(stats[key], np.ndarray) or isinstance(
            #     stats[key], torch.Tensor
            # ):
            #     time0 = time.time()
            #     self.mat_plot(stats[key], prefix + key, step)
            #     wandb.log(
            #         {prefix + "plot_time": time.time() - time0},
            #         step=step,
            #     )

    def mat_plot(self, matrix, title, step):
        matrix = np.array(matrix).astype(int)
        rows, cols = matrix.shape

        x_labels = [str(i) for i in range(cols)]
        y_labels = [str(i) for i in range(rows)]

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=x_labels,
            yticklabels=y_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        # Set the x-axis labels on the top and y-axis labels on the left
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("left")
        ax.tick_params(
            left=True,
            top=True,
            labelleft=True,
            labeltop=True,
            bottom=False,  # 不显示底部刻度
            labelbottom=False,  # 不显示底部标签
        )
        plt.tight_layout(pad=0)
        wandb.log({title: wandb.Image(fig)}, step=step)

        plt.close(fig)
