import dataclasses
import logging
import os
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd
import rich.logging
import simple_parsing
from tqdm import tqdm

from sarc.client.job import count_jobs, get_jobs
from sarc.config import MTL

logger = logging.getLogger(__name__)
Out = TypeVar("Out")
T = TypeVar("T")


def midnight(dt: datetime) -> datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class Args:
    """Configuration options for this script."""

    start: datetime = simple_parsing.field(
        default=(midnight(datetime.now(tz=MTL)) - timedelta(days=30)),
        type=datetime.fromisoformat,
    )
    """ Start date. """

    end: datetime = simple_parsing.field(
        default=midnight(datetime.now(tz=MTL)),
        type=datetime.fromisoformat,
    )
    """ End date. """

    user: str | None = None
    """ Which user to query information for. Leave blank to get a global compute profile."""

    # clusters: list[str] = dataclasses.field(default_factory=list)
    # """ Which clusters to query information for. Leave blank to get data from all clusters."""

    cache_dir: Path = dataclasses.field(
        default=Path(os.environ.get("SCRATCH", tempfile.gettempdir())), hash=False
    )
    """ Directory where temporary files will be stored."""

    verbose: int = simple_parsing.field(
        alias=["-v", "--verbose"], action="count", default=0, hash=False
    )

    def unique_path(self, label: str = "", extension: str = ".pkl") -> Path:
        user_portion = self.user or "all"
        # cluster_portion = "-".join(self.clusters) if self.clusters else "all"
        start_portion = (
            self.start.strftime("%Y-%m-%d")
            if self.start == midnight(self.start)
            else str(self.start).replace(" ", "_")
        )
        end_portion = (
            self.end.strftime("%Y-%m-%d")
            if self.end == midnight(self.end)
            else str(self.end).replace(" ", "_")
        )
        return (
            self.cache_dir
            / f"compute_profile-{user_portion}-{start_portion}-{end_portion}-{label}"
        ).with_suffix(extension)


def cached(label: str = ""):
    def _wrapper(fn: Callable[[Args], Out]):
        def _wrapped(config: Args) -> Out:
            cached_results_file = config.unique_path(label=label)
            # Check if the results are already cached.
            if cached_results_file.exists():
                logger.info(
                    f"Loading previous cached results of running `{fn.__name__}({config})` "
                    f"from {cached_results_file}"
                )
                with open(cached_results_file, "rb") as f:
                    return pickle.load(f)
            else:
                logger.debug(
                    f"Previous results not found at path {cached_results_file}"
                )
            # run the function.
            result = fn(config)

            logger.debug(f"Cached results written to {cached_results_file}")
            with open(cached_results_file, "wb") as f:
                pickle.dump(result, f)
            return result

        return _wrapped

    return _wrapper


@cached(label="jobs_df")
def get_jobs_dataframe(config: Args) -> pd.DataFrame:
    #  Fetch all jobs from the clusters
    # Precompute the total number of jobs to display a progress bar since get_jobs is a generator.
    # Fetch all jobs from the clusters
    total = count_jobs(user=config.user, start=config.start, end=config.end)
    job_dicts = [
        job.dict()
        for job in tqdm(
            get_jobs(user=config.user, start=config.start, end=config.end),
            total=total,
            desc="Gathering jobs",
            unit="jobs",
        )
    ]
    df = pd.json_normalize(job_dicts)
    df = df.convert_dtypes()
    assert isinstance(df, pd.DataFrame)
    return df


def _setup_logging(verbose: int):
    logging.basicConfig(
        handlers=[rich.logging.RichHandler()],
        format="%(message)s",
        level=logging.ERROR,
    )

    if verbose == 0:
        logger.setLevel("WARNING")
    elif verbose == 1:
        logger.setLevel("INFO")
    else:
        logger.setLevel("DEBUG")


config = simple_parsing.parse(Args)
_setup_logging(config.verbose)
print(f"Configuration: {config}")
df = get_jobs_dataframe(config)

all_missing_columns = df.columns[(df.isna().all(axis=0))]
full_columns = df.columns[df.notna().all(axis=0)]
partial_columns = df.columns[df.isna().any(axis=0)]
print(f"{all_missing_columns=}")
print(f"{full_columns=}")
print(f"{partial_columns=}")
# Compute the billed and used resource time in seconds
df["billed"] = df["elapsed_time"] * df["billing"]
df["used"] = df["elapsed_time"] * df["gres_gpu"]

df_mila = df[df["cluster_name"] == "mila"]
df_drac = df[df["cluster_name"] != "mila"]

print("Number of jobs:")
print("Mila-cluster", df_mila.shape[0])
print("DRAC clusters", df_drac.shape[0])

print("GPU hours:")
print("Mila-cluster", df_mila["used"].sum() / (3600))
print("DRAC clusters", df_drac["used"].sum() / (3600))


def compute_gpu_hours_per_duration(df: pd.DataFrame):
    categories = {
        "< 1hour": (0, 3600),
        "1-24 hours": (3600, 24 * 3600),
        "1-28 days": (24 * 3600, 28 * 24 * 3600),
        ">= 28 days": (28 * 24 * 3600, None),
    }
    categories_df = pd.DataFrame(columns=list(categories.keys()))
    for key, (min_time, max_time) in categories.items():
        condition = df["elapsed_time"] >= min_time
        if max_time is not None:
            condition *= df["elapsed_time"] < max_time
        categories_df[key] = condition.astype(bool) * df["used"]

    return categories_df[list(categories_df.keys())].sum() / df["used"].sum()


print("GPU hours per job duration")
print("Mila-cluster:")
print(compute_gpu_hours_per_duration(df_mila))
print("DRAC clusters:")
print(compute_gpu_hours_per_duration(df_drac))


def compute_jobs_per_gpu_hours(df: pd.DataFrame):
    categories = {
        "< 1 GPUhour": (0, 3600),
        "1-24 GPUhours": (3600, 24 * 3600),
        "1-28 GPUdays": (24 * 3600, 28 * 24 * 3600),
        ">= 28 GPUdays": (28 * 24 * 3600, None),
    }
    categories_df = pd.DataFrame(columns=list(categories.keys()))
    for key, (min_time, max_time) in categories.items():
        condition = df["used"] >= min_time
        if max_time is not None:
            condition *= df["used"] < max_time
        categories_df[key] = condition.astype(bool) * df["used"]

    return categories_df.sum() / df["used"].sum()


print("Binned GPU hours")
print("Mila-cluster:")
print(compute_jobs_per_gpu_hours(df_mila))
print("DRAC clusters:")
print(compute_jobs_per_gpu_hours(df_drac))


def compute_gpu_hours_per_gpu_count(df: pd.DataFrame):
    categories = {
        "1 GPU": (1, 2),
        "2-4 GPUs": (2, 5),
        "5-8 GPUs": (5, 9),
        "9-32 GPUs": (9, 33),
        ">= 33 PUdays": (33, None),
    }
    categories_df = pd.DataFrame(columns=list(categories.keys()))
    for key, (min_time, max_time) in categories.items():
        condition = df["gres_gpu"] >= min_time
        if max_time is not None:
            condition *= df["gres_gpu"] < max_time
        categories_df[key] = condition.astype(bool) * df["used"]

    return categories_df.sum() / df["used"].sum()


print("GPU hours per gpu job count")
print("Mila-cluster:")
print(compute_gpu_hours_per_gpu_count(df_mila))
print("DRAC clusters:")
print(compute_gpu_hours_per_gpu_count(df_drac))
