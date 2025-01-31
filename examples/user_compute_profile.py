import dataclasses
import logging
import os
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeVar

import pandas as pd
import rich.logging
import simple_parsing
from tqdm import tqdm

from sarc.client.job import JobStatistics, count_jobs, get_jobs
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


def get_jobs_dataframe(config: Args) -> pd.DataFrame:
    #  Fetch all jobs from the clusters

    cached_results_file = config.unique_path()
    # Check if the results are already cached.
    if cached_results_file.exists():
        logger.info(f"Loading previous cached results from {cached_results_file}")
        with open(cached_results_file, "rb") as f:
            df = pickle.load(f)
            assert isinstance(df, pd.DataFrame)
            return df
    elif (
        generic_config_path := dataclasses.replace(config, user=None).unique_path()
    ).exists():
        # Can reuse the results for all users and filter out our user:
        logger.info(
            f"Loading previous cached results for all users at {generic_config_path}"
        )
        with open(generic_config_path, "rb") as f:
            generic_df = pickle.load(f)
            assert isinstance(generic_df, pd.DataFrame)
            return generic_df[generic_df["user"] == config.user]

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
    logger.debug(f"Cached results written to {cached_results_file}")
    df.to_pickle(cached_results_file)
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


def filter_df(df: pd.DataFrame, config: Args) -> pd.DataFrame:
    sparsity_pct = df.isna().mean().sort_values(ascending=False)
    print("Sparsity of each column:")
    print(sparsity_pct.to_string())

    # Required columns. If a job has any of these missing, we drop that job.
    # We don't include the required fields of SlurmJob here, they already can't be None.
    required_full_columns = [
        "elapsed_time",
        "requested.mem",
        "requested.node",
        "requested.cpu",
    ]
    # Columns for which we can tolerate some missing values.
    partial_columns = [
        "stored_statistics.gpu_memory.mean",
        # "stored_statistics.gpu_utilization_fp16.mean",
        # "stored_statistics.gpu_utilization_fp16.mean",
        # "stored_statistics.gpu_utilization_fp32.mean",
        "stored_statistics.gpu_sm_occupancy.mean",
    ]

    df = (
        df
        # drop columns that have only NANs.
        .dropna(axis="columns", how="all")
        # Drop rows that have nas in any of the absolutely required columns.
        .dropna(axis="index", how="any", subset=required_full_columns)
        # Drop rows that have nas in all of the partially required columns.
        .dropna(axis="index", how="all", subset=partial_columns)
    )
    # todo: use fillna, get averages, etc etc.
    # TODO: Weird GPU utilization, e.g. jobid 2234959
    for col in df.columns:
        data = df[col]
        if data.isna().all():
            raise NotImplementedError(f"Column {col} is all NANs!")
        if "utilization" in col and 0 <= df[col].dropna().median() <= 1:
            outliers = df[(df[col] < 0) | (df[col] > 1)]["job_id"]
            logger.info(
                f"Clipping {len(outliers)} outliers ({len(outliers) / len(df):.2%}) "
                f"in column {col} (outside of [0-1] range)."
            )
            df[col] = df[col].clip(0, 1)
    assert (
        df["stored_statistics.gpu_utilization.mean"]
        .between(0, 1, inclusive="both")
        .all()
    )
    return df


config = simple_parsing.parse(Args)
_setup_logging(config.verbose)
print(f"Configuration: {config}")
df = get_jobs_dataframe(config)

filtered_df = filter_df(df, config)

print(
    f"Filtered out {len(df) - len(filtered_df)} ({(len(df) - len(filtered_df)) / len(df) * 100:.2f}%) of jobs."
)
df = filtered_df

# Compute the billed and used resource time in seconds
df["billed"] = df["elapsed_time"] * df["allocated.billing"]
df["used"] = df["elapsed_time"] * df["allocated.gres_gpu"]

df_mila = df[df["cluster_name"] == "mila"]
df_drac = df[df["cluster_name"] != "mila"]

print("Number of jobs:")
print("Mila-cluster", df_mila.shape[0])
print("DRAC clusters", df_drac.shape[0])

print("GPU hours:")
print("Mila-cluster", df_mila["used"].fillna(0).sum() / (3600))
print("DRAC clusters", df_drac["used"].fillna(0).sum() / (3600))

from sarc.client.job import Statistics


def get_mean_stats(df: pd.DataFrame, metric: str) -> Statistics:
    fields = list(Statistics.__fields__.keys())
    columns = [f"stored_statistics.{metric}.{f}" for f in fields]
    # drop any rows where there are some missing values
    data = df.dropna(how="any", subset=columns, axis=0)
    if not len(data):
        raise RuntimeError(
            f"Not a single row in the dataframe has no NANs in the stored statistics for metric {metric}!"
        )
    # todo: does it actually make sense to take the mean of these values
    # (std, min, max, q05, etc?) or should I combine them in a smarter way?
    stuff = {field: data[column].mean() for field, column in zip(fields, columns)}

    # unused_key = f"stored_statistics.{metric}.unused"
    # unused = stuff.pop(unused_key, 0)
    # if isinstance(unused, float):
    #     assert unused.is_integer()
    return Statistics(**stuff)  # type: ignore


def get_mean_job_stats(df: pd.DataFrame) -> JobStatistics:
    return JobStatistics(
        **{k: get_mean_stats(df, k) for k in JobStatistics.__fields__.keys()}
    )


gpu_util_metrics = get_mean_stats(df, "gpu_utilization")
print(gpu_util_metrics)
mean_job_stats = get_mean_job_stats(df)
print(mean_job_stats)


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
        condition = df["allocated.gres_gpu"] >= min_time
        if max_time is not None:
            condition *= df["allocated.gres_gpu"] < max_time
        categories_df[key] = condition.astype(bool) * df["used"]

    return categories_df.sum() / df["used"].sum()


print("GPU hours per gpu job count")
print("Mila-cluster:")
print(compute_gpu_hours_per_gpu_count(df_mila))
print("DRAC clusters:")
print(compute_gpu_hours_per_gpu_count(df_drac))
