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

from sarc.client.job import JobStatistics, Statistics, count_jobs, get_jobs
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
    # sparsity_pct = df.isna().mean().sort_values(ascending=False)
    # print("Sparsity of each column:")
    # print(sparsity_pct.to_string())

    # Required columns. If a job has any of these missing, we drop that job.
    # We don't include the required fields of SlurmJob here, they already can't be None.
    required_full_columns = [
        # "elapsed_time",
        "requested.mem",
        "requested.node",
        "requested.cpu",
    ]

    df["elapsed_time"] = pd.to_timedelta(df["elapsed_time"], unit="s")

    df = df[df["elapsed_time"] > timedelta(minutes=1)]
    df = df[df["elapsed_time"] < timedelta(days=30)]

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    df.loc[:, "requested.gres_gpu"] = df["requested.gres_gpu"].fillna(0)
    df.loc[:, "allocated.gres_gpu"] = df["allocated.gres_gpu"].fillna(0)

    # df[df["requested.gres_gpu"] > 0 & df["stored_statistics.gpu_utilization.mean"].isna()] =
    # df["stored_statistics.gpu_utilization.mean"] = df["stored_statistics.gpu_utilization.mean"].fillna(

    df = (
        df
        # drop columns that have only NANs.
        # .dropna(axis="columns", how="all")
        # Drop rows that have nas in any of the absolutely required columns.
        .dropna(axis="index", how="any", subset=required_full_columns)
        # Drop rows that have nas in all of the partially required columns.
        # .dropna(axis="index", how="all", subset=partial_columns)
    )

    # Replace outliers with NAs for the jobs with insanely high GPU utilization. (H100 GPU bug).
    stored_gpu_stats_columns = [
        c for c in df.columns if c.startswith("stored_statistics.") and "gpu" in c
    ]
    columns_with_potential_outliers = [
        c
        for c in df.columns
        if c.startswith(
            (
                "stored_statistics.gpu_utilization",
                "stored_statistics.gpu_power.",
                "stored_statistics.gpu_memory.",
            )
        )
    ]
    outliers: pd.Series | None = None
    for col in columns_with_potential_outliers:
        if "utilization" in col:
            new_outliers = (utilization := df[col]).notna() & (
                (utilization < 0) | (utilization > 1)
            )
        elif "gpu_power" in col:
            new_outliers = (power := df[col]).notna() & (power > 10e5)
        elif "gpu_memory" in col:
            new_outliers = (mem_util := df[col]).notna() & (
                (mem_util < 0) | (mem_util > 1)
            )
        else:
            raise NotImplementedError(col)
        if outliers is None:
            outliers = new_outliers
        else:
            outliers |= new_outliers
    assert outliers is not None
    n_outliers = outliers.sum()
    logger.info(
        f"GPU utilization metrics had {n_outliers} outliers "
        f"({n_outliers / len(outliers):.2%})."
    )
    # Replace the outlier metrics with NA.
    # TODO: They will then be filled in with the mean of the other correct jobs for that user on that cluster.
    df.loc[
        outliers,
        stored_gpu_stats_columns,
    ] = pd.NA

    across_clusters_mean_job_stats = get_mean_job_stats(df)

    for cluster_name in df["cluster_name"].unique():
        cluster_mask = df["cluster_name"] == cluster_name
        cluster_mean_job_statistics = get_mean_job_stats(df[cluster_mask])

        # TODO: fill in with the cluster-local mean when available, otherwise use the "across-cluster" mean.
        stats_to_use = (
            cluster_mean_job_statistics.dict() | across_clusters_mean_job_stats.dict()
        )

        # idea: For all jobs that have a GPU allocated, but the `stored_statistics.gpu_utilization` is missing,
        # fill in the missing values with the mean of the available stored statistics (*for that same cluster*).
        missing_gpu_stats = (
            cluster_mask
            & (df["allocated.gres_gpu"] > 0)
            & df[stored_gpu_stats_columns].isna().any(axis=1)
        )

        # Assign the average of the available statistics to the missing ones.
        # TODO: This assignment will probably fail. We might need to flatten the JobStatistics to an array first.
        df[missing_gpu_stats] = stats_to_use

    assert (
        df["stored_statistics.gpu_utilization.mean"]
        .between(0, 1, inclusive="both")
        .all()
    )

    return df


def get_mean_stats(df: pd.DataFrame, metric: str) -> Statistics | None:
    fields = list(Statistics.__fields__.keys())
    columns = [f"stored_statistics.{metric}.{f}" for f in fields]
    # drop any rows where there are some missing values
    data = df.dropna(how="any", subset=columns, axis=0)
    if not len(data):
        logger.warning(
            RuntimeWarning(
                f"Statistics for 'stored_statistics.{metric}' are all missing!"
            )
        )
        return None
    # todo: does it actually make sense to take the mean of these values
    # (std, min, max, q05, etc?) or should I combine them in a smarter way?
    stuff = {
        field: (data[column].max() if field.endswith("max") else data[column].mean())
        for field, column in zip(fields, columns)
    }

    # unused_key = f"stored_statistics.{metric}.unused"
    # unused = stuff.pop(unused_key, 0)
    # if isinstance(unused, float):
    #     assert unused.is_integer()
    return Statistics(**stuff)  # type: ignore


def try_get_mean_stats(df: pd.DataFrame, metric: str):
    try:
        return get_mean_stats(df, metric)
    except KeyError:
        return None


def get_mean_job_stats(df: pd.DataFrame) -> JobStatistics:
    return JobStatistics(
        **{k: try_get_mean_stats(df, k) for k in JobStatistics.__fields__.keys()}
    )


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


def main():
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

    gpu_util_metrics = get_mean_stats(df, "gpu_utilization")
    print(gpu_util_metrics)
    mean_job_stats = get_mean_job_stats(df)
    print(mean_job_stats.gpu_utilization)

    print("GPU hours per job duration")
    print("Mila-cluster:")
    print(compute_gpu_hours_per_duration(df_mila))
    print("DRAC clusters:")
    print(compute_gpu_hours_per_duration(df_drac))
    print("Binned GPU hours")
    print("Mila-cluster:")
    print(compute_jobs_per_gpu_hours(df_mila))
    print("DRAC clusters:")
    print(compute_jobs_per_gpu_hours(df_drac))

    print("GPU hours per gpu job count")
    print("Mila-cluster:")
    print(compute_gpu_hours_per_gpu_count(df_mila))
    print("DRAC clusters:")
    print(compute_gpu_hours_per_gpu_count(df_drac))


if __name__ == "__main__":
    main()
