import dataclasses
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import rich
import rich.logging
import simple_parsing
import yaml

from sarc.client.job import JobStatistics
from sarc.client.series import (
    compute_cost_and_waste,
    load_job_series,
)
from sarc.config import MTL, ClusterConfig
from sarc.jobs.series import (
    update_cluster_job_series_rgu,
)

logger = logging.getLogger(__name__)
pd.options.display.max_colwidth = 300
pd.options.display.max_rows = 1000


gpu_name_mapping = {
    "gpu:tesla_v100-sxm2-16gb:4": "v100-16gb",
    "p100": "p100-12gb",
    "gpu:p100:4": "p100-12gb",
    "gpu:p100:2": "p100-12gb",
    "gpu:p100l:4": "p100-16gb",
    "v100": "v100-16gb",
    "gpu:v100:6": "v100-16gb",
    "gpu:v100:8": "v100-16gb",
    "gpu:v100l:4": "v100-32gb",
    "gpu:t4:4": "t4-16gb",
    "4g.20gb": "a100-40gb-4g.20gb",
    "3g.20gb": "a100-40gb-3g.20gb",
    "a100_4g.20gb": "a100-40gb-4g.20gb",
    "gpu:a100_4g.20gb:4": "a100-40gb-4g.20gb",
    "a100_3g.20gb": "a100-40gb-3g.20gb",
    "gpu:a100_3g.20gb:4": "a100-40gb-3g.20gb",
    "a100": "a100-40gb",
    "gpu:a100:4": "a100-40gb",
    "gpu:a100:8": "a100-40gb",
    "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4": "a100-mixup",
    "gpu:a100l:4": "a100-80gb",
    "gpu:a100l:8": "a100-80gb",
    "gpu:a6000:8": "a6000",
    "gpu:rtx8000:8": "rtx8000-48gb",
    "gpu:h100:8": "h100-80gb",
    "NVIDIA A100-SXM4-40GB": "a100-40gb",
    "NVIDIA A100-80GB PCIe": "a100-80gb",
    "NVIDIA A100 80GB PCIe": "a100-80gb",
    "NVIDIA A100-SXM4-80GB": "a100-80gb",
    "NVIDIA H100 80GB HBM3": "h100-80gb",
    "NVIDIA L40S": "l40s",
    "NVIDIA RTX A6000": "a6000",
    "gpu:l40s:4": "l40s",
    "a100_2g.10gb": "a100-40gb-2g.10gb",
    "2g.10gb": "a100-40gb-2g.10gb",
    "2g.20gb": "a100-80gb-2g.20gb",
    "3g.40gb": "a100-80gb-3g.40gb",
    "4g.40gb": "a100-80gb-4g.40gb",
    "Tesla V100-SXM2-16GB": "v100-16gb",
    "Tesla V100-SXM2-32GB": "v100-32gb",
    "Tesla V100-SXM2-32GB-LS": "v100-32gb",
    "NVIDIA V100-SXM2-32GB-LS": "v100-32gb",
    "Quadro RTX 8000": "rtx8000-48gb",  # Dummy
    "gpu:a5000:4": "a5000-24gb",
}

gpu_ram = {
    "p100-12gb": 12,
    "p100-16gb": 16,
    "t4-16gb": 16,
    "v100-16gb": 16,
    "v100-32gb": 32,
    "a100-40gb": 40,
    "a100-mixup": 40,
    "a100-40gb-2g.10gb": 10,
    "a100-40gb-4g.20gb": 20,
    "a100-40gb-3g.20gb": 20,
    "rtx8000-48gb": 48,
    "a5000-24gb": 24,
    "a6000": 48,  # Dummy
    "a100-80gb-4g.40gb": 40,
    "a100-80gb-3g.40gb": 40,
    "a100-80gb-2g.20gb": 20,
    "a100-80gb": 80,
    "h100-80gb": 80,
    "l40s": 48,
}

rgus = {
    "p100-12gb": 1,
    "p100-16gb": 1.1,
    "t4-16gb": 1.3,
    "v100-16gb": 2.2,
    "v100-32gb": 2.6,
    "a100-40gb": 4,
    "a100-mixup": 4,
    "a100-40gb-4g.20gb": 2.3,
    "a100-40gb-3g.20gb": 2,
    "a100-40gb-2g.10gb": 1,
    "rtx8000-48gb": 2.81,  # dummy
    "a5000-24gb": 2.6,  # dummy
    "a100-80gb": 4.8,
    "a100-80gb-2g.20gb": 4.8 * 2 / 7,
    "a100-80gb-3g.40gb": 4.8 * 3 / 7,
    "a100-80gb-4g.40gb": 4.8 * 4 / 7,
    "a6000": 4.93,
    "h100-80gb": 12.2,
    "l40s": 10.4,
}


def midnight(dt: datetime) -> datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class Options:
    """Configuration options for this script."""

    start: datetime = simple_parsing.field(
        default=(midnight(datetime.now(tz=MTL)) - timedelta(days=30)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ Start date. """

    end: datetime = simple_parsing.field(
        default=midnight(datetime.now(tz=MTL)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ End date. """

    user: list[str] = dataclasses.field(default_factory=list)
    """ Which user(s) to query information for. Leave blank to get a global compute profile."""

    users_file: Path | None = None
    # clusters: list[str] = dataclasses.field(default_factory=list)
    # """ Which clusters to query information for. Leave blank to get data from all clusters."""

    cache_dir: Path = dataclasses.field(
        default=Path(os.environ.get("SCRATCH", tempfile.gettempdir())), hash=False
    )
    """ Directory where temporary files will be stored."""

    verbose: int = simple_parsing.field(
        alias=["-v", "--verbose"], action="count", default=0, hash=False
    )

    def get_users(self) -> list[str]:
        if self.users_file:
            assert not self.user, "can't use both user_file and users!"
            return self.users_file.read_text().splitlines(keepends=False)
        return self.user

    def unique_path(self, label: str = "", extension: str = ".pkl") -> Path:
        users = self.get_users()
        user_portion = "+".join(sorted(users)) if users else "all"
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


def _setup_logging(verbose: int):
    logging.basicConfig(
        handlers=[rich.logging.RichHandler()],
        format="%(message)s",
        level=logging.ERROR,
    )
    logging.getLogger("sarc").setLevel(logging.WARNING)

    if verbose == 0:
        logger.setLevel("WARNING")
    elif verbose == 1:
        logger.setLevel("INFO")
    else:
        logger.setLevel("DEBUG")


def main():
    options = simple_parsing.parse(Options)
    _setup_logging(options.verbose)
    logger.debug(options)

    cache_file = options.unique_path()
    users = options.get_users()

    print(
        f"Looking up for data between {options.start} and {options.end} for users: {users or 'all'}"
    )

    if cache_file.exists():
        logger.info(f"Reading previous data from {cache_file}.")
        df = pd.read_pickle(cache_file)
        assert isinstance(df, pd.DataFrame)
    elif (
        options.user
        and (
            all_users_cache_file := dataclasses.replace(options, user=[]).unique_path()
        ).exists()
    ):
        logger.info(
            f"Reusing and filtering previous data for all users at {all_users_cache_file}."
        )
        df = pd.read_pickle(all_users_cache_file)
        assert isinstance(df, pd.DataFrame)
        df = df[df["user"].isin(users)]
    else:
        logger.info(
            f"Did not find previous results at {cache_file}. Fetching job data."
        )
        df = load_job_series(
            start=options.start,
            end=options.end,
            user=users or None,  # support querying for multiple users.
            clip_time=False,  # True,
        )
        logger.info(f"Saving data to {cache_file}")
        df.to_pickle(cache_file)

    for time_column in ["submit_time", "start_time", "end_time"]:
        # df[time_column] = df[time_column].dt.tz_localize("UTC").dt.tz_convert(MTL)
        df[time_column] = df[time_column].dt.tz_convert(MTL)

    validate_gpu_ram()
    df.fillna({"requested.gres_gpu": 0, "allocated.gres_gpu": 0}, inplace=True)
    df = fix_lost_jobs(df)
    df = fix_unaligned_cache(df, options.start, options.end)
    df = remove_old_nodes(df)
    replace_outlier_stats_with_na(df)

    # Clusters we want to compare
    clusters = ["mila", "narval", "beluga", "cedar", "graham"]

    # Filter clusters
    df = df[df["cluster_name"].isin(clusters)]

    df = fix_missing_gpu_type(df, clusters)

    df = fix_rgu_discrepencies(df)

    df = fill_missing_metrics_using_means(df, clusters)

    df = compute_cost_and_waste(df)

    # Filter out non-started jobs
    df = df[df["start_time"] != 0]
    assert (
        df["requested.gres_gpu"].notnull().all()
        and (df["requested.gres_gpu"] >= 0).all()
    )

    print("Validate these values compared to DRAC ccdb stats.")
    validate_data(df, options.start, options.end)

    # print("Multi-GPU jobs")
    # print(">1")
    # print(
    #     df[df["requested.gres_gpu"] > 1].groupby("cluster_name")["gpu_cost"].sum()
    #     / df.groupby("cluster_name")["gpu_cost"].sum()
    # )
    # print(">3")
    # print(
    #     df[df["requested.gres_gpu"] > 3].groupby("cluster_name")["gpu_cost"].sum()
    #     / df.groupby("cluster_name")["gpu_cost"].sum()
    # )
    # print("Multi-node jobs")
    # print(
    #     df[df["nodes"].str.len() > 1].groupby("cluster_name")["gpu_cost"].sum()
    #     / df.groupby("cluster_name")["gpu_cost"].sum()
    # )
    # print()


# TODO: Fix RGU stuff by adding the json files in secrets


def replace_outlier_stats_with_na(df: pd.DataFrame):
    """`load_job_series` only removes the H100 outliers for gpu_utilization.

    Here we do the rest.
    """
    # Shouldn't really be necessary.
    df.loc[df["gpu_utilization"] > 1, "gpu_utilization"] = pd.NA
    for bits in [16, 32, 64]:
        df.loc[df[f"gpu_utilization_fp{bits}"] > 1, f"gpu_utilization_fp{bits}"] = pd.NA
    df.loc[df["gpu_memory"] > 1, "gpu_memory"] = pd.NA
    df.loc[df["gpu_sm_occupancy"] > 1, "gpu_sm_occupancy"] = pd.NA
    df.loc[df["gpu_power"] > 10e10, "gpu_power"] = pd.NA


def fill_missing_metrics_using_means(df: pd.DataFrame, clusters: list[str]):
    """TODO: Fill in missing JobStatistics metrics using average of available data."""
    stat_columns = list(JobStatistics.__fields__.keys())

    # no_na = df.dropna(subset=stat_columns, how="any")
    # assert no_na.shape[0] > 0

    across_cluster_means = {col: df[col].dropna().mean() for col in stat_columns}
    logger.debug(
        f"Mean of stats across all clusters: {get_stats_str(across_cluster_means)}"
    )

    # todo: cpu_utilization and system_memory should be there for all jobs, right?
    gpu_columns = [col for col in stat_columns if col.startswith("gpu")]
    cpu_system_stats_columns = list(set(stat_columns) - set(gpu_columns))

    # todo: why does the previous step create NANs again in the allocated.gres_gpu?
    assert df["allocated.gres_gpu"].notnull().all()

    # Create some masks
    has_gpu = df["allocated.gres_gpu"] > 0
    is_missing_gpu_stats = has_gpu & df[gpu_columns].isna().any(axis=1)
    is_missing_system_stats = df[cpu_system_stats_columns].isna().any(axis=1)

    logger.debug(f"{has_gpu.mean()=:.2%}")
    logger.debug(f"{is_missing_gpu_stats.mean()=:.2%}")
    logger.debug(f"{is_missing_system_stats.mean()=:.2%}")

    for cluster in clusters:
        is_in_cluster = df["cluster_name"] == cluster
        cluster_mean_stats = {
            col: df[is_in_cluster][col].dropna().mean() for col in stat_columns
        }
        # Use the cluster average if possible, otherwise use the average across all clusters.
        missing_stats = [k for k, v in cluster_mean_stats.items() if np.isnan(v)]
        if missing_stats:
            logger.warning(
                f"Missing stats for {cluster=}: {missing_stats}.\n"
                f"The average of available stats across other clusters will be used."
            )
        stats_to_use = {
            col: (
                cluster_mean
                if not np.isnan(cluster_mean)
                else across_cluster_means[col]
                # todo: else use across-users mean.
            )
            for col, cluster_mean in cluster_mean_stats.items()
        }
        missing_stats_str = get_stats_str(
            {k: v for k, v in stats_to_use.items() if k in missing_stats}
        )
        if missing_stats:
            logger.info(
                f"Stats to be used when infilling missing values for {cluster}: {missing_stats_str}"
            )
        df.loc[is_in_cluster & is_missing_gpu_stats, gpu_columns] = [
            stats_to_use[col] for col in gpu_columns
        ]
        df.loc[is_in_cluster & is_missing_system_stats, cpu_system_stats_columns] = [
            stats_to_use[col] for col in cpu_system_stats_columns
        ]
    return df


def get_stats_str(stats_to_use: dict[str, np.ndarray | float]):
    return {
        k: (f"{v:.1f}" if k == "gpu_power" else f"{v:.2%}")
        for k, v in stats_to_use.items()
    }


def fix_lost_jobs(df: pd.DataFrame):
    _28_days = timedelta(days=28)
    lost_jobs = df["elapsed_time"] > _28_days.total_seconds()
    df.loc[lost_jobs, "elapsed_time"] = _28_days.total_seconds()
    df.loc[lost_jobs, "end_time"] = df.loc[lost_jobs, "start_time"] + _28_days
    return df


def fix_unaligned_cache(df: pd.DataFrame, start: datetime, end: datetime):
    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    df = df[df["end_time"].isnull() | (df["end_time"] > start)]
    df = df[(~df["start_time"].isnull()) & (df["start_time"] < end)]

    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    return df


def filter_users(df: pd.DataFrame, users_file: Path):
    with users_file.open("r") as file:
        users = set(file.read().splitlines())
        df = df[df["user"].isin(users)]

    return df


def remove_old_nodes(df: pd.DataFrame):
    # Filter old nodes and unallocated jobs
    nodes = df["nodes"].str[0]
    old_nodes = [
        "kepler3",
        "kepler4",
        "kepler5",
        "mila01",
        "mila02",
        "mila03",
        "rtx1",
        "rtx3",
        "rtx4",
        "rtx5",
        "rtx7",
    ]
    df = df[~(nodes.isnull() | (nodes.isin(old_nodes)))]
    return df


def validate_gpu_ram():
    missing_ram = set(gpu_name_mapping.values()) - set(gpu_ram.keys())
    if missing_ram:
        raise ValueError(f"Missing ram: {missing_ram}")


# todo: replace with the actual `get_node_to_gpu` function once it works with the client config.
def _get_node_to_gpu(cluster_name: str):
    with open(Path(__file__).parent.parent / "config/node_to_gpu.json") as f:
        cluster_configs: dict[str, dict[str, str]] = json.load(f)
    return cluster_configs[cluster_name]


def _get_cluster_configs() -> dict[str, ClusterConfig]:
    with open(Path(__file__).parent.parent / "config/sarc-dev.json") as f:
        cluster_configs = {
            k: ClusterConfig.validate(v) for k, v in json.load(f)["clusters"].items()
        }
    return cluster_configs


def fix_missing_gpu_type(df: pd.DataFrame, clusters: list[str]):
    # Fix missing gpu_type
    for cluster_name in clusters:
        node_to_gpu = _get_node_to_gpu(cluster_name=cluster_name)
        non_mapped_gpu_types_mask = (
            (df["cluster_name"] == cluster_name)
            & (df["elapsed_time"] > 0)
            & df["allocated.gpu_type"].isnull()
        )
        # NOTE: We assume uniformity of gpu types on all nodes

        nodes = df[non_mapped_gpu_types_mask]["nodes"].str[0]
        # NOTE: some nodes don't have GPUs, so we have 'allocated.gpu_type' set to `None` in that case.
        mapping = {node: node_to_gpu.get(node) for node in nodes.unique()}
        df.loc[non_mapped_gpu_types_mask, "allocated.gpu_type"] = nodes.map(mapping)

    missing_gpu_types_mask = (
        (df["requested.gres_gpu"] > 0)
        * (df["elapsed_time"] > 0)
        * (df["allocated.gpu_type"].isnull())
    )
    missing_gpu_types = df[missing_gpu_types_mask]

    if missing_gpu_types.shape[0] > 0:
        print(
            "GPU types not mapped",
            missing_gpu_types.groupby(["cluster_name"]).count()["id"],
        )
        print(missing_gpu_types["nodes"].str[0].unique())

        breakpoint()

    missing_mappings = set(
        df[~df["allocated.gpu_type"].isnull()]["allocated.gpu_type"].unique()
    ) - set(gpu_name_mapping.keys())
    if missing_mappings:
        print("Missing mappings:", missing_mappings)
        print(
            df[df["allocated.gpu_type"].isin(missing_mappings)]
            .groupby(["allocated.gpu_type", "cluster_name"])
            .count()["id"]
        )
        breakpoint()

    df["allocated.gpu_type"] = df["allocated.gpu_type"].map(gpu_name_mapping)
    df.fillna({"allocated.gpu_type": "unknown"}, inplace=True)

    return df


def fix_rgu_discrepencies(df: pd.DataFrame):
    # NOTE: Fixing switch to RGU billing for a second time on Narval
    # narval_config = config().clusters["narval"]
    cluster_configs = _get_cluster_configs()
    narval_config = cluster_configs["narval"]

    slice_during_rgu_time = (
        (df["cluster_name"] == "narval")
        & (df["start_time"] >= datetime(2023, 11, 28, tzinfo=MTL))
        & (
            df["start_time"]
            < datetime.fromisoformat(narval_config.rgu_start_date).astimezone(MTL)
        )
        & (df["elapsed_time"] > 0)
    )
    non_updated_df = df[slice_during_rgu_time]

    # NOTE: Hacky fix, because we don't use the sarc-dev config.
    # df = update_job_series_rgu(df)
    # for cluster_config in config().clusters.values():
    #     update_cluster_job_series_rgu(df, cluster_config)
    # return df
    for cluster_config in cluster_configs.values():
        update_cluster_job_series_rgu(df, cluster_config)

    gpu_to_rgu_billing = {
        "a100-40gb": 700,
        "a100-40gb-3g.20gb": 1714.29 / 4000 * 700,
        "a100-40gb-4g.20gb": 2285.71 / 4000 * 700,
    }
    col_ratio_rgu_by_gpu = df.loc[slice_during_rgu_time, "allocated.gpu_type"].map(
        gpu_to_rgu_billing
    )
    df.loc[slice_during_rgu_time, "allocated.gpu_type_rgu"] = col_ratio_rgu_by_gpu
    df.loc[slice_during_rgu_time, "allocated.gres_rgu"] = non_updated_df[
        "allocated.gres_gpu"
    ]
    df.loc[slice_during_rgu_time, "allocated.gres_gpu"] = (
        non_updated_df["allocated.gres_gpu"] / col_ratio_rgu_by_gpu
    )

    # TODO: Apply only during this period

    # narval_rgu['mappings'] = {"a100-40gb": 700, "a100-40gb-3g.20gb": 1714.29/4000*700, "a100-40gb-4g.20gb": 2285.71/4000*700}
    # narval_config.rgu_start_date = "2024-04-01"
    # with open(narval_config.gpu_to_rgu_billing, 'w', encoding='utf-8') as file:
    #     json.dump(narval_rgu, file)
    # df = update_cluster_job_series_rgu(df, narval_config)

    # narval_rgu['mappings'] = previous_mappings
    # with open(narval_config.gpu_to_rgu_billing, 'w', encoding='utf-8') as file:
    #     json.dump(narval_rgu, file)
    # End of hacky fix

    # TODO we should fix this in SARC.
    slice_during_rgu_time = (
        (df["cluster_name"] != "mila")
        & (df["start_time"] >= datetime(2024, 4, 1, tzinfo=MTL))
        & (df["elapsed_time"] > 0)
    )
    df.loc[slice_during_rgu_time, "allocated.cpu"] /= 1000.0

    df["allocated.gpu_type_rgu"] = df["allocated.gpu_type"].map(rgus)

    # todo: why are there NANs again in the allocated.gres_gpu?
    assert df["allocated.gres_gpu"].notnull().all()
    assert df["requested.gres_gpu"].notnull().all()
    return df


def validate_data(stats: pd.DataFrame, start: datetime, end: datetime):
    seconds_in_a_year = timedelta(days=365.242374).total_seconds()

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    stats["cpu_billed"] = stats["elapsed_time"] * stats["allocated.cpu"]
    stats["gpu_billed"] = stats["elapsed_time"] * stats["allocated.gres_gpu"]

    max_delta = end - start  # timedelta(seconds=(end - start).total_seconds())

    if max_delta > timedelta(days=30):
        frame_size = "MS"
    else:
        frame_size = max_delta

    stats = compute_time_frames(
        stats,
        ["gpu_cost", "cpu_cost", "cpu_equivalent_cost", "gpu_equivalent_cost"],
        start,
        end,
        frame_size=frame_size,
    )

    def _get_cost_per_month_per_year_unclear(name: str):
        # todo: pretty unclear what exactly this is calculating.
        # Need to clarify again a bit with Xavier.

        # cost per month :=
        cost_per_month = stats.groupby(["cluster_name", "timestamp"])[name].sum()
        return (
            (cost_per_month / seconds_in_a_year)
            .reset_index()
            .pivot(index="timestamp", columns="cluster_name")
        )

    print("CPU equivalent usage per month")
    print(
        cpu_years_per_month := _get_cost_per_month_per_year_unclear(
            "cpu_equivalent_cost"
        )
    )
    # print(
    #     # todo: ask @bouthilx why there is this offset, /6, * 12 stuff.
    #     cpu_years_per_month[6:].sum() / 6.0 * 12
    # )

    # gpu cost per month := GPU reserved (full-time) for a month.
    gpu_years_per_month = _get_cost_per_month_per_year_unclear("gpu_cost")
    print("GPU usage per month")
    print(gpu_years_per_month)
    # print(gpu_years_per_month[6:].sum())  # TODO: why this [6:].sum()

    gpu_year_equivalent_cost_per_month = _get_cost_per_month_per_year_unclear(
        "gpu_equivalent_cost"
    )
    print("GPU equivalent usage per months")
    print(gpu_year_equivalent_cost_per_month)
    # print(gpu_year_equivalent_cost_per_month[6:].sum())

    stats["rgu_equivalent_cost"] = (
        stats["gpu_equivalent_cost"] * stats["allocated.gpu_type_rgu"]
    )
    rgu_years_per_month = _get_cost_per_month_per_year_unclear("rgu_equivalent_cost")
    print("RGU equivalent usage per months")
    print(rgu_years_per_month)
    # print(rgu_years_per_month[6:].sum() / 6.0 * 12)

    ccdb_usage_cpu, ccdb_usage_gpu = get_ccdb_usage_data()

    print("CPU usage data from CCDB website:")
    print(ccdb_usage_cpu)

    _index = "timestamp"
    ccdb_usage_cpu.index.name = _index
    ccdb_usage_gpu.index.name = _index

    def _sort_columns(df: pd.DataFrame):
        df.columns = pd.MultiIndex.from_tuples(
            sorted(df.reorder_levels(["cluster_name", None], axis="columns").columns),
            names=["cluster_name", "source"],
        )

    merged_cpu = (
        ccdb_usage_cpu.rename(columns={"usage": "CCDB"}, level=0)
        # Merging with the "cpu_cost" doesn't seem to make sense. There may be a bug, the values are higher than expected.
        .merge(
            _get_cost_per_month_per_year_unclear("cpu_cost"),
            on=_index,
            how="outer",
        )
        .merge(
            _get_cost_per_month_per_year_unclear("cpu_equivalent_cost"),
            on=_index,
            how="outer",
        )
    )
    _sort_columns(merged_cpu)

    print("CCDB vs SARC (CPU):")
    print("```")
    print(merged_cpu.to_markdown())
    print("```")

    merged_gpu = (
        ccdb_usage_gpu.rename(columns={"usage": "CCDB"}, level=0)
        .merge(
            _get_cost_per_month_per_year_unclear("gpu_cost"),
            on=_index,
            how="outer",
        )
        # Merging with the "gpu_equivalent_cost" doesn't seem to make sense. There may be a bug.
        .merge(
            _get_cost_per_month_per_year_unclear("gpu_equivalent_cost"),
            on=_index,
            how="outer",
        )
    )
    _sort_columns(merged_gpu)

    print("CCDB vs SARC (GPU):")
    print("```")
    print(merged_gpu.to_markdown())
    print("```")

    # merged_cpu.to_csv("cpu_comparison.csv")
    # merged_gpu.to_csv("gpu_comparison.csv")

    return


def compute_time_frames(
    jobs: pd.DataFrame,
    columns: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    start_column: str = "start_time",
    end_column: str = "end_time",
    frame_size: str | timedelta = "MS",
    # frame_size: timedelta = timedelta(days=7),
    callback: None | Callable = None,
):
    """Slice jobs into time frames and adjust columns to fit the time frames.

    Jobs that start before `start` or ends after `end` will have their running
    time clipped to fitting within the interval (`start`, `end`).

    Jobs spanning multiple time frames will have their running time sliced
    according to the time frames.

    The resulting DataFrame will have the additional columns 'elapsed_time' and 'timestamp'
    which represent the elapsed_time of a job within a time frame and the start of the time frame.

    Parameters
    ----------
    jobs: pandas.DataFrame
        DataFrame containing jobs data. Typically generated with `load_job_series`.
        Must contain columns `start` and `end`.
    columns: list of str
        Columns to adjust based on time frames.
    start: datetime, optional
        Start of the time frame. If None, use the first job start time.
    end: datetime, optional
        End of the time frame. If None, use the last job end time.
    frame_size: timedelta, optional
        Size of the time frames used to compute histograms. Default to 7 days.

    Examples
    --------
    >>> data = pd.DataFrame(
        [
            [datetime(2023, 3, 5), datetime(2023, 3, 6), "a", "A", 10],
            [datetime(2023, 3, 6), datetime(2023, 3, 9), "a", "B", 10],
            [datetime(2023, 3, 6), datetime(2023, 3, 7), "b", "B", 20],
            [datetime(2023, 3, 6), datetime(2023, 3, 8), "b", "B", 20],
        ],
        columns=["start_time", "end_time", "user", "cluster", 'cost'],
    )
    >>> compute_time_frames(data, columns=['cost'], frame_size=timedelta(days=2))
           start        end user cluster       cost  elapsed_time  timestamp
    0 2023-03-05 2023-03-06    a       A  10.000000   86400.0 2023-03-05
    1 2023-03-06 2023-03-09    a       B   3.333333   86400.0 2023-03-05
    2 2023-03-06 2023-03-07    b       B  20.000000   86400.0 2023-03-05
    3 2023-03-06 2023-03-08    b       B  10.000000   86400.0 2023-03-05
    1 2023-03-06 2023-03-09    a       B   6.666667  172800.0 2023-03-07
    3 2023-03-06 2023-03-08    b       B  10.000000   86400.0 2023-03-07
    """
    if columns is None:
        columns = []

    if start is None:
        start = jobs[start_column].min()

    if end is None:
        end = jobs[end_column].max()

    data_frames = []

    total_elapsed_times = (jobs[end_column] - jobs[start_column]).dt.total_seconds()

    jobs = jobs.copy()
    for time_column in [start_column, end_column]:
        jobs[time_column] = (
            jobs[time_column].dt.tz_localize(None).astype("datetime64[ns]")
        )

    timestamps = pd.date_range(
        start, end, freq=frame_size, inclusive="both"
    ).tz_localize(None)
    # for frame_start in pd.date_range(start, end, freq=f"MS"):
    for frame_start, frame_end in zip(timestamps, timestamps[1:]):
        mask = (jobs[start_column] < frame_end) & (jobs[end_column] > frame_start)
        frame = jobs[mask].copy()
        total_elapsed_times_in_frame = total_elapsed_times[mask]
        frame["elapsed_time"] = (
            frame[end_column].clip(frame_start, frame_end)
            - frame[start_column].clip(frame_start, frame_end)
        ).dt.total_seconds()

        # Adjust columns to fit the time frame.
        for column in columns:
            frame[column] *= frame["elapsed_time"] / total_elapsed_times_in_frame

        frame["timestamp"] = frame_start

        if callback:
            callback(frame, frame_start, frame_end)

        data_frames.append(frame)

    return pd.concat(data_frames, axis=0)


AllocationName = str
ClusterName = str


def get_ccdb_usage_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the CPU and GPU usage data from the CCDB website."""
    ccdb_data_file = Path(__file__).parent / "ccdb_data.yaml"
    ccdb_compute_usage: dict[
        AllocationName, dict[Literal["cpu", "gpu"], dict[str, float]]
    ] = yaml.safe_load(ccdb_data_file.read_text())
    month_str_to_month_index = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    data: dict[tuple[datetime, ClusterName, Literal["cpu", "gpu"]], float] = {}
    for allocation, usage_dict in ccdb_compute_usage.items():
        assert "_" in allocation
        cluster, _, _type = allocation.partition("_")
        for usage_type, usage_data in usage_dict.items():
            for month_str, usage in usage_data.items():
                month, year = month_str.split(" ")
                month = month_str_to_month_index[month]
                date = datetime(year=int(year), month=month, day=1)
                if (date, cluster, usage_type) not in data:
                    data[(date, cluster, usage_type)] = usage
                else:
                    data[(date, cluster, usage_type)] += usage
    index = pd.MultiIndex.from_tuples(
        data.keys(), names=["date", "cluster_name", "type"]
    )
    df = pd.DataFrame(list(data.values()), index=index, columns=["usage"])
    df = df.reorder_levels(["type", "cluster_name", "date"]).sort_index()

    def pivot_ccdb_data(ccdb_data: pd.DataFrame, type: Literal["cpu", "gpu"]):
        df = (
            ccdb_data.xs(type, level="type")
            .groupby(["cluster_name", "date"])
            .sum()
            .unstack(level="cluster_name")
            .fillna(0)
        )
        assert isinstance(df, pd.DataFrame)
        return df

    return pivot_ccdb_data(df, "cpu"), pivot_ccdb_data(df, "gpu")


if __name__ == "__main__":
    main()
