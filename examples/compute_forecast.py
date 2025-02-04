from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from sarc.client.series import compute_cost_and_waste, load_job_series
from sarc.config import MTL, ClusterConfig
from sarc.jobs.series import (
    update_cluster_job_series_rgu,
)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),
        help="Start date to fetch data. Default corresponds to 7 days ago.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date to fetch data. Default corresponds to current time.",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="File to save loaded data as a panda array in a pickle file. Does not save anything by default.",
    )
    parser.add_argument(
        "--include-plots",
        type=str,
        nargs="*",
        help="Plots to include: analysis, projected, cumulative, gpu_dist, allocation_barplot, weekly_demands",
    )

    parser.add_argument(
        "--filter-users",
        type=Path,
        help="Path to a file containing a list of users to filter. One user per line. If not provided, all users are included.",
    )

    options = parser.parse_args()

    if not options.include_plots:
        options.include_plots = [
            "analysis",
            "projected",
            "cumulative",
            "gpu_dist",
            "allocation_barplot",
            "weekly_demands",
        ]

    start = datetime.strptime(options.start, "%Y-%m-%d").replace(tzinfo=MTL)
    end = (
        datetime.strptime(options.end, "%Y-%m-%d").replace(tzinfo=MTL)
        if options.end
        else datetime.now(tz=MTL)
    )

    print(f"Looking up for data between {start} and {end}")

    if options.cache_file and os.path.exists(options.cache_file):
        df = pd.read_pickle(options.cache_file)
    else:
        df = load_job_series(
            start=start,
            end=end,
            clip_time=False,  # True,
            # progress_bar=True,
        )
        if options.cache_file:
            df.to_pickle(options.cache_file)

    if options.filter_users:
        df = filter_users(df, options.filter_users)

    for time_column in ["submit_time", "start_time", "end_time"]:
        # df[time_column] = df[time_column].dt.tz_localize("UTC").dt.tz_convert(MTL)
        df[time_column] = df[time_column].dt.tz_convert(MTL)

    df = fix_lost_jobs(df)
    df = fix_unaligned_cache(df, start, end)
    df = remove_old_nodes(df)

    # Clusters we want to compare
    clusters = ["mila", "narval", "beluga", "cedar", "graham"]

    # Filter clusters
    df = df[df["cluster_name"].isin(clusters)]

    validate_gpu_ram(df)
    df = fix_missing_gpu_type(df, clusters)

    # NOTE: This will also compute `update_job_series_rgu`
    df = fix_rgu_discrepencies(df)

    df = compute_cost_and_waste(df)

    # Filter out non-started jobs
    df = df[df["start_time"] != 0]

    df["requested.gres_gpu"].fillna(0, inplace=True)

    print("Validate these values compared to DRAC ccdb stats.")
    validate_data(df, start, end)

    print("Multi-GPU jobs")
    print(">1")
    print(
        df[df["requested.gres_gpu"] > 1].groupby("cluster_name")["gpu_cost"].sum()
        / df.groupby("cluster_name")["gpu_cost"].sum()
    )
    print(">3")
    print(
        df[df["requested.gres_gpu"] > 3].groupby("cluster_name")["gpu_cost"].sum()
        / df.groupby("cluster_name")["gpu_cost"].sum()
    )
    print("Multi-node jobs")
    print(
        df[df["nodes"].str.len() > 1].groupby("cluster_name")["gpu_cost"].sum()
        / df.groupby("cluster_name")["gpu_cost"].sum()
    )
    print()


def fix_lost_jobs(df: pd.DataFrame):
    lost_jobs = df["elapsed_time"] > (28 * 24 * 60 * 60)
    df.loc[lost_jobs, "elapsed_time"] = 28 * 24 * 60 * 60
    df.loc[lost_jobs, "end_time"] = df.loc[lost_jobs, "start_time"] + timedelta(
        seconds=28 * 24 * 60 * 60
    )
    return df


def fix_unaligned_cache(df: pd.DataFrame, start: datetime, end: datetime):
    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    df = df[df["end_time"].isnull() | (df["end_time"] > start)]
    df = df[(~df["start_time"].isnull()) & (df["start_time"] < end)]

    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    return df


def filter_users(df, users_file):
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


def validate_gpu_ram(df: pd.DataFrame):
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
            k: ClusterConfig.validate(v) for k, v in json.load(f).items()
        }
    return cluster_configs


def fix_missing_gpu_type(df: pd.DataFrame, clusters: list[str]):
    # Fix missing gpu_type
    for cluster_name in clusters:
        node_to_gpu = _get_node_to_gpu(cluster_name=cluster_name)
        non_mapped_gpu_types_mask = (
            (df["cluster_name"] == cluster_name)
            & (df["allocated.gpu_type"].isnull())
            & (df["elapsed_time"] > 0)
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
    df["allocated.gpu_type"].fillna("unknown", inplace=True)

    return df


def fix_rgu_discrepencies(df: pd.DataFrame):
    # NOTE: Fixing switch to RGU billing for a second time on Narval

    cluster_configs = _get_cluster_configs()
    narval_config = cluster_configs["narval"]
    # narval_config = config().clusters["narval"]

    # with open(narval_config.gpu_to_rgu_billing, "r", encoding="utf-8") as file:
    #     narval_rgu = json.load(file)

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

    return df


def validate_data(stats: pd.DataFrame, start: datetime, end: datetime):
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    stats["cpu_billed"] = stats["elapsed_time"] * stats["allocated.cpu"]
    stats["gpu_billed"] = stats["elapsed_time"] * stats["allocated.gres_gpu"]

    max_delta = timedelta(seconds=(end - start).total_seconds())
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
    cpu_cost_per_month = stats.groupby(["cluster_name", "timestamp"])[
        "cpu_equivalent_cost"
    ].sum()
    print("CPU usage per months")
    print(
        (cpu_cost_per_month / (end - start).total_seconds())
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")
    )
    print(
        (cpu_cost_per_month / (end - start).total_seconds())
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")[6:]
        .sum()
        / 6.0
        * 12
    )

    gpu_cost_per_month = stats.groupby(["cluster_name", "timestamp"])["gpu_cost"].sum()
    print("GPU usage per months")
    # breakpoint()
    print(
        (gpu_cost_per_month / (365 * 24 * 60 * 60))
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")
    )
    print(
        (gpu_cost_per_month / (365 * 24 * 60 * 60))
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")[6:]
        .sum()
    )

    gpu_cost_per_month = stats.groupby(["cluster_name", "timestamp"])[
        "gpu_equivalent_cost"
    ].sum()
    print("GPU equivalent usage per months")
    print(
        (gpu_cost_per_month / (365 * 24 * 60 * 60))
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")
    )
    print(
        (gpu_cost_per_month / (365 * 24 * 60 * 60))
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")[6:]
        .sum()
    )

    # print(stats[(stats['start_time'] > datetime(2024, 4, 1)) & (stats['cluster_name'] == 'cedar') & ~stats['gpu_cost'].isnull()].sort_values('gpu_overbilling_cost')[['requested.cpu', 'requested.mem', 'requested.gres_gpu', 'gpu_overbilling_cost', 'user', 'allocated.gres_gpu', 'allocated.node', 'job_id', 'allocated.gpu_type', 'nodes']][-100:])

    stats["rgu_equivalent_cost"] = (
        stats["gpu_equivalent_cost"] * stats["allocated.gpu_type_rgu"]
    )
    rgu_cost_per_month = stats.groupby(["cluster_name", "timestamp"])[
        "rgu_equivalent_cost"
    ].sum()
    print("RGU equivalent usage per months")
    print(
        (rgu_cost_per_month / (365 * 24 * 60 * 60))
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")
    )
    print(
        (rgu_cost_per_month / (365 * 24 * 60 * 60))
        .reset_index()
        .pivot(index="timestamp", columns="cluster_name")[6:]
        .sum()
        / 6.0
        * 12
    )

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

    print(start, end)

    jobs = jobs.copy()
    for time_column in [start_column, end_column]:
        jobs[time_column] = (
            jobs[time_column].dt.tz_localize(None).astype("datetime64[ns]")
        )

    timestamps = pd.date_range(start, end, freq=frame_size, inclusive="both")
    # for frame_start in pd.date_range(start, end, freq=f"MS"):
    for frame_start, frame_end in zip(timestamps, timestamps[1:]):
        frame_start = frame_start.tz_localize(None)
        frame_end = frame_end.tz_localize(None)

        mask = (jobs[start_column] < frame_end) * (jobs[end_column] > frame_start)
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


if __name__ == "__main__":
    main()
