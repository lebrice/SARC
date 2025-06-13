from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import random
from re import sub
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import simple_parsing
import yaml
from typing_extensions import Self
import time

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    MofNCompleteColumn,
)

os.environ.setdefault("SARC_CONFIG", "config/sarc-client.yaml")

from sarc.client.job import JobStatistics
from sarc.client.series import compute_cost_and_waste, load_job_series
from sarc.client.users.api import get_users
from sarc.config import MTL, ClusterConfig
from sarc.jobs.series import update_cluster_job_series_rgu

if (_sarc_config := os.environ.get("SARC_CONFIG")) and Path(_sarc_config).exists():
    CONFIG_FOLDER = Path(_sarc_config).parent
else:
    CONFIG_FOLDER = Path(__file__).parent / "config"


def main():
    with Live(make_layout(), refresh_per_second=4) as live:
        for _ in range(40):
            live.update(make_layout())
            time.sleep(5)


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(Header(), name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=7),
    )
    layout["main"].split_column(
        Layout(make_waste_overview_table(), name="waste_overview_table"),
        Layout(make_cluster_overview_table(), name="cluster_overvier"),
        Layout(make_biggest_waster_job_info_table(), name="biggest_waster_jobs"),
    )
    # layout["side"].split(
    #     Layout(make_avail_gpus_panel(), name="box1"),
    #     Layout(make_avg_gpu_util_panel(), name="box2"),
    # )
    return layout


def make_waste_overview_table() -> Table:
    """Make a new table."""
    table = Table(expand=True)
    table.add_column("User")
    table.add_column("Cluster")
    table.add_column("Total GPUs")
    table.add_column("Average GPU Utilization")
    table.add_column("Wasted GPU*hours in last 7 days")
    n_users = 5
    used_gpuss = np.random.uniform(0, 100, size=n_users)
    avg_gpu_util = np.random.uniform(0, 1, size=n_users)
    wastes = used_gpuss * (1 - avg_gpu_util)
    for index in reversed(np.argsort(wastes)):
        username = random.choice(["Bob", "Alice", "Charlie", "Dave", "Eve"])
        cluster = random.choice(["mila", "narval", "drac", "beluga", "tamia"])
        used_gpus = used_gpuss[index]
        gpu_util = avg_gpu_util[index]
        waste = wastes[index]

        table.add_row(
            username,
            cluster,
            str(used_gpus),
            (f"[red]{gpu_util:.2%}" if gpu_util < 0.2 else f"[yellow]{gpu_util:.2%}"),
            f"[blue]{waste:.2f}",
        )
    return table


def make_cluster_overview_table() -> Table:
    table = Table(expand=True)
    table.add_column("Cluster", justify="left")
    table.add_column("Available / Total GPUs", justify="right")
    table.add_column("Average GPU Utilization")
    table.add_column("Mila students using this cluster", justify="right")
    total_mila_users = 1052
    for cluster in ["mila", "narval", "drac", "beluga", "tamia"]:
        total_gpus = random.randint(500, 1000)
        avail_gpus = random.randint(0, total_gpus)
        mila_users = random.randint(0, total_mila_users)
        if cluster == "mila":
            mila_users = 0.8 * total_mila_users
        used_gpus_pct = avail_gpus / total_gpus
        pct_of_mila_users = mila_users / total_mila_users
        gpu_util = random.random()
        table.add_row(
            cluster,
            f"{avail_gpus} / {total_gpus} ({used_gpus_pct:.2%})",
            f"{gpu_util:.2%}",
            f"{mila_users} / {total_mila_users} ({pct_of_mila_users:.2%})",
        )
    return table


def make_biggest_waster_job_info_table() -> Table:
    table = Table(title="Biggest wasting individual jobs", expand=True)
    table.add_column("Job ID", justify="right")
    table.add_column("User", justify="left")
    table.add_column("Cluster", justify="left")
    table.add_column("Workdir", justify="left")
    table.add_column("submit command", justify="left")
    table.add_column("Average GPU Utilization", justify="right")
    table.add_column("Wasted GPU*hours", justify="right")
    njobs = 10
    avg_gpu_utilizations = sorted(np.random.uniform(0, 1, size=njobs), reverse=True)

    for avg_gpu_utilization in avg_gpu_utilizations:
        job_id = random.randint(1000000, 9999999)
        username = random.choice(["Bob", "Alice", "Charlie", "Dave", "Eve"])
        cluster = random.choice(["mila", "narval", "drac", "beluga", "tamia"])
        workdir = f"/path/to/workdir/{job_id}"
        submit_command = (
            f"sbatch --gres=gpu:{random.randint(1, 4)} --time=01:00:00 {workdir}/run.sh"
        )
        wasted_gpu_hours = random.uniform(0, 100)
        table.add_row(
            str(job_id),
            username,
            cluster,
            workdir,
            submit_command,
            f"[red]{avg_gpu_utilization:.2%}"
            if avg_gpu_utilization < 0.2
            else f"[green]{avg_gpu_utilization:.2%}",
            f"[blue]{wasted_gpu_hours:.2f}",
        )
    return table


def make_avail_gpus_panel() -> Panel:
    avail_gpus = Progress(
        "{task.description}",
        # SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    mila_total_gpus = 853
    mila_available_gpus = random.randint(0, mila_total_gpus)
    avail_gpus.add_task("Mila", completed=mila_available_gpus, total=mila_total_gpus)
    tamia_total_gpus = 164
    tamia_avail_gpus = random.randint(0, tamia_total_gpus)
    avail_gpus.add_task("Tamia", completed=tamia_avail_gpus, total=tamia_total_gpus)
    return Panel(
        avail_gpus,
        title="Available GPUs",
        border_style="green",
        padding=(2, 2),
    )


def make_avg_gpu_util_panel() -> Table:
    avg_gpu_util = Progress(
        "{task.description}",
        # SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    avg_gpu_util.add_task("Mila", completed=100 * random.random(), total=100)
    avg_gpu_util.add_task("Tamia", completed=100 * random.random(), total=100)
    avg_gpu_util.add_task("Narval", completed=100 * random.random(), total=100)
    avg_gpu_util.add_task("Beluga", completed=100 * random.random(), total=100)
    return Panel(
        avg_gpu_util,
        title="Available GPUs",  # border_style="green", padding=(2, 2)
    )


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            "[b]SARC[/b] Waste Monitoring",
            datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(grid, style="white on green")


def _midnight(dt: datetime) -> datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


@functools.total_ordering
@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class FilteringOptions:
    """Configuration options for this script."""

    start: datetime = simple_parsing.field(
        default=(_midnight(datetime.now(tz=MTL)) - timedelta(days=30)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ Start date. """

    end: datetime = simple_parsing.field(
        default=_midnight(datetime.now(tz=MTL)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ End date. """

    user: list[str] = dataclasses.field(default_factory=list)
    """ Which user(s) to query information for. Leave blank to get a global compute profile."""

    users_file: Path | None = dataclasses.field(default=None, repr=False)

    clusters: list[str] = dataclasses.field(default_factory=list)
    """ Which clusters to query information for. Leave blank to get data from all clusters."""

    cache_dir: Path = dataclasses.field(
        default=(
            Path(os.environ["CF_DATA"])
            if "CF_DATA" in os.environ
            else Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
        ),
        hash=False,
        repr=False,
    )
    """ Directory where temporary files will be stored."""
    verbose: int = simple_parsing.field(
        alias=["-v", "--verbose"], action="count", default=0, hash=False, repr=False
    )

    def get_users(self, assume_mila_email: bool = False) -> list[str]:
        if self.users_file:
            assert not self.user, "can't use both user_file and users!"
            users = sorted(set(self.users_file.read_text().splitlines(keepends=False)))
        else:
            users = self.user
        user_emails = []
        for user in users:
            if "@" in user:
                user_emails.append(user.strip())
            elif assume_mila_email:
                user_emails.append(user.strip() + "@mila.quebec")
            else:
                raise ValueError(
                    f"User '{user}' does not contain an email address. "
                    "Please provide a valid email address or set `assume_mila_email=True`."
                )
        return sorted(user_emails)

    def unique_path(self, label: str = "", extension: str = ".pkl") -> Path:
        user_emails = self.get_users()
        user_portion = (
            hashlib.md5("+".join(sorted(user_emails)).encode()).hexdigest()
            if user_emails is not None and len(user_emails)
            else "all"
        )
        # cluster_portion = "-".join(self.clusters) if self.clusters else "all"
        start_portion = (
            self.start.strftime("%Y-%m-%d")
            if self.start == _midnight(self.start)
            else str(self.start).replace(" ", "_")
        )
        end_portion = (
            self.end.strftime("%Y-%m-%d")
            if self.end == _midnight(self.end)
            else str(self.end).replace(" ", "_")
        )
        return (
            self.cache_dir
            / f"compute_profile-{user_portion}-{start_portion}-{end_portion}-{label}"
        ).with_suffix(extension)

    def __eq__(self, other: object) -> bool:
        """Returns whether this filter is equal to the other."""
        if not isinstance(other, FilteringOptions):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and set(self.user) == set(other.user)
            and set(self.clusters) == set(other.clusters)
        )

    def __lt__(self, other: Self) -> bool:
        """Returns whether this filter is strictly more restrictive than the other."""
        if not isinstance(other, FilteringOptions):
            return NotImplemented
        self_users = self.get_users(assume_mila_email=True)
        other_users = other.get_users(assume_mila_email=True)
        if self_users == [] and other_users != []:
            # This filter matches all users while the other one doesn't.
            return False
        if self_users != [] and other_users == []:
            # Pretend that the other filter matches one more user than this one,
            # to make the comparison below cleaner.
            other_users = self_users + ["some_random_user_that_isnt_in_self"]
        self_clusters = sorted(set(self.clusters))
        other_clusters = sorted(set(other.clusters))
        if self_clusters == [] and other_clusters != []:
            # This filter matches all clusters while the other one doesn't.
            return False
        if self_clusters != [] and other_clusters == []:
            # Pretend that the other filter matches one more cluster than this one,
            # to make the comparison below cleaner.
            other_clusters = self_clusters + ["some_random_cluster_that_isnt_in_self"]

        return (
            (other.start < self.start)
            and (self.end < other.end)
            and (set(self_users) < set(other_users))
            and (set(self.clusters) < set(other.clusters))
        )


def get_clean_sarc_data(options: FilteringOptions) -> pd.DataFrame:
    """Gets "cleaned" SARC data for a given period, including *lots* of patches."""
    options = dataclasses.replace(
        options,
        start=options.start.astimezone(MTL),
        end=options.end.astimezone(MTL),
    )
    cache_file = options.unique_path()
    _user_emails = options.get_users(assume_mila_email=True)
    assert all(map(_check_is_email_and_lower, _user_emails))

    _all_users = get_users(
        query={
            "$and": [
                {"mila.email": {"$in": _user_emails}},
                {
                    # We have a record with either no start (before 2023), or a start before the end of the period.
                    "$or": [
                        {"record_start": {"$exists": False}},
                        {"record_start": {"$lt": options.end}},
                    ]
                },
                {
                    # We noticed a change in the users data at some point after the start of the period,
                    # or we didnt notice a change (and the user is still active).
                    "$or": [
                        {"record_end": {"$exists": False}},
                        {"record_end": None},
                        {"record_end": {"$gt": options.start}},
                    ]
                },
            ]
        },
        latest=False,
    )
    email_to_usernames: dict[str, list[str]] = defaultdict(list)
    for user in _all_users:
        email_to_usernames[user.mila.email].append(user.mila.username)
        if user.drac:
            email_to_usernames[user.mila.email].append(user.drac.username)

    logger.debug(
        f"Looking up for data between {options.start} and {options.end} for users: {_user_emails or 'all'} and clusters {options.clusters or 'all'}"
    )

    # TODO: Somethign weird
    if cache_file.exists():
        logger.info(f"Reading previous data from {cache_file}.")
        df = pd.read_pickle(cache_file)
        assert isinstance(df, pd.DataFrame)
    elif (
        options.user
        and (
            all_users_cache_file := dataclasses.replace(
                options, user=[], users_file=None
            ).unique_path()
        ).exists()
    ):
        logger.info(
            f"Reusing and filtering previous data for all users at {all_users_cache_file}."
        )
        df = pd.read_pickle(all_users_cache_file)
        assert isinstance(df, pd.DataFrame)
        if _user_emails:
            df = df[df["user.primary_email"].isin(_user_emails)]
    else:
        logger.info(
            f"Did not find previous results at {cache_file}. Fetching job data."
        )
        all_usernames_of_students = sum(
            [email_to_usernames[email] for email in _user_emails], []
        )
        logger.debug(f"Usernames used when querying SARC: {all_usernames_of_students}")
        # In SARC we currently can't query by user.mila.email, so we query with all
        # usernames and filter by user.mila.email after.
        df = load_job_series(
            start=options.start,
            end=options.end,
            user=(
                {"$in": all_usernames_of_students}
                if all_usernames_of_students
                else None
            ),  # support querying for multiple users.
            clip_time=False,  # True,
        )
        if _user_emails and "user.primary_email" in df.columns:
            df = df[df["user.primary_email"].isin(_user_emails)]
        logger.info(f"Saving data to {cache_file}")
        df.to_pickle(cache_file)

    if df.empty:
        return df

    for time_column in ["submit_time", "start_time", "end_time"]:
        # df[time_column] = df[time_column].dt.tz_localize("UTC").dt.tz_convert(MTL)
        df[time_column] = df[time_column].dt.tz_convert(MTL)

    if df.shape[0] == 0:
        # NO data in SARC!
        logger.warning(f"No data found in SARC for {options}.")

    _validate_gpu_ram()

    # Clusters we want to compare
    if options.clusters:
        # Filter clusters
        df = df[df["cluster_name"].isin(options.clusters)]

    df.fillna({"requested.gres_gpu": 0.0, "allocated.gres_gpu": 0.0}, inplace=True)
    df = _fix_lost_jobs(df)
    df = _fix_unaligned_cache(df, options.start, options.end)
    df = _remove_old_nodes(df)
    df = _replace_outlier_stats_with_na(df)
    df = _fix_missing_gpu_type(df)

    _fix_rgu_discrepencies_inplace(df)

    # todo: double-check if this is still needed here.
    df.fillna({"requested.gres_gpu": 0, "allocated.gres_gpu": 0}, inplace=True)

    _fix_allocated_cpus_drac_inplace(df)
    # todo: Do we want to get the averages from the other jobs of the same user on other clusers?
    # Or from the average utilization on that same cluster by different users?
    # IF so, we might need to reload the data for all users here
    # if users := options.get_users():
    #     all_users_data_for_same_period = _get_cleaned_df(
    #         dataclasses.replace(options, user=[], user_file=None)
    #     )
    df = _fix_requested_allocated_gres_gpu(df)
    df = _fill_missing_metrics_using_means(df)

    df = compute_cost_and_waste(df)
    logger.info(
        "Means of GPU cost: %s, waste: %s, equivalent cost: %s",
        df["gpu_cost"].mean(),
        df["gpu_waste"].mean(),
        df["gpu_equivalent_cost"].mean(),
    )
    logger.info(
        "Means of CPU cost: %s, waste: %s, equivalent cost: %s",
        df["cpu_cost"].mean(),
        df["cpu_waste"].mean(),
        df["cpu_equivalent_cost"].mean(),
    )

    # Sanity checks.
    assert (df["start_time"] != 0).all()
    assert (_requested_gres_gpu := df["requested.gres_gpu"]).notnull().all() and (
        _requested_gres_gpu >= 0
    ).all()

    df = _set_cpu_gpu_billed(df)

    df, missing_users = _find_missing_user_to_mila_emails(df)
    if missing_users:
        logger.info(f"Missing the mila email for these users: {sorted(missing_users)}")

    return df


def _fix_requested_allocated_gres_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Fix: Some jobs on Narval have requested.gres_gpu>0 but have allocated.gres_gpu=0!

    Job ids of examples: 5083814, 5083815, 5113377
    """
    # Note: This is a workaround for a bug in SARC, where some jobs have requested.gres_gpu > 0
    # but allocated.gres_gpu = 0. This is not correct, since it means that the job was not actually
    # allocated any GPUs, but it was requested.
    # So we set allocated.gres_gpu = requested.gres_gpu for those jobs.
    # mask = (df["requested.gres_gpu"] > 0) & (df["allocated.gres_gpu"] == 0)
    # Some jobs on Narval have requested.gres_gpu>0 but have allocated.gres_gpu=0!

    requested_gres_gpu = df["requested.gres_gpu"]
    allocated_gres_gpu = df["allocated.gres_gpu"]
    # If a job requested a GPU, it has to be allocated at least one GPU.
    # In general, we assume that 1 <= requested.gres_gpu <= allocated.gres_gpu
    # If 0 < requested.gres_gpu < 1, then set it to 1.0.
    # NOTE: Actually, because of MIG GPUs, we can have requested.gres_gpu < 1.0
    # so we leave it as-is.
    # requested_gres_gpu = requested_gres_gpu.where(
    #     requested_gres_gpu > 0, np.maximum(requested_gres_gpu, 1.0)
    # )

    # IDEA:
    # requested_gres_gpu = requested_gres_gpu.where(
    #     requested_gres_gpu > 0, np.maximum(requested_gres_gpu, 1.0)
    # )

    # If allocated.gres_gpu == 0 but requested.gres_gpu > 0, set it to requested.gres_gpu.
    allocated_gres_gpu = allocated_gres_gpu.mask(
        (allocated_gres_gpu == 0) & (requested_gres_gpu > 0), requested_gres_gpu
    )
    # If allocated.gres_gpu < requested.gres_gpu, set it to requested.gres_gpu.
    # allocated_gres_gpu = allocated_gres_gpu.mask(
    #     allocated_gres_gpu < requested_gres_gpu,
    #     requested_gres_gpu,
    # )
    return df.assign(
        **{
            # "requested.gres_gpu": requested_gres_gpu,
            "allocated.gres_gpu": allocated_gres_gpu,
        }
    )


def _find_missing_user_to_mila_emails(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing_mila_email = df["user.mila.email"].isna()
    missing_mila_email_users = df[missing_mila_email]["user"].unique()

    n_missing = missing_mila_email.sum()
    if not n_missing:
        return df, []

    N = df.shape[0]
    logger.warning(
        f"'user.mila.email' is missing in {n_missing} jobs ({n_missing / N:.2%}) "
        f"from {len(missing_mila_email_users)} users."
    )

    # Find jobs from the same users, where the email is not missing.
    missing_user_to_emails = df[
        df["user"].isin(missing_mila_email_users) & df["user.mila.email"].notna()
    ][["user", "user.mila.email"]]

    # NOTE: edge case here: Is it be possible for the same user to have different emails?
    # If so, here calling `dict` will use the last email found.
    user_to_email_dict = dict(
        list(missing_user_to_emails.drop_duplicates().itertuples(index=False))
    )
    # Use those to fill the missing entries.
    df.loc[missing_mila_email, "user.mila.email"] = df.loc[
        missing_mila_email, "user"
    ].map(user_to_email_dict)

    still_missing_mila_email = df["user.mila.email"].isna()
    still_missing_mila_email_users = df[still_missing_mila_email]["user"].unique()

    n_still_missing = still_missing_mila_email.sum()
    n_users_fixed = len(
        set(missing_mila_email_users) - set(still_missing_mila_email_users)
    )
    n_fixed = n_missing - n_still_missing
    logger.info(
        f"Able add the missing 'user.mila.email' for the {n_fixed} jobs from {n_users_fixed} users "
        f"({(n_fixed / n_missing):.2%} of the jobs with a missing value)."
    )
    return df, sorted(still_missing_mila_email_users)


def _replace_outlier_stats_with_na(df: pd.DataFrame):
    """`load_job_series` only removes the H100 outliers for gpu_utilization.

    Here we do the rest.
    """
    # Shouldn't really be necessary anymore, but still.
    df = df.assign(
        **{
            col: df[col].where(lambda v: (0 <= v) & (v <= 1), pd.NA)
            for col in [
                "gpu_utilization",
                *(f"gpu_utilization_fp{bits}" for bits in [16, 32, 64]),
                "gpu_memory",
                "gpu_sm_occupancy",
            ]
        },
        gpu_power=df["gpu_power"].where(lambda v: (0 <= v) & (v <= 10e10), pd.NA),
    )
    # df.loc[df["gpu_utilization"] > 1, "gpu_utilization"] = pd.NA
    # for bits in [16, 32, 64]:
    #     df.loc[df[f"gpu_utilization_fp{bits}"] > 1, f"gpu_utilization_fp{bits}"] = pd.NA
    # df.loc[df["gpu_memory"] > 1, "gpu_memory"] = pd.NA
    # df.loc[df["gpu_sm_occupancy"] > 1, "gpu_sm_occupancy"] = pd.NA
    # df.loc[df["gpu_power"] > 10e10, "gpu_power"] = pd.NA
    return df


def _fill_missing_metrics_using_means(
    df: pd.DataFrame, all_users_data: pd.DataFrame | None = None
):
    """Fill in missing JobStatistics metrics using average of available data."""
    stat_columns = list(JobStatistics.__fields__.keys())
    clusters = df["cluster_name"].unique()

    # no_na = df.dropna(subset=stat_columns, how="any")
    # assert no_na.shape[0] > 0

    across_cluster_means = {col: df[col].dropna().mean() for col in stat_columns}
    logger.debug(
        f"Mean of stats across all clusters: {_get_stats_str(across_cluster_means)}"
    )

    # todo: cpu_utilization and system_memory should be there for all jobs, right?
    gpu_columns = [col for col in stat_columns if col.startswith("gpu")]
    cpu_system_stats_columns = list(set(stat_columns) - set(gpu_columns))

    assert df["allocated.gres_gpu"].notna().all()

    # Create some masks
    has_gpu = df["allocated.gres_gpu"] > 0
    is_missing_gpu_stats = has_gpu & df[gpu_columns].isna().any(axis="columns")
    is_missing_system_stats = df[cpu_system_stats_columns].isna().any(axis="columns")

    sparsity_info_across_clusters = {
        "has_gpu": has_gpu.mean(),
        "is_missing_gpu_stats": is_missing_gpu_stats.mean(),
        "is_missing_system_stats": is_missing_system_stats.mean(),
    }
    logger.info({k: f"{v:.2%}" for k, v in sparsity_info_across_clusters.items()})

    for cluster in clusters:
        is_in_cluster = df["cluster_name"] == cluster
        cluster_mean_stats = {
            col: df[is_in_cluster][col].dropna().mean() for col in stat_columns
        }
        # Use the cluster average if possible, otherwise use the average across all clusters.
        missing_stats = [k for k, v in cluster_mean_stats.items() if np.isnan(v)]
        if missing_stats:
            logger.debug(
                f"Missing stats for {cluster=}: {missing_stats}.\n"
                f"The average of available stats across other clusters will be used."
            )
        stats_to_use = {
            col: (
                cluster_mean
                if not np.isnan(cluster_mean)
                else across_cluster_means[col]
            )
            for col, cluster_mean in cluster_mean_stats.items()
        }
        missing_stats_str = _get_stats_str(
            {k: v for k, v in stats_to_use.items() if k in missing_stats}
        )
        if missing_stats:
            logger.debug(
                f"Stats to be used when infilling missing values for {cluster}: {missing_stats_str}"
            )
        df.loc[is_in_cluster & is_missing_gpu_stats, gpu_columns] = [
            stats_to_use[col] for col in gpu_columns
        ]
        df.loc[is_in_cluster & is_missing_system_stats, cpu_system_stats_columns] = [
            stats_to_use[col] for col in cpu_system_stats_columns
        ]
    return df


def _get_stats_str(stats_to_use: Mapping[str, np.ndarray | float]):
    return {
        k: (f"{v:.1f}" if k == "gpu_power" else f"{v:.2%}")
        for k, v in stats_to_use.items()
    }


def _fix_lost_jobs(df: pd.DataFrame):
    _28_days = timedelta(days=28)
    lost_jobs = df["elapsed_time"] > _28_days.total_seconds()
    df.loc[lost_jobs, "elapsed_time"] = _28_days.total_seconds()
    df.loc[lost_jobs, "end_time"] = df.loc[lost_jobs, "start_time"] + _28_days
    return df


def _fix_unaligned_cache(df: pd.DataFrame, start: datetime, end: datetime):
    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    df = df[df["end_time"].isnull() | (df["end_time"] > start)]
    df = df[df["start_time"].notnull() & (df["start_time"] < end)]

    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    return df


def _remove_old_nodes(df: pd.DataFrame):
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


def _validate_gpu_ram():
    missing_ram = set(_gpu_name_mapping.values()) - set(_gpu_ram.keys())
    if missing_ram:
        raise ValueError(f"Missing ram: {missing_ram}")


# todo: replace with the actual `get_node_to_gpu` function once it works with the client config.
def _get_node_to_gpu(cluster_name: str):
    # node_to_gpu = get_node_to_gpu(cluster_name=cluster_name)
    # return node_to_gpu
    with open(f"{CONFIG_FOLDER}/node_to_gpu.json") as f:
        cluster_configs: dict[str, dict[str, str]] = json.load(f)
    return cluster_configs[cluster_name]


def _get_cluster_configs() -> dict[str, ClusterConfig]:
    with open(f"{CONFIG_FOLDER}/sarc-dev.json") as f:
        cluster_configs = {
            k: ClusterConfig(**v) for k, v in json.load(f)["clusters"].items()
        }
    return cluster_configs

    with open(Path(__file__).parent.parent / "config/sarc-dev.yaml") as f:
        cluster_configs = {
            k: ClusterConfig(**v)
            for k, v in yaml.safe_load(f)["sarc"]["clusters"].items()
        }
    return cluster_configs


def _fix_missing_gpu_type(df: pd.DataFrame, clusters: list[str] | None = None):
    # Fix missing gpu_type
    if not clusters:
        clusters = df["cluster_name"].unique().tolist()
    if not clusters:
        assert df.shape[0] == 0
        clusters = ALL_CLUSTERS
    assert clusters is not None and len(clusters)

    for cluster_name in clusters:
        node_to_gpu = _get_node_to_gpu(cluster_name=cluster_name)
        # node_to_gpu = get_node_to_gpu(cluster_name=cluster_name)
        assert node_to_gpu is not None
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
    ) - set(_gpu_name_mapping.keys())
    if missing_mappings:
        print("Missing mappings:", missing_mappings)
        print(
            df[df["allocated.gpu_type"].isin(missing_mappings)]
            .groupby(["allocated.gpu_type", "cluster_name"])
            .count()["id"]
        )
        breakpoint()
        # How can this produce NaNs if we made sure no GPU types were missing?!

    def _fn(x):
        if x in _gpu_name_mapping:
            return _gpu_name_mapping[x]
        elif x is None:
            return x
        else:
            logger.warning(f"Missing GPU name mapping: {x}")
            return x

    # df["allocated.gpu_type"] = df["allocated.gpu_type"].map(_gpu_name_mapping)
    df["allocated.gpu_type"] = df["allocated.gpu_type"].map(_fn)
    df.fillna({"allocated.gpu_type": "unknown"}, inplace=True)

    # ugly patch:
    unknown_gpu = df["allocated.gpu_type"] == "unknown"
    for cluster_name in clusters:
        mask = (df["cluster_name"] == cluster_name) & unknown_gpu
        df.loc[mask, "allocated.gpu_type"] = (
            df[mask]["nodes"]
            .str[0]
            .map(_get_node_to_gpu(cluster_name))
            .map(_gpu_name_mapping)
        )

    return df


def _fix_allocated_cpus_drac_inplace(df: pd.DataFrame):
    # TODO we should fix this in SARC.
    is_drac = df["cluster_name"] != "mila"
    slice_during_rgu_time = (
        is_drac
        & (df["start_time"] >= datetime(2024, 4, 1, tzinfo=MTL))
        & (df["elapsed_time"] > 0)
    )
    df.loc[slice_during_rgu_time, "allocated.cpu"] /= 1000.0

    # df.loc[df["job_id"] == 48738025, "allocated.cpu"] /= 1000
    # is_narval = df["cluster_name"] == "narval"

    # Here we do it for all timeframes.
    outrageous_num_of_cpus = df["allocated.cpu"] >= 1000
    df.loc[is_drac & outrageous_num_of_cpus, "allocated.cpu"] /= 1000.0


def _fix_rgu_discrepencies_inplace(df: pd.DataFrame) -> None:
    # NOTE: Fixing switch to RGU billing for a second time on Narval
    # narval_config = config().clusters["narval"]
    cluster_configs = _get_cluster_configs()
    narval_config = cluster_configs["narval"]

    assert df["allocated.gres_gpu"].notnull().all()
    assert df["requested.gres_gpu"].notnull().all()

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
        # Make sure that we are indeed doing this processing for each cluster.
        assert cluster_config.name
        # name = cluster_config.host
        if cluster_config.name == "mila":
            assert (
                cluster_config.rgu_start_date is None
                and cluster_config.gpu_to_rgu_billing is None
            )
        else:
            assert (
                cluster_config.rgu_start_date
                and cluster_config.gpu_to_rgu_billing
                and Path(cluster_config.gpu_to_rgu_billing).is_file()
            )
        # note: This might introduce some NANs in the `allocated.gres_gpu` for some jobs.
        update_cluster_job_series_rgu(df, cluster_config)

    # TODO: isn't this supposed to be fixed in SARC? Why do we need this mapping here?
    gpu_to_rgu_billing = {
        "a100-40gb": 700,
        "a100-40gb-3g.20gb": 1714.29 / 4000 * 700,
        "a100-40gb-4g.20gb": 2285.71 / 4000 * 700,
    }
    col_ratio_rgu_by_gpu = df.loc[slice_during_rgu_time, "allocated.gpu_type"].map(
        gpu_to_rgu_billing
    )
    df.loc[slice_during_rgu_time, "allocated.gpu_type_rgu"] = col_ratio_rgu_by_gpu
    # todo: warning about type non-compatible with int64.
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

    # Overwrite all RGU values.
    df["allocated.gpu_type_rgu"] = df["allocated.gpu_type"].map(_RGUS)


def _set_cpu_gpu_billed(stats: pd.DataFrame):
    assert (
        stats["allocated.cpu"].notna().all()
        and (stats["allocated.cpu"] > 0).all()
        # todo: some jobs have 0.001 cpus (because of the /1000).
        and (stats["allocated.cpu"] < 1000).all()
    )
    assert (
        stats["allocated.gres_gpu"].notna().all()
        and (stats["allocated.gres_gpu"] >= 0).all()
    )
    return stats.assign(
        **{
            "cpu_billed": stats["elapsed_time"] * stats["allocated.cpu"],
            "gpu_billed": stats["elapsed_time"] * stats["allocated.gres_gpu"],
        }
    )


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
    return pd.concat(data_frames, axis=0)
    return pd.concat(data_frames, axis=0)


if __name__ == "__main__":
    main()
    main()
    main()
    main()
    main()
