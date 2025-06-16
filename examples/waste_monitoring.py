from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import os
import pickle
import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, ParamSpec, Sequence, TypeVar

import numpy as np
import pandas as pd
import rich.logging
import simple_parsing
import yaml
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
)
from rich.table import Table
from typing_extensions import Self

os.environ.setdefault("SARC_CONFIG", "config/sarc-client.yaml")

from examples.compute_forecast import _setup_logging
from sarc.client.job import JobStatistics, SlurmState
from sarc.client.series import compute_cost_and_waste, load_job_series
from sarc.client.users.api import User
from sarc.config import MTL, ClusterConfig
from sarc.jobs.series import update_cluster_job_series_rgu

if (_sarc_config := os.environ.get("SARC_CONFIG")) and Path(_sarc_config).exists():
    CONFIG_FOLDER = Path(_sarc_config).parent
else:
    CONFIG_FOLDER = Path(__file__).parent / "config"

CACHE_DIR: Path | None = (
    Path(os.environ["CF_DATA"]) if "CF_DATA" in os.environ else None
)
P = ParamSpec("P")
OutT = TypeVar("OutT")


logger = logging.getLogger(__name__)
seconds_in_a_day = timedelta(days=1).total_seconds()


def main():
    _setup_logging(verbose=2)
    with Live(make_layout(), refresh_per_second=4) as live:
        for _ in range(40):
            live.update(make_layout())
            time.sleep(5)


def make_layout() -> Layout:
    """Define the layout."""
    midnight_tonight = _midnight(datetime.now() + timedelta(days=1))
    options = FilteringOptions(
        start=(midnight_tonight - timedelta(days=7)),
        end=midnight_tonight,
        user=(),
        clusters=(),
    )
    data = get_clean_sarc_data(options)
    # stats = _get_stats(data, options, frame_size="D")  # Daily

    layout = Layout(name="root")

    layout.split(
        Layout(Header(), name="header", size=3),
        Layout(name="main_top"),
        Layout(name="main_bottom"),
        Layout(name="footer", size=7),
    )
    layout["main_top"].split_row(
        Layout(name="body_left"),
        Layout(name="body_right"),
    )
    layout["body_left"].update(make_cluster_overview_table(data))
    layout["body_right"].update(make_waste_overview_table(data))
    layout["main_bottom"].update(make_biggest_waster_job_info_table(data))
    # layout["side"].split(
    #     Layout(make_avail_gpus_panel(), name="box1"),
    #     Layout(make_avg_gpu_util_panel(), name="box2"),
    # )
    return layout


def make_waste_overview_table(data: pd.DataFrame) -> Table:
    """Make a new table."""

    data_by_user = data.groupby(["cluster_name", "user.mila.email"]).aggregate(
        {
            "job_id": "nunique",
            "job_state": lambda v: (v == SlurmState.COMPLETED).mean(),
            "allocated.gres_gpu": "sum",
            "gpu_equivalent_cost": "sum",
            "cpu_equivalent_waste": "sum",
            "gpu_equivalent_waste": "sum",
            "rgu_equivalent_waste": "sum",
        },
    )
    data_by_user = data_by_user.rename(columns={"job_state": "job_success_rate"})
    ordered_by_waste = data_by_user.nlargest(
        columns="gpu_equivalent_waste",
        n=100,
        keep="all",
    )
    gpu_util_stats = data.groupby(["cluster_name", "user.mila.email"]).aggregate(
        {"gpu_utilization": "describe"}
    )

    table = Table(expand=True)
    table.add_column("User")
    table.add_column("Cluster")
    table.add_column("Job success rate", justify="right")
    table.add_column("Total GPUs", justify="right")
    table.add_column("GPU Utilization", justify="right")
    table.add_column("Used GPU*days in last 7 days", justify="right")
    table.add_column("Wasted GPU*days in last 7 days", justify="right")

    seconds_in_a_day = timedelta(days=1).total_seconds()
    for index, row in itertools.islice(ordered_by_waste.iterrows(), 20):
        assert isinstance(index, tuple) and len(index) == 2
        (cluster, user_email) = index
        used_gpus = row["allocated.gres_gpu"]
        gpu_util = gpu_util_stats.loc[index, "gpu_utilization"]

        gpu_equiv_cost = row["gpu_equivalent_cost"] / seconds_in_a_day
        gpu_equiv_waste = row["gpu_equivalent_waste"] / seconds_in_a_day
        table.add_row(
            user_email,
            cluster,
            _colorize_utilization(row["job_success_rate"], red=0.1, orange=0.2),
            str(used_gpus),
            _colorize_utilization(gpu_util["mean"]) + " ± " + f"{gpu_util['std']:.2%}",
            f"[blue]{gpu_equiv_cost:.2f}",
            f"[red]{gpu_equiv_waste:.2f}",
        )
    return table


def make_cluster_overview_table(data: pd.DataFrame) -> Table:

    total_mila_users = data["user.mila.email"].nunique()
    grouped_data = data.groupby("cluster_name").aggregate(
        {
            "job_id": "nunique",
            "user.mila.email": "nunique",
            "gpu_utilization": "mean",
        }
    )

    # todo: Somehow get the total number of GPUs on each cluster.
    # _n_clusters = data["cluster_name"].unique()
    # total_gpus = np.random.randint(500, 1000, size=len(_n_clusters))
    # avail_gpus = np.random.randint(0, total_gpus, size=len(_n_clusters))

    table = Table(expand=True)
    # TODO: Show Min / Mean / Median / Max GPUs per user?
    table.add_column("Cluster", justify="left")
    table.add_column("# of jobs", justify="right")
    # table.add_column("STD of GPUs per user", justify="right")
    table.add_column("Average GPU Utilization")
    table.add_column("Mila students using this cluster", justify="right")

    for index, row in grouped_data.iterrows():
        assert isinstance(index, str)
        cluster = index
        # used_gpus_pct = avail_gpu / total_gpu
        num_jobs = row["job_id"]
        gpu_util = row["gpu_utilization"]
        mila_users = row["user.mila.email"]
        pct_of_mila_users = mila_users / total_mila_users
        table.add_row(
            cluster,
            str(num_jobs),
            # f"{avail_gpu} / {total_gpu} ({used_gpus_pct:.2%})",
            _colorize_utilization(gpu_util),
            f"{mila_users} / {total_mila_users} ({pct_of_mila_users:.2%})",
        )
    return table


def make_biggest_waster_job_info_table(data: pd.DataFrame) -> Table:

    # Mock data (TODO: replace)
    most_wasteful_jobs = data.nlargest(n=20, columns="rgu_equivalent_waste", keep="all")

    table = Table(title="Biggest wasting individual jobs", expand=True)
    table.add_column("Job ID", justify="right")
    table.add_column("Cluster", justify="left")
    table.add_column("User", justify="left")
    table.add_column("Elapsed time (hours)", justify="left")
    table.add_column("Wasted GPU*hours", justify="right")
    table.add_column("Average GPU Utilization", justify="right")
    table.add_column("Workdir", justify="left")
    table.add_column("Requested Ressources", justify="right")
    # table.add_column("submit command", justify="left")

    for index, row in most_wasteful_jobs.iterrows():
        requested_cols = [
            col for col in most_wasteful_jobs.columns if col.startswith("requested.")
        ]
        requested_resources = {
            k.removeprefix("requested."): (
                row[k] if k != "mem" else f"{row[k]//1024}GB"
            )
            for k in requested_cols
        }

        table.add_row(
            str(row["job_id"]),
            row["cluster_name"],
            row["user.mila.email"],
            f"{row['elapsed_time'] / timedelta(hours=1).total_seconds():.1f}",
            f"[red]{row['rgu_equivalent_waste']/seconds_in_a_day:.2f}",
            _colorize_utilization(row["gpu_utilization"]),
            row["work_dir"],
            " ".join(f"{k}={v}" for k, v in requested_resources.items() if v),
        )
    return table


def _colorize_utilization(util: float, red: float = 0.2, orange=0.5) -> str:
    """Colorize the (GPU/CPU/whatever) utilization value based on thresholds."""
    if np.isnan(util):
        return f"[bold red]{util}"
    assert 0 <= util <= 1, "utilization must be between 0 and 1"
    util_pct = f"{util:.2%}"
    if util < red:
        return f"[red]{util_pct}"
    elif util < orange:
        return f"[yellow]{util_pct}"
    else:
        return f"[green]{util_pct}"


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

    user: Sequence[str] = dataclasses.field(default_factory=tuple)
    """ Which user(s) to query information for. Leave blank to get a global compute profile."""

    users_file: Path | None = dataclasses.field(default=None, repr=False)

    clusters: Sequence[str] = dataclasses.field(default_factory=tuple)
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


def cached(fn: Callable[P, OutT]) -> Callable[P, OutT]:
    """Caches a function in a given cache dir."""
    if CACHE_DIR is not None:
        cache_dir = CACHE_DIR
    else:
        parser = argparse.ArgumentParser(add_help=False)
        default_cache_dir = Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
        parser.add_argument("--cache_dir", type=Path, default=default_cache_dir)
        cache_dir: Path = parser.parse_known_args()[0].cache_dir

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> OutT:
        """Decorator to cache the results of a function."""
        # TODO: in _get_cache_file_name, use a .txt extension if the return annotation is `str`
        cache_file = cache_dir / _get_cache_file_name(fn, *args, **kwargs)
        if cache_file.exists():
            logger.info(f"Loading result of {fn.__name__} from {cache_file}")
            return pickle.loads(cache_file.read_bytes())
        else:
            logger.debug(f"Cache miss for {fn.__name__} at {cache_file}")
            result = fn(*args, **kwargs)
            # TODO: Save to a text file if the result is a string.
            cache_file.write_bytes(pickle.dumps(result))
            logger.info(f"Saved result of computing {fn.__name__} to {cache_file}")
            return result

    return wrapper


def _get_cache_file_name(
    fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
) -> str:
    # More interpretable than using this:
    # return hashlib.md5(
    #     json.dumps((fn.__name__, args, kwargs), sort_keys=True, default=str).encode()
    # ).hexdigest()

    def _hash(v) -> str:
        if isinstance(v, pd.DataFrame):
            # Important: Assuming that the other function arguments will be used
            # to recover the same dataframe, so not including it in the hash.
            return ""
        if v is None:
            return str(v)
        if isinstance(v, str):
            return v.removesuffix("@mila.quebec")  # no quotes around strings.
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, datetime):
            if v.hour == 0 and v.minute == 0 and v.second == 0:
                return v.strftime("%Y-%m-%d-%z")
            return v.strftime("%Y-%m-%dT%H:%M:%S%z")
        if isinstance(v, list):
            # Some profs have so many students that we can't concat them.
            if v and isinstance(v[0], User):
                return hashlib.md5(
                    "+".join(sorted(student.mila.username for student in v)).encode()
                ).hexdigest()[:12]
            return "+".join(sorted(map(_hash, v)))
        if isinstance(v, User):
            return v.mila.username
        if isinstance(v, FilteringOptions):
            return (
                v.unique_path()
                .relative_to(v.cache_dir)
                .stem.removeprefix("compute_profile-")
            )
        raise NotImplementedError(f"Unsupported arg type: {v} of type {type(v)}")

    hashed_args = "-".join(map(_hash, args)) + "-".join(
        f"{k}-{_hash(v)}" for k, v in kwargs.items()
    )
    return f"{fn.__name__}-{hashed_args}.pkl"


@functools.lru_cache(maxsize=1)  # cache results in memory
# @cached  # Cache results to a file
def get_clean_sarc_data(options: FilteringOptions) -> pd.DataFrame:
    """Gets "cleaned" SARC data for a given period, including *lots* of patches."""
    options = dataclasses.replace(
        options,
        start=options.start.astimezone(MTL),
        end=options.end.astimezone(MTL),
    )
    logger.debug(
        f"Looking up for data between {options.start} and {options.end} for all users on all clusters."
    )

    # In SARC we currently can't query by user.mila.email, so we query with all
    # usernames and filter by user.mila.email after.
    # Cache results of SARC query to a file.
    df = cached(load_job_series)(
        start=options.start,
        end=options.end,
        clip_time=False,
    )

    if df.empty:
        raise RuntimeError(f"NO SARC data for that period: {options}")

    for time_column in ["submit_time", "start_time", "end_time"]:
        # df[time_column] = df[time_column].dt.tz_localize("UTC").dt.tz_convert(MTL)
        df[time_column] = df[time_column].dt.tz_convert(MTL)

    _validate_gpu_ram()

    # Clusters we want to compare
    if options.clusters:
        # Filter clusters
        df = df[df["cluster_name"].isin(options.clusters)]

    df = df.fillna({"requested.gres_gpu": 0.0, "allocated.gres_gpu": 0.0})
    df = _fix_lost_jobs(df)
    df = _fix_unaligned_cache(df, options.start, options.end)
    df = _remove_old_nodes(df)
    df = _replace_outlier_stats_with_na(df)
    df = _fix_missing_gpu_type(df)

    _fix_rgu_discrepencies_inplace(df)

    # todo: double-check if this is still needed here.
    df = df.fillna({"requested.gres_gpu": 0, "allocated.gres_gpu": 0})

    df = _fix_allocated_cpus_drac(df)
    # todo: Do we want to get the averages from the other jobs of the same user on other clusers?
    # Or from the average utilization on that same cluster by different users?
    # IF so, we might need to reload the data for all users here
    # if users := options.get_users():
    #     all_users_data_for_same_period = _get_cleaned_df(
    #         dataclasses.replace(options, user=[], user_file=None)
    #     )
    df = _fix_requested_allocated_gres_gpu(df)
    # TODO: Turning this off for now. Causes bugs with gpu_utilization, apparently.
    # df = _fill_missing_metrics_using_means(df)

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

    df, missing_users = _find_missing_user_to_mila_emails(df)
    if missing_users:
        logger.info(f"Missing the mila email for these users: {sorted(missing_users)}")
    df = df.assign(
        rgu_equivalent_cost=(df["gpu_equivalent_cost"] * df["allocated.gpu_type_rgu"]),
        rgu_equivalent_waste=(
            df["gpu_equivalent_waste"] * df["allocated.gpu_type_rgu"]
        ),
    )
    return df


def _get_stats(
    sarc_data: pd.DataFrame,
    options: FilteringOptions,
    frame_size: timedelta | str | None = None,
) -> pd.DataFrame:
    stats = compute_time_frames(
        sarc_data,
        [
            "gpu_cost",
            "cpu_cost",
            "cpu_equivalent_cost",
            "gpu_equivalent_cost",
            "rgu_equivalent_cost",  # todo: double-check that this gets split up correctly.
            "cpu_equivalent_waste",
            "gpu_equivalent_waste",
        ],
        start=options.start,
        end=options.end,
        frame_size=(
            frame_size
            if frame_size is not None
            else (
                "MS"
                if (_period := (options.end - options.start)) > timedelta(days=90)
                else (
                    timedelta(days=7)
                    if _period > timedelta(days=30)
                    else timedelta(days=1)
                )
            )
        ),
    )

    return stats


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
        if missing_stats:
            missing_stats_str = _get_stats_str(
                {k: v for k, v in stats_to_use.items() if k in missing_stats}
            )
            n_to_fill = (is_in_cluster & is_missing_gpu_stats).sum()
            logger.debug(
                f"Stats to be used when infilling missing values for {n_to_fill} GPU jobs on {cluster}: {missing_stats_str}"
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


def _check_is_email_and_lower(v: str):
    if not v:
        return v
    if "@" not in v or v.count("@") != 1:
        raise ValueError(f"'{v}' is not a valid email address.")

    return v.lower()


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


def _fix_allocated_cpus_drac(df: pd.DataFrame):
    """Fix for some issue where some jobs on Drac have allocated.cpu > 1000 because of something related to RGUs.

    Example job id: 48738025 (on Narval I think).
    """
    is_drac = df["cluster_name"] != "mila"
    slice_during_rgu_time = (
        is_drac
        & (df["start_time"] >= datetime(2024, 4, 1, tzinfo=MTL))
        & (df["elapsed_time"] > 0)
    )
    allocated_cpu = df["allocated.cpu"]
    df = df.assign(
        **{
            "allocated.cpu": allocated_cpu.mask(
                # TODO: Should we also check if it's a multiple of 1000 before dividing?
                slice_during_rgu_time & (allocated_cpu >= 1000),
                allocated_cpu / 1000,
            )
        }
    )

    # df.loc[df["job_id"] == 48738025, "allocated.cpu"] /= 1000
    # is_narval = df["cluster_name"] == "narval"

    # Here we do it again but for all timeframes.
    # TODO: Why not just do it once?
    allocated_cpu = df["allocated.cpu"]
    outrageous_num_of_cpus = allocated_cpu >= 1000
    df = df.assign(
        **{
            "allocated.cpu": allocated_cpu.mask(
                is_drac & outrageous_num_of_cpus, allocated_cpu / 1000
            )
        }
    )
    return df
    # df.loc[slice_during_rgu_time, "allocated.cpu"] /= 1000.0

    # df.loc[df["job_id"] == 48738025, "allocated.cpu"] /= 1000
    # is_narval = df["cluster_name"] == "narval"

    # Here we do it for all timeframes.
    # df.loc[is_drac & outrageous_num_of_cpus, "allocated.cpu"] /= 1000.0


def _fix_rgu_discrepencies_inplace(df: pd.DataFrame) -> None:
    # NOTE: Fixing switch to RGU billing for a second time on Narval
    cluster_configs = _get_cluster_configs()
    narval_config = cluster_configs["narval"]

    assert df["allocated.gres_gpu"].notnull().all()
    assert df["requested.gres_gpu"].notnull().all()

    is_narval = df["cluster_name"] == "narval"
    slice_during_rgu_time = (
        is_narval
        & (df["start_time"] >= datetime(2023, 11, 28, tzinfo=MTL))
        & (
            df["start_time"]
            < datetime.fromisoformat(narval_config.rgu_start_date).astimezone(MTL)
        )
        & (df["elapsed_time"] > 0)
    )
    non_updated_df = df[slice_during_rgu_time]

    # TODO: Seems like part of this patching is supposed to be done in SARC with the `update_job_series_rgu` function.
    # However, we can't call that function here atm because it requires using the sarc-dev config.
    # update_job_series_rgu
    # df = update_job_series_rgu(df)
    # return df
    for cluster_config in cluster_configs.values():
        # Make sure that we are indeed doing this processing for each cluster.
        assert cluster_config.name
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

    # TODO: Why do we need this mapping here?
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
        df[slice_during_rgu_time]["allocated.gres_gpu"] / col_ratio_rgu_by_gpu
    )
    # Overwrite all RGU values
    # TODO: Why?!
    df["allocated.gpu_type_rgu"] = df["allocated.gpu_type"].map(_RGUS)


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


def _setup_logging(verbose: int):
    logging.basicConfig(
        handlers=[rich.logging.RichHandler(show_time=False)],
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


_gpu_name_mapping = {
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
    # NOTE: Added for narval. Might be fixed with `get_node_to_gpu`, unclear.
    "a100_1g.5gb": "a100-weird",
    "1g.5gb": "a100-weird",
}

_gpu_ram = {
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
    # NOTE: Added for narval. Might be fixed with `get_node_to_gpu`, unclear.
    "a100-weird": 5,
}

_RGUS = {
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


if __name__ == "__main__":
    main()
