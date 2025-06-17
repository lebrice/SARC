from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import itertools
import logging
import os
import pickle
import random
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, ParamSpec, Sequence, TypeVar, get_type_hints
import typing

import gifnoc
import numpy as np
import pandas as pd
import paramiko
import paramiko.config
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
from simple_parsing.helpers.serialization.serializable import from_dict
from typing_extensions import Self

sarc_client_config_file = Path(__file__).parent.parent / "config/sarc-client.yaml"
sarc_dev_config_file = Path(__file__).parent.parent / "config/sarc-dev.yaml"
if sarc_client_config_file.exists():
    # This script is being executed either from the SARC root, or maybe from an editable install
    # of the SARC package.
    assert sarc_dev_config_file.exists()
elif (
    other_possible_config_path := sarc_client_config_file.parent.parent
    / "sarc"
    / "config"
    / sarc_client_config_file.name
).exists():
    # SARC was installed as a package, and the package-data was included a the path `sarc/config`
    # (via `tool.hatch.build.targets.wheel.force-include`), so the sarc configs are actually now
    # inside SARC (instead of being a separate package in the site-packages directory).
    sarc_client_config_file = other_possible_config_path
    sarc_dev_config_file = sarc_client_config_file.parent / sarc_dev_config_file.name

assert sarc_client_config_file.exists(), sarc_client_config_file
assert sarc_dev_config_file.exists(), sarc_dev_config_file

# TODO: Important to do this BEFORE importing anything from SARC.
# Otherwise it doesn't work.
# This also prevents this from being a "script" in `pyproject.toml` placed under `sarc/cli`
# because the executable that gets placed in `.venv/bin/waste_monitor` would do
# `from sarc.cli.waste_monitor import main`, and the `sarc.cli.__init__.py` causes the
# config to be instantiated, so the script can't work.
os.environ.setdefault("SARC_CONFIG", str(sarc_client_config_file))

import sarc.config
from sarc.alerts.common import HealthMonitorConfig
from sarc.client.job import JobStatistics, SlurmState
from sarc.client.series import (
    compute_cost_and_waste,
    load_job_series,
    update_cluster_job_series_rgu,
    update_job_series_rgu,
)
from sarc.client.users.api import User
from sarc.config import MTL, ClientConfig, ClusterConfig, Config
from sarc.jobs.node_gpu_mapping import get_node_to_gpu

# TODO: Idea: reload the config module?
# importlib.reload(sarc.config)  # Reload the config module to use the new config file.

sarc_client_config = from_dict(
    ClientConfig, yaml.safe_load(sarc_client_config_file.read_text())["sarc"]
)
# sarc_dev_config = from_dict(
#     Config, yaml.safe_load(sarc_dev_config_file.read_text())["sarc"]
# )
# assert False, sarc_client_config

CACHE_DIR: Path | None = (
    Path(os.environ["CF_DATA"]) if "CF_DATA" in os.environ else None
)
P = ParamSpec("P")
OutT = TypeVar("OutT")


logger = logging.getLogger(__name__)


def main():
    cached(setup_sarc_connection)()

    _setup_logging(verbose=2)
    with Live(make_layout(), refresh_per_second=4) as live:
        for _ in range(40):
            live.update(make_layout())
            time.sleep(5)


def setup_sarc_connection():
    ssh_config = paramiko.config.SSHConfig.from_path(Path.home() / ".ssh" / "config")
    mila_config = ssh_config.lookup("mila")
    if "user" not in mila_config:
        raise ValueError(
            "You need to have a `mila` entry in your SSH configuration file."
        )
    mila_user: str = mila_config["user"]
    control_socket_path = (
        Path(
            ssh_config.lookup("sarc").get(
                "controlpath", Path.home() / ".cache" / "ssh" / "%r@%h:%p"
            )
        )
        .expanduser()
        .resolve()
    )
    control_socket_path.parent.mkdir(parents=True, exist_ok=True)
    multiplexing_args = (
        f"-o ControlMaster=auto "
        f"-o 'ControlPath={control_socket_path}' "
        f"-o ControlPersist=yes"
    )

    sarc_client_connection_string = yaml.safe_load(sarc_client_config_file.read_text())[
        "sarc"
    ]["mongo"]["connection_string"]
    assert isinstance(sarc_client_connection_string, str)
    # "mongodb://readuser:readpwd@localhost:8123/sarc" --> "8123"
    sarc_client_local_port = (
        sarc_client_connection_string.rpartition("@")[2]
        .partition(":")[2]
        .partition("/")[0]
    )

    sarc_dev_connection_string = yaml.safe_load(sarc_dev_config_file.read_text())[
        "sarc"
    ]["mongo"]["connection_string"]
    assert isinstance(sarc_dev_connection_string, str)
    # "mongodb://localhost:27017/sarc-dev" --> "27017"
    sarc_dev_remote_port = (
        sarc_dev_connection_string.rpartition("@")[
            2
        ]  # might return the whole string if there is no user:pwd@...
        .partition("://")[2]  # "localhost:27017/sarc-dev"
        .partition("/")[0]  # "localhost:27017"
        .partition(":")[2]  # "27017"
    )
    assert sarc_client_local_port
    assert sarc_dev_remote_port

    port_forwarding_args = (
        f"-o 'LocalForward={sarc_client_local_port} 127.0.0.1:{sarc_dev_remote_port}'"
    )

    subprocess.check_call(
        shlex.split(
            f"ssh -o ProxyJump=mila {port_forwarding_args} -o User={mila_user} {multiplexing_args} sarc01-dev echo OK"
        )
    )


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
        Layout(name="footer", size=2),
    )
    layout["main_top"].split_row(
        Layout(name="body_left"),
        Layout(name="body_right", ratio=2),
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
            "allocated.gres_rgu": "sum",
            "gpu_equivalent_cost": "sum",
            "rgu_equivalent_cost": "sum",
            "cpu_equivalent_waste": "sum",
            "gpu_equivalent_waste": "sum",
            "rgu_equivalent_waste": "sum",
            "gpu_overbilling_cost": "sum",
            "rgu_overbilling_cost": "sum",
        },
    )
    data_by_user = data_by_user.rename(columns={"job_state": "job_success_rate"})
    ordered_by_waste = data_by_user.nlargest(
        columns="rgu_equivalent_waste",
        n=100,
        keep="all",
    )
    gpu_util_stats = data.groupby(["cluster_name", "user.mila.email"]).aggregate(
        {"gpu_utilization": "describe"}
    )

    table = Table(title="Most wasteful users (last 7 days)", expand=True)
    table.add_column("User")
    table.add_column("Cluster")
    table.add_column("GPU Utilization", justify="right")
    table.add_column("Job success rate", justify="right")
    table.add_column("Total allocated GPUs/RGUs", justify="right")
    table.add_column("Used/Wasted/Obstructed GPU days")
    table.add_column("U/W/Obs RGU*days")

    for index, row in itertools.islice(ordered_by_waste.iterrows(), 20):
        assert isinstance(index, tuple) and len(index) == 2
        (cluster, user_email) = index
        used_gpus = row["allocated.gres_gpu"]
        gpu_util = gpu_util_stats.loc[index, "gpu_utilization"]

        table.add_row(
            user_email,
            cluster,
            f"{_colorize_utilization(gpu_util['mean'])} ± {gpu_util['std']:.1%}",
            _colorize_utilization(row["job_success_rate"], red=0.1, orange=0.2),
            f"{round(row['allocated.gres_gpu'])} / {round(row['allocated.gres_rgu'])}",
            f"[green]{row['gpu_equivalent_cost'].days}[/green] / [red]{row['gpu_equivalent_waste'].days}[/red] / [red]{row['gpu_overbilling_cost'].days}[/red]",
            f"[green]{row['rgu_equivalent_cost'].days}[/green] / [red]{row['rgu_equivalent_waste'].days}[/red] / [red]{row['rgu_overbilling_cost'].days}[/red]",
            # f"[red] {row['gpu_equivalent_waste'].days:.2f} / {row['rgu_equivalent_waste'].days:.2f}",
        )
    return table


def make_cluster_overview_table(data: pd.DataFrame) -> Table:
    total_mila_users = int(data["user.mila.email"].nunique())
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

    table = Table(title="Overview by cluster (last 7 days)", expand=True)
    # TODO: Show Min / Mean / Median / Max GPUs per user?
    table.add_column("Cluster", justify="left")
    table.add_column("# of jobs", justify="right")
    table.add_column("Average GPU Utilization")
    table.add_column("Mila students using this cluster", justify="right")
    table.add_column("GPUs per user", justify="right")
    gpus_per_cluster_per_user = data.groupby(["cluster_name", "user.mila.email"])[
        ["allocated.gres_gpu"]
    ].sum()
    gpus_per_user = gpus_per_cluster_per_user.groupby("cluster_name").describe()

    for index, row in grouped_data.iterrows():
        assert isinstance(index, str)
        cluster = index

        mila_users = int(row["user.mila.email"])

        # used_gpus_pct = avail_gpu / total_gpu
        num_jobs = row["job_id"]
        gpu_util = row["gpu_utilization"]
        pct_of_mila_users = mila_users / total_mila_users

        gpus_per_user_here = gpus_per_user.loc[cluster, "allocated.gres_gpu"]
        gpus_per_user_str = (
            f"min={gpus_per_user_here['min']} mean={gpus_per_user_here['mean']:.1f} "
            f"std={gpus_per_user_here['std']:.1f} max={gpus_per_user_here['max']}"
        )
        table.add_row(
            cluster,
            str(num_jobs),
            # f"{avail_gpu} / {total_gpu} ({used_gpus_pct:.2%})",
            _colorize_utilization(gpu_util),
            f"{mila_users} / {total_mila_users} ({pct_of_mila_users:.2%})",
            gpus_per_user_str,
        )
    return table


def make_biggest_waster_job_info_table(data: pd.DataFrame) -> Table:
    # Mock data (TODO: replace)
    most_wasteful_jobs = data.nlargest(n=20, columns="rgu_equivalent_waste", keep="all")
    table = Table(
        title="Most wasteful jobs (last 7 days, all clusters combined)", expand=True
    )
    table.add_column("Job ID", justify="right")
    table.add_column("Cluster", justify="left")
    table.add_column("User", justify="left")
    table.add_column("Elapsed time", justify="left")
    table.add_column("Avg GPU Util", justify="right")
    table.add_column("Requested Ressources", overflow="fold", max_width=30)
    # TODO: Have the interval be displayed with the unit selected dynamically instead (e.g. "days" or "hours")
    table.add_column("Used/Wasted/Obstructed GPU days", justify="right")
    # table.add_column("Wasted GPU/RGU days", justify="right")
    # table.add_column("Obstructed GPU days", justify="right")
    table.add_column("SubmitLine", overflow="crop", ratio=5)
    # table.add_column("Workdir", justify="left")
    # table.add_column("submit command", justify="left")

    for index, row in most_wasteful_jobs.iterrows():
        requested_cols = [
            col for col in most_wasteful_jobs.columns if col.startswith("requested.")
        ]
        submit_line = get_submit_line(
            job_id=row["job_id"], cluster_name=row["cluster_name"]
        )
        submit_line = "\n".join(
            line.strip() for line in submit_line.splitlines() if line.strip()
        )

        requested_resources = {
            k.removeprefix("requested."): (
                # TODO: Colorize the mem based on mem per GPU ratio?
                f"{row[k] // 1024}GB"
                if k.endswith("mem")
                else (
                    f"[bold]{row['allocated.gpu_type']}:{int(row[k])}[/bold]"
                    if k.endswith("gres_gpu")
                    else str(row[k])
                )
            )
            for k in requested_cols
        }
        # IDEA: Use another (nested) table: https://stackoverflow.com/questions/74144874/constructing-a-table-with-multilevel-headers-using-rich-table
        # _requested_table = Table(
        #     padding=(0, 0),
        #     show_edge=False,
        #     show_lines=True,
        #     show_header=False,
        # )
        # _requested_table.add_column("cpus")
        # _requested_table.add_column("mem")
        # _requested_table.add_column("nodes")
        # _requested_table.add_column("nodes")
        # _requested_table.add_column("gres_gpu")
        # _requested_table.add_row(
        #     requested_resources["cpu"],
        #     requested_resources["mem"],
        #     requested_resources["node"],
        #     requested_resources["gres_gpu"],
        # )
        table.add_row(
            str(row["job_id"]),
            row["cluster_name"],
            row["user.mila.email"].removesuffix("@mila.quebec"),
            f"{row['elapsed_time']}",
            _colorize_utilization(row["gpu_utilization"]),
            # _requested_table,
            " ".join(f"{k}={v}" for k, v in requested_resources.items() if v),
            f"{row['gpu_equivalent_cost'].days} / [red]{row['gpu_equivalent_waste'].days}[/] / [red]{row['gpu_overbilling_cost'].days}",
            # f"[red]{row['gpu_equivalent_waste'].days} / {row['rgu_equivalent_waste'].days}",
            # f"[red]{row['gpu_overbilling_cost'].days}",
            submit_line,
            # IDEA: Show it as a Syntax block:
            # rich.syntax.Syntax(submit_line, lexer="bash"),
        )
    return table


def _colorize_utilization(util: float, red: float = 0.2, orange=0.5) -> str:
    """Colorize the (GPU/CPU/whatever) utilization value based on thresholds."""
    if np.isnan(util):
        return f"[bold red]{util}"
    assert 0 <= util <= 1, "utilization must be between 0 and 1"
    util_pct = f"{util:.1%}"
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
            if cache_file.suffix == ".txt":
                result = cache_file.read_text()
                # the function returns a string (`OutT` is `str`)
                return typing.cast(OutT, result)
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
        match v:
            case FilteringOptions():
                return "-".join(
                    [
                        _hash(v.get_users()),
                        _hash(v.start),
                        _hash(v.end),
                        _hash(v.clusters),
                    ]
                )
                #     .relative_to(v.cache_dir)
                #     .stem.removeprefix("compute_profile-")
                # )
            # case pd.DataFrame():
            #     # Important: Assuming that the other function arguments will be used
            #     # to recover the same dataframe, so not including it in the hash.
            #     return ""
            case None | str():
                return str(v).removesuffix("@mila.quebec")  # no quotes around strings.
            case int() | float():
                return repr(v)
            case datetime(hour=0, minute=0, second=0, tzinfo=MTL) as d:
                return d.strftime("%Y-%m-%d")
            case datetime() as v:
                return v.strftime("%Y-%m-%dT%H:%M:%S%z")
            case [User(), *_]:
                return _hash([student.mila.username for student in v])
            case [str(), *_] if len(v) > 2:
                # If there are more than 3 strings, hash them together.
                return hashlib.md5("+".join(sorted(v)).encode()).hexdigest()[:12]
            case list():
                return "+".join(sorted(map(_hash, v)))
            # case User():
            #     return v.mila.username
            case _:
                raise NotImplementedError(
                    f"Unsupported arg type: {v} of type {type(v)}"
                )

    hashed_args = "-".join(map(_hash, args)) + "-".join(
        f"{k}-{_hash(v)}" for k, v in kwargs.items()
    )
    extension = ".pkl"
    try:
        if typing.get_type_hints(fn).get("return") is str:
            extension = ".txt"
    except TypeError:
        pass
    # extension = ".pkl" if typing.get_type_hints(fn)["return"]
    return f"{fn.__name__}-{hashed_args}.{extension}"


@functools.lru_cache(maxsize=None)
@cached
def get_submit_line(job_id: int, cluster_name: str) -> str:
    ssh_config = paramiko.config.SSHConfig.from_path(Path.home() / ".ssh" / "config")
    control_path = (
        Path(
            ssh_config.lookup(cluster_name).get(
                "controlpath", Path.home() / ".cache" / "ssh" / "%r@%h:%p"
            )
        )
        .expanduser()
        .resolve()
    )
    control_path.parent.mkdir(parents=True, exist_ok=True)
    multiplexing_args = (
        f"-o ControlMaster=auto -o 'ControlPath={control_path}' -o ControlPersist=yes"
    )
    # Need to first establish the multiplexed SSH connection to the cluster, in case it uses 2FA.
    # If we didn't and used a single command, the login banner / 2FA message on DRAC would be
    # also included in the output of the command.
    subprocess.check_call(
        shlex.split(f"ssh {multiplexing_args} {cluster_name} echo 'OK'")
    )
    return subprocess.getoutput(
        f"ssh {multiplexing_args} {cluster_name} sacct -j {job_id} --noheader -o submitline%300"
    ).strip()


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

    # TODO: I guess we won't be able to apply this, unless we inline the cluster configs.
    df = _fix_rgu_discrepencies_inplace(df)

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

    # TODO: Use timedelta as the dtype for the cost / waste / overbilled columns.
    df = df.assign(
        **{
            col: pd.to_timedelta(df[col], unit="s")
            for col in df.columns
            if col.endswith(("_cost", "_waste")) or col == "elapsed_time"
        }
    )

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
        **{
            "allocated.gres_rgu": df["allocated.gres_gpu"]
            * df["allocated.gpu_type_rgu"]
        }
    )

    df = df.assign(
        rgu_equivalent_cost=(df["gpu_equivalent_cost"] * df["allocated.gpu_type_rgu"]),
        rgu_equivalent_waste=(
            df["gpu_equivalent_waste"] * df["allocated.gpu_type_rgu"]
        ),
        rgu_overbilling_cost=(
            df["gpu_overbilling_cost"] * df["allocated.gpu_type_rgu"]
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
    # BUG: Times out atm, so skipping this to make it faster.
    try:
        return get_node_to_gpu(cluster_name=cluster_name)
    except (
        gifnoc.proxy.MissingConfigurationError,
        pymongo.errors.OperationFailure,
        pymongo.errors.ServerSelectionTimeoutError,
    ):
        return _NODE_TO_GPU[cluster_name]


def _get_cluster_configs() -> dict[str, ClusterConfig]:
    cluster_configs = {
        k: ClusterConfig(**v)
        for k, v in yaml.safe_load(sarc_dev_config_file.read_text())["sarc"][
            "clusters"
        ].items()
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


def _fix_rgu_discrepencies_inplace(df: pd.DataFrame) -> pd.DataFrame:
    narval_rgu_start_date = datetime(year=2023, month=11, day=28, tzinfo=MTL)
    _beluga_rgu_start_date = datetime(year=2024, month=4, day=3, tzinfo=MTL)
    _graham_rgu_start_date = datetime(year=2024, month=4, day=3, tzinfo=MTL)
    _cedar_rgu_start_date = datetime(year=2024, month=4, day=3, tzinfo=MTL)
    # TODO: Clearly define the context for each patch.
    _patch_context = FilteringOptions(
        start=datetime(2023, 11, 28, tzinfo=MTL),
        end=narval_rgu_start_date,
    )

    # NOTE: Fixing switch to RGU billing for a second time on Narval
    # TODO: Unclear if this is fixed by the `update_job_series_rgu` function in SARC.
    # with sarc.config.using_sarc_mode("scraping"):
    return update_job_series_rgu(df)

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
    for name, cluster_config in cluster_configs.items():
        # Make sure that we are indeed doing this processing for each cluster.
        assert name
        if name == "mila":
            assert (
                cluster_config.rgu_start_date is None
                and cluster_config.gpu_to_rgu_billing is None
            )
        else:
            assert (
                cluster_config.rgu_start_date and cluster_config.gpu_to_rgu_billing
                # and Path(cluster_config.gpu_to_rgu_billing).is_file()
            ), (name, cluster_config)
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
        non_updated_df["allocated.gres_gpu"] / col_ratio_rgu_by_gpu
    )
    # Overwrite all RGU values
    # TODO: Why?!
    df["allocated.gpu_type_rgu"] = df["allocated.gpu_type"].map(_RGUS)
    return df


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

_NODE_TO_GPU = {
    "beluga": {
        "bg11201": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11202": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11203": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11204": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11205": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11206": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11207": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11208": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11209": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11210": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11211": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11212": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11213": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11214": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11301": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11302": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11303": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11304": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11305": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11306": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11307": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11308": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11309": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11310": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11311": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11312": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11313": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11401": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11402": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11403": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11404": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11405": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11406": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11407": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11408": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11409": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11410": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11411": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11412": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11413": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11414": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11501": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11502": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11503": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11504": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11505": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11506": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11507": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11508": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11509": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11510": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11511": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11512": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11513": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11601": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11602": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11603": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11604": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11605": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11606": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11607": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11608": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11609": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11610": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11611": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11612": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11613": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11614": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11701": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11702": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11703": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11704": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11705": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11706": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11707": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11708": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11709": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11710": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11711": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11712": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11713": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11801": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11802": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11803": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11804": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11805": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11806": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11807": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11808": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11809": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11810": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11811": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11812": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11813": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11814": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11901": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11902": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11903": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11904": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11905": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11906": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11907": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11908": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11909": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11910": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11911": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11912": "gpu:tesla_v100-sxm2-16gb:4",
        "bg11913": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12001": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12002": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12003": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12004": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12005": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12006": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12007": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12008": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12009": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12010": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12011": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12012": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12013": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12014": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12101": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12102": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12103": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12104": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12105": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12106": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12107": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12108": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12109": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12110": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12111": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12112": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12113": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12201": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12202": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12203": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12204": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12205": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12206": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12207": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12208": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12209": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12210": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12211": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12212": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12213": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12214": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12301": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12302": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12303": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12304": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12305": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12306": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12307": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12308": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12309": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12310": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12311": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12312": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12313": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12401": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12402": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12403": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12404": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12405": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12406": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12407": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12408": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12409": "gpu:tesla_v100-sxm2-16gb:4",
        "bg12410": "gpu:tesla_v100-sxm2-16gb:4",
    },
    "narval": {
        "ng10101": "gpu:a100:4",
        "ng10102": "gpu:a100:4",
        "ng10103": "gpu:a100:4",
        "ng10104": "gpu:a100:4",
        "ng10201": "gpu:a100:4",
        "ng10202": "gpu:a100:4",
        "ng10203": "gpu:a100:4",
        "ng10204": "gpu:a100:4",
        "ng10301": "gpu:a100:4",
        "ng10302": "gpu:a100:4",
        "ng10303": "gpu:a100:4",
        "ng10304": "gpu:a100:4",
        "ng10401": "gpu:a100:4",
        "ng10402": "gpu:a100:4",
        "ng10403": "gpu:a100:4",
        "ng10404": "gpu:a100:4",
        "ng10501": "gpu:a100:4",
        "ng10502": "gpu:a100:4",
        "ng10503": "gpu:a100:4",
        "ng10504": "gpu:a100:4",
        "ng10601": "gpu:a100:4",
        "ng10602": "gpu:a100:4",
        "ng10603": "gpu:a100:4",
        "ng10604": "gpu:a100:4",
        "ng10605": "gpu:a100:4",
        "ng10606": "gpu:a100:4",
        "ng10607": "gpu:a100:4",
        "ng10608": "gpu:a100:4",
        "ng10609": "gpu:a100:4",
        "ng10610": "gpu:a100:4",
        "ng10701": "gpu:a100:4",
        "ng10702": "gpu:a100:4",
        "ng10703": "gpu:a100:4",
        "ng10704": "gpu:a100:4",
        "ng10705": "gpu:a100:4",
        "ng10706": "gpu:a100:4",
        "ng10707": "gpu:a100:4",
        "ng10708": "gpu:a100:4",
        "ng10709": "gpu:a100:4",
        "ng10710": "gpu:a100:4",
        "ng10711": "gpu:a100:4",
        "ng10712": "gpu:a100:4",
        "ng10801": "gpu:a100:4",
        "ng10802": "gpu:a100:4",
        "ng10803": "gpu:a100:4",
        "ng10804": "gpu:a100:4",
        "ng10805": "gpu:a100:4",
        "ng10806": "gpu:a100:4",
        "ng10807": "gpu:a100:4",
        "ng10808": "gpu:a100:4",
        "ng10901": "gpu:a100:4",
        "ng10902": "gpu:a100:4",
        "ng10903": "gpu:a100:4",
        "ng10904": "gpu:a100:4",
        "ng10905": "gpu:a100:4",
        "ng10906": "gpu:a100:4",
        "ng11001": "gpu:a100:4",
        "ng11002": "gpu:a100:4",
        "ng11003": "gpu:a100:4",
        "ng11004": "gpu:a100:4",
        "ng11005": "gpu:a100:4",
        "ng11006": "gpu:a100:4",
        "ng11101": "gpu:a100:4",
        "ng11102": "gpu:a100:4",
        "ng11103": "gpu:a100:4",
        "ng11104": "gpu:a100:4",
        "ng11105": "gpu:a100:4",
        "ng11106": "gpu:a100:4",
        "ng20101": "gpu:a100:4",
        "ng20102": "gpu:a100:4",
        "ng20103": "gpu:a100:4",
        "ng20104": "gpu:a100:4",
        "ng20201": "gpu:a100:4",
        "ng20202": "gpu:a100:4",
        "ng20203": "gpu:a100:4",
        "ng20204": "gpu:a100:4",
        "ng20301": "gpu:a100:4",
        "ng20302": "gpu:a100:4",
        "ng20303": "gpu:a100:4",
        "ng20304": "gpu:a100:4",
        "ng20401": "gpu:a100:4",
        "ng20402": "gpu:a100:4",
        "ng20403": "gpu:a100:4",
        "ng20404": "gpu:a100:4",
        "ng20501": "gpu:a100:4",
        "ng20502": "gpu:a100:4",
        "ng20503": "gpu:a100:4",
        "ng20504": "gpu:a100:4",
        "ng20601": "gpu:a100:4",
        "ng20602": "gpu:a100:4",
        "ng20603": "gpu:a100:4",
        "ng20604": "gpu:a100:4",
        "ng30101": "gpu:a100:4",
        "ng30102": "gpu:a100:4",
        "ng30103": "gpu:a100:4",
        "ng30104": "gpu:a100:4",
        "ng30201": "gpu:a100:4",
        "ng30202": "gpu:a100:4",
        "ng30203": "gpu:a100:4",
        "ng30204": "gpu:a100:4",
        "ng30601": "gpu:a100:4",
        "ng30602": "gpu:a100:4",
        "ng30603": "gpu:a100:4",
        "ng30604": "gpu:a100:4",
        "ng30605": "gpu:a100:4",
        "ng30701": "gpu:a100:4",
        "ng30702": "gpu:a100:4",
        "ng30703": "gpu:a100:4",
        "ng30704": "gpu:a100:4",
        "ng30705": "gpu:a100:4",
        "ng30706": "gpu:a100:4",
        "ng30707": "gpu:a100:4",
        "ng30708": "gpu:a100:4",
        "ng30709": "gpu:a100:4",
        "ng30710": "gpu:a100:4",
        "ng30711": "gpu:a100:4",
        "ng30712": "gpu:a100:4",
        "ng30801": "gpu:a100:4",
        "ng30802": "gpu:a100:4",
        "ng30803": "gpu:a100:4",
        "ng30804": "gpu:a100:4",
        "ng30805": "gpu:a100:4",
        "ng30806": "gpu:a100:4",
        "ng30807": "gpu:a100:4",
        "ng30808": "gpu:a100:4",
        "ng30809": "gpu:a100:4",
        "ng30810": "gpu:a100:4",
        "ng30811": "gpu:a100:4",
        "ng30901": "gpu:a100:4",
        "ng30902": "gpu:a100:4",
        "ng30903": "gpu:a100:4",
        "ng30904": "gpu:a100:4",
        "ng30905": "gpu:a100:4",
        "ng30906": "gpu:a100:4",
        "ng30907": "gpu:a100:4",
        "ng30908": "gpu:a100:4",
        "ng30909": "gpu:a100:4",
        "ng30910": "gpu:a100:4",
        "ng30911": "gpu:a100:4",
        "ng30912": "gpu:a100:4",
        "ng31001": "gpu:a100:4",
        "ng31002": "gpu:a100:4",
        "ng31003": "gpu:a100:4",
        "ng31004": "gpu:a100:4",
        "ng31005": "gpu:a100:4",
        "ng31006": "gpu:a100:4",
        "ng31101": "gpu:a100:4",
        "ng31102": "gpu:a100:4",
        "ng31103": "gpu:a100:4",
        "ng31104": "gpu:a100:4",
        "ng31201": "gpu:a100:4",
        "ng31202": "gpu:a100:4",
        "ng30301": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30302": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30303": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30304": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30401": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30402": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30403": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30404": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30501": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30502": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30503": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
        "ng30504": "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4",
    },
    "mila": {
        "apollor06": "gpu:rtx8000:8",
        "cn-a001": "gpu:rtx8000:8",
        "cn-a002": "gpu:rtx8000:8",
        "cn-a003": "gpu:rtx8000:8",
        "cn-a004": "gpu:rtx8000:8",
        "cn-a005": "gpu:rtx8000:8",
        "cn-a006": "gpu:rtx8000:8",
        "cn-a007": "gpu:rtx8000:8",
        "cn-a008": "gpu:rtx8000:8",
        "cn-a009": "gpu:rtx8000:8",
        "cn-a010": "gpu:rtx8000:8",
        "cn-a011": "gpu:rtx8000:8",
        "cn-b001": "gpu:v100:8",
        "cn-b002": "gpu:v100:8",
        "cn-b003": "gpu:v100:8",
        "cn-b004": "gpu:v100:8",
        "cn-b005": "gpu:v100:8",
        "cn-c001": "gpu:rtx8000:8",
        "cn-c002": "gpu:rtx8000:8",
        "cn-c003": "gpu:rtx8000:8",
        "cn-c004": "gpu:rtx8000:8",
        "cn-c005": "gpu:rtx8000:8",
        "cn-c006": "gpu:rtx8000:8",
        "cn-c007": "gpu:rtx8000:8",
        "cn-c008": "gpu:rtx8000:8",
        "cn-c009": "gpu:rtx8000:8",
        "cn-c010": "gpu:rtx8000:8",
        "cn-c011": "gpu:rtx8000:8",
        "cn-c012": "gpu:rtx8000:8",
        "cn-c013": "gpu:rtx8000:8",
        "cn-c014": "gpu:rtx8000:8",
        "cn-c015": "gpu:rtx8000:8",
        "cn-c016": "gpu:rtx8000:8",
        "cn-c017": "gpu:rtx8000:8",
        "cn-c018": "gpu:rtx8000:8",
        "cn-c019": "gpu:rtx8000:8",
        "cn-c020": "gpu:rtx8000:8",
        "cn-c021": "gpu:rtx8000:8",
        "cn-c022": "gpu:rtx8000:8",
        "cn-c023": "gpu:rtx8000:8",
        "cn-c024": "gpu:rtx8000:8",
        "cn-c025": "gpu:rtx8000:8",
        "cn-c026": "gpu:rtx8000:8",
        "cn-c027": "gpu:rtx8000:8",
        "cn-c028": "gpu:rtx8000:8",
        "cn-c029": "gpu:rtx8000:8",
        "cn-c030": "gpu:rtx8000:8",
        "cn-c031": "gpu:rtx8000:8",
        "cn-c032": "gpu:rtx8000:8",
        "cn-c033": "gpu:rtx8000:8",
        "cn-c034": "gpu:rtx8000:8",
        "cn-c035": "gpu:rtx8000:8",
        "cn-c036": "gpu:rtx8000:8",
        "cn-c037": "gpu:rtx8000:8",
        "cn-c038": "gpu:rtx8000:8",
        "cn-c039": "gpu:rtx8000:8",
        "cn-c040": "gpu:rtx8000:8",
        "cn-d001": "gpu:a100:8",
        "cn-d002": "gpu:a100:8",
        "cn-d003": "gpu:a100l:8",
        "cn-d004": "gpu:a100l:8",
        "cn-e002": "gpu:v100:8",
        "cn-e003": "gpu:v100:8",
        "cn-g001": "gpu:a100l:4",
        "cn-g002": "gpu:a100l:4",
        "cn-g003": "gpu:a100l:4",
        "cn-g004": "gpu:a100l:4",
        "cn-g005": "gpu:a100l:4",
        "cn-g006": "gpu:a100l:4",
        "cn-g007": "gpu:a100l:4",
        "cn-g008": "gpu:a100l:4",
        "cn-g009": "gpu:a100l:4",
        "cn-g010": "gpu:a100l:4",
        "cn-g011": "gpu:a100l:4",
        "cn-g012": "gpu:a100l:4",
        "cn-g013": "gpu:a100l:4",
        "cn-g014": "gpu:a100l:4",
        "cn-g015": "gpu:a100l:4",
        "cn-g016": "gpu:a100l:4",
        "cn-g017": "gpu:a100l:4",
        "cn-g018": "gpu:a100l:4",
        "cn-g019": "gpu:a100l:4",
        "cn-g020": "gpu:a100l:4",
        "cn-g021": "gpu:a100l:4",
        "cn-g022": "gpu:a100l:4",
        "cn-g023": "gpu:a100l:4",
        "cn-g024": "gpu:a100l:4",
        "cn-g025": "gpu:a100l:4",
        "cn-g026": "gpu:a100l:4",
        "cn-g027": "gpu:a100l:4",
        "cn-g028": "gpu:a100l:4",
        "cn-g029": "gpu:a100l:4",
        "cn-i001": "gpu:a100l:4",
        "cn-j001": "gpu:a6000:8",
        "cn-k001": "gpu:a100:4",
        "cn-k002": "gpu:a100:4",
        "cn-k003": "gpu:a100:4",
        "cn-k004": "gpu:a100:4",
        "cn-l001": "gpu:l40s:4",
        "cn-l002": "gpu:l40s:4",
        "cn-l003": "gpu:l40s:4",
        "cn-l004": "gpu:l40s:4",
        "cn-l005": "gpu:l40s:4",
        "cn-l006": "gpu:l40s:4",
        "cn-l007": "gpu:l40s:4",
        "cn-l008": "gpu:l40s:4",
        "cn-l009": "gpu:l40s:4",
        "cn-l010": "gpu:l40s:4",
        "cn-l011": "gpu:l40s:4",
        "cn-l012": "gpu:l40s:4",
        "cn-l013": "gpu:l40s:4",
        "cn-l014": "gpu:l40s:4",
        "cn-l015": "gpu:l40s:4",
        "cn-l016": "gpu:l40s:4",
        "cn-l017": "gpu:l40s:4",
        "cn-l018": "gpu:l40s:4",
        "cn-l019": "gpu:l40s:4",
        "cn-l020": "gpu:l40s:4",
        "cn-l021": "gpu:l40s:4",
        "cn-l022": "gpu:l40s:4",
        "cn-l023": "gpu:l40s:4",
        "cn-l024": "gpu:l40s:4",
        "cn-l025": "gpu:l40s:4",
        "cn-l026": "gpu:l40s:4",
        "cn-l027": "gpu:l40s:4",
        "cn-l028": "gpu:l40s:4",
        "cn-l029": "gpu:l40s:4",
        "cn-l030": "gpu:l40s:4",
        "cn-l031": "gpu:l40s:4",
        "cn-l032": "gpu:l40s:4",
        "cn-l033": "gpu:l40s:4",
        "cn-l034": "gpu:l40s:4",
        "cn-l035": "gpu:l40s:4",
        "cn-l036": "gpu:l40s:4",
        "cn-l037": "gpu:l40s:4",
        "cn-l038": "gpu:l40s:4",
        "cn-l039": "gpu:l40s:4",
        "cn-l040": "gpu:l40s:4",
        "cn-l041": "gpu:l40s:4",
        "cn-l042": "gpu:l40s:4",
        "cn-l043": "gpu:l40s:4",
        "cn-l044": "gpu:l40s:4",
        "cn-l045": "gpu:l40s:4",
        "cn-l046": "gpu:l40s:4",
        "cn-l047": "gpu:l40s:4",
        "cn-l048": "gpu:l40s:4",
        "cn-l049": "gpu:l40s:4",
        "cn-l050": "gpu:l40s:4",
        "cn-l051": "gpu:l40s:4",
        "cn-l052": "gpu:l40s:4",
        "cn-l053": "gpu:l40s:4",
        "cn-l054": "gpu:l40s:4",
        "cn-l055": "gpu:l40s:4",
        "cn-l056": "gpu:l40s:4",
        "cn-l057": "gpu:l40s:4",
        "cn-l058": "gpu:l40s:4",
        "cn-l059": "gpu:l40s:4",
        "cn-l060": "gpu:l40s:4",
        "cn-l061": "gpu:l40s:4",
        "cn-l062": "gpu:l40s:4",
        "cn-l063": "gpu:l40s:4",
        "cn-l064": "gpu:l40s:4",
        "cn-l065": "gpu:l40s:4",
        "cn-l066": "gpu:l40s:4",
        "cn-l067": "gpu:l40s:4",
        "cn-l068": "gpu:l40s:4",
        "cn-l069": "gpu:l40s:4",
        "cn-l070": "gpu:l40s:4",
        "cn-l071": "gpu:l40s:4",
        "cn-l072": "gpu:l40s:4",
        "cn-l073": "gpu:l40s:4",
        "cn-l074": "gpu:l40s:4",
        "cn-l075": "gpu:l40s:4",
        "cn-l076": "gpu:l40s:4",
        "cn-l077": "gpu:l40s:4",
        "cn-l078": "gpu:l40s:4",
        "cn-l079": "gpu:l40s:4",
        "cn-l080": "gpu:l40s:4",
        "cn-l081": "gpu:l40s:4",
        "cn-l082": "gpu:l40s:4",
        "cn-l083": "gpu:l40s:4",
        "cn-l084": "gpu:l40s:4",
        "cn-l085": "gpu:l40s:4",
        "cn-l086": "gpu:l40s:4",
        "cn-l087": "gpu:l40s:4",
        "cn-l088": "gpu:l40s:4",
        "cn-l089": "gpu:l40s:4",
        "cn-l090": "gpu:l40s:4",
        "cn-l091": "gpu:l40s:4",
        "cn-l092": "gpu:l40s:4",
        "cn-n001": "gpu:h100:8",
        "cn-n002": "gpu:h100:8",
    },
    "cedar": {
        "cdr250": "gpu:p100:4",
        "cdr255": "gpu:p100:4",
        "cdr257": "gpu:p100:4",
        "cdr294": "gpu:p100:4",
        "cdr301": "gpu:p100:4",
        "cdr302": "gpu:p100:4",
        "cdr305": "gpu:p100:4",
        "cdr338": "gpu:p100:4",
        "cdr340": "gpu:p100:4",
        "cdr341": "gpu:p100:4",
        "cdr342": "gpu:p100:4",
        "cdr343": "gpu:p100:4",
        "cdr344": "gpu:p100:4",
        "cdr345": "gpu:p100:4",
        "cdr346": "gpu:p100:4",
        "cdr347": "gpu:p100:4",
        "cdr348": "gpu:p100:4",
        "cdr349": "gpu:p100:4",
        "cdr350": "gpu:p100:4",
        "cdr351": "gpu:p100:4",
        "cdr385": "gpu:p100:4",
        "cdr386": "gpu:p100:4",
        "cdr906": "gpu:p100:4",
        "cdr912": "gpu:p100:4",
        "cdr245": "gpu:p100:4",
        "cdr246": "gpu:p100:4",
        "cdr247": "gpu:p100:4",
        "cdr248": "gpu:p100:4",
        "cdr249": "gpu:p100:4",
        "cdr251": "gpu:p100:4",
        "cdr252": "gpu:p100:4",
        "cdr253": "gpu:p100:4",
        "cdr254": "gpu:p100:4",
        "cdr256": "gpu:p100:4",
        "cdr258": "gpu:p100:4",
        "cdr259": "gpu:p100:4",
        "cdr292": "gpu:p100:4",
        "cdr293": "gpu:p100:4",
        "cdr295": "gpu:p100:4",
        "cdr296": "gpu:p100:4",
        "cdr297": "gpu:p100:4",
        "cdr298": "gpu:p100:4",
        "cdr299": "gpu:p100:4",
        "cdr300": "gpu:p100:4",
        "cdr303": "gpu:p100:4",
        "cdr304": "gpu:p100:4",
        "cdr306": "gpu:p100:4",
        "cdr917": "gpu:p100:4",
        "cdr26": "gpu:p100:4",
        "cdr27": "gpu:p100:4",
        "cdr28": "gpu:p100:4",
        "cdr29": "gpu:p100:4",
        "cdr30": "gpu:p100:4",
        "cdr31": "gpu:p100:4",
        "cdr32": "gpu:p100:4",
        "cdr33": "gpu:p100:4",
        "cdr34": "gpu:p100:4",
        "cdr35": "gpu:p100:4",
        "cdr36": "gpu:p100:4",
        "cdr37": "gpu:p100:4",
        "cdr38": "gpu:p100:4",
        "cdr39": "gpu:p100:4",
        "cdr40": "gpu:p100:4",
        "cdr197": "gpu:p100:4",
        "cdr198": "gpu:p100:4",
        "cdr199": "gpu:p100:4",
        "cdr200": "gpu:p100:4",
        "cdr201": "gpu:p100:4",
        "cdr202": "gpu:p100:4",
        "cdr203": "gpu:p100:4",
        "cdr204": "gpu:p100:4",
        "cdr205": "gpu:p100:4",
        "cdr206": "gpu:p100:4",
        "cdr207": "gpu:p100:4",
        "cdr208": "gpu:p100:4",
        "cdr209": "gpu:p100:4",
        "cdr210": "gpu:p100:4",
        "cdr211": "gpu:p100:4",
        "cdr212": "gpu:p100:4",
        "cdr922": "gpu:p100:4",
        "cdr104": "gpu:p100:4",
        "cdr105": "gpu:p100:4",
        "cdr106": "gpu:p100:4",
        "cdr107": "gpu:p100:4",
        "cdr108": "gpu:p100:4",
        "cdr116": "gpu:p100:4",
        "cdr150": "gpu:p100:4",
        "cdr151": "gpu:p100:4",
        "cdr153": "gpu:p100:4",
        "cdr154": "gpu:p100:4",
        "cdr155": "gpu:p100:4",
        "cdr157": "gpu:p100:4",
        "cdr159": "gpu:p100:4",
        "cdr160": "gpu:p100:4",
        "cdr161": "gpu:p100:4",
        "cdr164": "gpu:p100:4",
        "cdr109": "gpu:p100:4",
        "cdr110": "gpu:p100:4",
        "cdr111": "gpu:p100:4",
        "cdr112": "gpu:p100:4",
        "cdr113": "gpu:p100:4",
        "cdr114": "gpu:p100:4",
        "cdr115": "gpu:p100:4",
        "cdr117": "gpu:p100:4",
        "cdr118": "gpu:p100:4",
        "cdr149": "gpu:p100:4",
        "cdr152": "gpu:p100:4",
        "cdr156": "gpu:p100:4",
        "cdr158": "gpu:p100:4",
        "cdr162": "gpu:p100:4",
        "cdr163": "gpu:p100:4",
        "cdr905": "gpu:p100:4",
        "cdr352": "gpu:p100:4",
        "cdr353": "gpu:p100:4",
        "cdr902": "gpu:p100l:4",
        "cdr903": "gpu:p100l:4",
        "cdr897": "gpu:p100l:4",
        "cdr898": "gpu:p100l:4",
        "cdr899": "gpu:p100l:4",
        "cdr900": "gpu:p100l:4",
        "cdr885": "gpu:p100l:4",
        "cdr886": "gpu:p100l:4",
        "cdr887": "gpu:p100l:4",
        "cdr888": "gpu:p100l:4",
        "cdr889": "gpu:p100l:4",
        "cdr890": "gpu:p100l:4",
        "cdr891": "gpu:p100l:4",
        "cdr892": "gpu:p100l:4",
        "cdr893": "gpu:p100l:4",
        "cdr894": "gpu:p100l:4",
        "cdr895": "gpu:p100l:4",
        "cdr896": "gpu:p100l:4",
        "cdr880": "gpu:p100l:4",
        "cdr881": "gpu:p100l:4",
        "cdr882": "gpu:p100l:4",
        "cdr883": "gpu:p100l:4",
        "cdr884": "gpu:p100l:4",
        "cdr908": "gpu:p100l:4",
        "cdr7": "gpu:p100l:4",
        "cdr876": "gpu:p100l:4",
        "cdr877": "gpu:p100l:4",
        "cdr878": "gpu:p100l:4",
        "cdr901": "gpu:p100l:4",
        "cdr910": "gpu:p100l:4",
        "cdr911": "gpu:p100l:4",
        "cdr2636": "gpu:v100l:4",
        "cdr2637": "gpu:v100l:4",
        "cdr2639": "gpu:v100l:4",
        "cdr2640": "gpu:v100l:4",
        "cdr2641": "gpu:v100l:4",
        "cdr2643": "gpu:v100l:4",
        "cdr2628": "gpu:v100l:4",
        "cdr2629": "gpu:v100l:4",
        "cdr2630": "gpu:v100l:4",
        "cdr2635": "gpu:v100l:4",
        "cdr2638": "gpu:v100l:4",
        "cdr2642": "gpu:v100l:4",
        "cdr2631": "gpu:v100l:4",
        "cdr2632": "gpu:v100l:4",
        "cdr2633": "gpu:v100l:4",
        "cdr2634": "gpu:v100l:4",
        "cdr2644": "gpu:v100l:4",
        "cdr2645": "gpu:v100l:4",
        "cdr2646": "gpu:v100l:4",
        "cdr2647": "gpu:v100l:4",
        "cdr2652": "gpu:v100l:4",
        "cdr2653": "gpu:v100l:4",
        "cdr2654": "gpu:v100l:4",
        "cdr2655": "gpu:v100l:4",
        "cdr2656": "gpu:v100l:4",
        "cdr2657": "gpu:v100l:4",
        "cdr2648": "gpu:v100l:4",
        "cdr2649": "gpu:v100l:4",
        "cdr2650": "gpu:v100l:4",
        "cdr2651": "gpu:v100l:4",
        "cdr2658": "gpu:v100l:4",
        "cdr2659": "gpu:v100l:4",
        "cdr2596": "gpu:v100l:4",
        "cdr2597": "gpu:v100l:4",
        "cdr2598": "gpu:v100l:4",
        "cdr2599": "gpu:v100l:4",
        "cdr2600": "gpu:v100l:4",
        "cdr2601": "gpu:v100l:4",
        "cdr2602": "gpu:v100l:4",
        "cdr2603": "gpu:v100l:4",
        "cdr2604": "gpu:v100l:4",
        "cdr2605": "gpu:v100l:4",
        "cdr2606": "gpu:v100l:4",
        "cdr2607": "gpu:v100l:4",
        "cdr2608": "gpu:v100l:4",
        "cdr2609": "gpu:v100l:4",
        "cdr2610": "gpu:v100l:4",
        "cdr2611": "gpu:v100l:4",
        "cdr2612": "gpu:v100l:4",
        "cdr2613": "gpu:v100l:4",
        "cdr2614": "gpu:v100l:4",
        "cdr2615": "gpu:v100l:4",
        "cdr2616": "gpu:v100l:4",
        "cdr2617": "gpu:v100l:4",
        "cdr2618": "gpu:v100l:4",
        "cdr2619": "gpu:v100l:4",
        "cdr2620": "gpu:v100l:4",
        "cdr2621": "gpu:v100l:4",
        "cdr2622": "gpu:v100l:4",
        "cdr2623": "gpu:v100l:4",
        "cdr2624": "gpu:v100l:4",
        "cdr2625": "gpu:v100l:4",
        "cdr2626": "gpu:v100l:4",
        "cdr2627": "gpu:v100l:4",
        "cdr2564": "gpu:v100l:4",
        "cdr2565": "gpu:v100l:4",
        "cdr2566": "gpu:v100l:4",
        "cdr2567": "gpu:v100l:4",
        "cdr2568": "gpu:v100l:4",
        "cdr2569": "gpu:v100l:4",
        "cdr2570": "gpu:v100l:4",
        "cdr2571": "gpu:v100l:4",
        "cdr2572": "gpu:v100l:4",
        "cdr2573": "gpu:v100l:4",
        "cdr2574": "gpu:v100l:4",
        "cdr2575": "gpu:v100l:4",
        "cdr2576": "gpu:v100l:4",
        "cdr2577": "gpu:v100l:4",
        "cdr2578": "gpu:v100l:4",
        "cdr2579": "gpu:v100l:4",
        "cdr2580": "gpu:v100l:4",
        "cdr2581": "gpu:v100l:4",
        "cdr2582": "gpu:v100l:4",
        "cdr2583": "gpu:v100l:4",
        "cdr2584": "gpu:v100l:4",
        "cdr2585": "gpu:v100l:4",
        "cdr2586": "gpu:v100l:4",
        "cdr2587": "gpu:v100l:4",
        "cdr2588": "gpu:v100l:4",
        "cdr2589": "gpu:v100l:4",
        "cdr2590": "gpu:v100l:4",
        "cdr2591": "gpu:v100l:4",
        "cdr2592": "gpu:v100l:4",
        "cdr2593": "gpu:v100l:4",
        "cdr2594": "gpu:v100l:4",
        "cdr2595": "gpu:v100l:4",
        "cdr2532": "gpu:v100l:4",
        "cdr2533": "gpu:v100l:4",
        "cdr2534": "gpu:v100l:4",
        "cdr2535": "gpu:v100l:4",
        "cdr2536": "gpu:v100l:4",
        "cdr2537": "gpu:v100l:4",
        "cdr2538": "gpu:v100l:4",
        "cdr2539": "gpu:v100l:4",
        "cdr2540": "gpu:v100l:4",
        "cdr2541": "gpu:v100l:4",
        "cdr2542": "gpu:v100l:4",
        "cdr2543": "gpu:v100l:4",
        "cdr2544": "gpu:v100l:4",
        "cdr2545": "gpu:v100l:4",
        "cdr2546": "gpu:v100l:4",
        "cdr2547": "gpu:v100l:4",
        "cdr2548": "gpu:v100l:4",
        "cdr2549": "gpu:v100l:4",
        "cdr2550": "gpu:v100l:4",
        "cdr2551": "gpu:v100l:4",
        "cdr2552": "gpu:v100l:4",
        "cdr2553": "gpu:v100l:4",
        "cdr2554": "gpu:v100l:4",
        "cdr2555": "gpu:v100l:4",
        "cdr2556": "gpu:v100l:4",
        "cdr2557": "gpu:v100l:4",
        "cdr2558": "gpu:v100l:4",
        "cdr2559": "gpu:v100l:4",
        "cdr2560": "gpu:v100l:4",
        "cdr2561": "gpu:v100l:4",
        "cdr2562": "gpu:v100l:4",
        "cdr2563": "gpu:v100l:4",
        "cdr2500": "gpu:v100l:4",
        "cdr2501": "gpu:v100l:4",
        "cdr2502": "gpu:v100l:4",
        "cdr2503": "gpu:v100l:4",
        "cdr2504": "gpu:v100l:4",
        "cdr2505": "gpu:v100l:4",
        "cdr2506": "gpu:v100l:4",
        "cdr2507": "gpu:v100l:4",
        "cdr2508": "gpu:v100l:4",
        "cdr2509": "gpu:v100l:4",
        "cdr2510": "gpu:v100l:4",
        "cdr2511": "gpu:v100l:4",
        "cdr2512": "gpu:v100l:4",
        "cdr2513": "gpu:v100l:4",
        "cdr2514": "gpu:v100l:4",
        "cdr2515": "gpu:v100l:4",
        "cdr2516": "gpu:v100l:4",
        "cdr2517": "gpu:v100l:4",
        "cdr2518": "gpu:v100l:4",
        "cdr2519": "gpu:v100l:4",
        "cdr2520": "gpu:v100l:4",
        "cdr2521": "gpu:v100l:4",
        "cdr2522": "gpu:v100l:4",
        "cdr2523": "gpu:v100l:4",
        "cdr2524": "gpu:v100l:4",
        "cdr2525": "gpu:v100l:4",
        "cdr2526": "gpu:v100l:4",
        "cdr2527": "gpu:v100l:4",
        "cdr2528": "gpu:v100l:4",
        "cdr2529": "gpu:v100l:4",
        "cdr2530": "gpu:v100l:4",
        "cdr2531": "gpu:v100l:4",
        "cdr2468": "gpu:v100l:4",
        "cdr2469": "gpu:v100l:4",
        "cdr2470": "gpu:v100l:4",
        "cdr2471": "gpu:v100l:4",
        "cdr2472": "gpu:v100l:4",
        "cdr2473": "gpu:v100l:4",
        "cdr2474": "gpu:v100l:4",
        "cdr2475": "gpu:v100l:4",
        "cdr2476": "gpu:v100l:4",
        "cdr2477": "gpu:v100l:4",
        "cdr2478": "gpu:v100l:4",
        "cdr2479": "gpu:v100l:4",
        "cdr2480": "gpu:v100l:4",
        "cdr2481": "gpu:v100l:4",
        "cdr2482": "gpu:v100l:4",
        "cdr2483": "gpu:v100l:4",
        "cdr2484": "gpu:v100l:4",
        "cdr2485": "gpu:v100l:4",
        "cdr2486": "gpu:v100l:4",
        "cdr2487": "gpu:v100l:4",
        "cdr2488": "gpu:v100l:4",
        "cdr2489": "gpu:v100l:4",
        "cdr2490": "gpu:v100l:4",
        "cdr2491": "gpu:v100l:4",
        "cdr2492": "gpu:v100l:4",
        "cdr2493": "gpu:v100l:4",
        "cdr2494": "gpu:v100l:4",
        "cdr2495": "gpu:v100l:4",
        "cdr2496": "gpu:v100l:4",
        "cdr2497": "gpu:v100l:4",
        "cdr2498": "gpu:v100l:4",
        "cdr2499": "gpu:v100l:4",
        "cdr2678": "gpu:v100l:4",
        "cdr2683": "gpu:a40:4",
        "cdr2684": "gpu:a40:4",
        "cdr2685": "gpu:a40:4",
        "cdr2686": "gpu:a40:4",
        "cdr2687": "gpu:a40:4",
    },
    "graham": {
        "gra828": "gpu:p100:2",
        "gra829": "gpu:p100:2",
        "gra830": "gpu:p100:2",
        "gra831": "gpu:p100:2",
        "gra832": "gpu:p100:2",
        "gra833": "gpu:p100:2",
        "gra834": "gpu:p100:2",
        "gra835": "gpu:p100:2",
        "gra836": "gpu:p100:2",
        "gra837": "gpu:p100:2",
        "gra838": "gpu:p100:2",
        "gra839": "gpu:p100:2",
        "gra840": "gpu:p100:2",
        "gra841": "gpu:p100:2",
        "gra842": "gpu:p100:2",
        "gra843": "gpu:p100:2",
        "gra844": "gpu:p100:2",
        "gra845": "gpu:p100:2",
        "gra846": "gpu:p100:2",
        "gra847": "gpu:p100:2",
        "gra848": "gpu:p100:2",
        "gra849": "gpu:p100:2",
        "gra850": "gpu:p100:2",
        "gra851": "gpu:p100:2",
        "gra852": "gpu:p100:2",
        "gra853": "gpu:p100:2",
        "gra854": "gpu:p100:2",
        "gra855": "gpu:p100:2",
        "gra856": "gpu:p100:2",
        "gra857": "gpu:p100:2",
        "gra858": "gpu:p100:2",
        "gra859": "gpu:p100:2",
        "gra860": "gpu:p100:2",
        "gra861": "gpu:p100:2",
        "gra862": "gpu:p100:2",
        "gra863": "gpu:p100:2",
        "gra864": "gpu:p100:2",
        "gra865": "gpu:p100:2",
        "gra866": "gpu:p100:2",
        "gra867": "gpu:p100:2",
        "gra868": "gpu:p100:2",
        "gra869": "gpu:p100:2",
        "gra870": "gpu:p100:2",
        "gra871": "gpu:p100:2",
        "gra872": "gpu:p100:2",
        "gra873": "gpu:p100:2",
        "gra874": "gpu:p100:2",
        "gra875": "gpu:p100:2",
        "gra876": "gpu:p100:2",
        "gra877": "gpu:p100:2",
        "gra878": "gpu:p100:2",
        "gra879": "gpu:p100:2",
        "gra880": "gpu:p100:2",
        "gra881": "gpu:p100:2",
        "gra882": "gpu:p100:2",
        "gra883": "gpu:p100:2",
        "gra884": "gpu:p100:2",
        "gra885": "gpu:p100:2",
        "gra886": "gpu:p100:2",
        "gra887": "gpu:p100:2",
        "gra888": "gpu:p100:2",
        "gra889": "gpu:p100:2",
        "gra890": "gpu:p100:2",
        "gra891": "gpu:p100:2",
        "gra892": "gpu:p100:2",
        "gra893": "gpu:p100:2",
        "gra894": "gpu:p100:2",
        "gra895": "gpu:p100:2",
        "gra896": "gpu:p100:2",
        "gra897": "gpu:p100:2",
        "gra898": "gpu:p100:2",
        "gra899": "gpu:p100:2",
        "gra900": "gpu:p100:2",
        "gra901": "gpu:p100:2",
        "gra902": "gpu:p100:2",
        "gra903": "gpu:p100:2",
        "gra904": "gpu:p100:2",
        "gra905": "gpu:p100:2",
        "gra906": "gpu:p100:2",
        "gra907": "gpu:p100:2",
        "gra908": "gpu:p100:2",
        "gra909": "gpu:p100:2",
        "gra910": "gpu:p100:2",
        "gra911": "gpu:p100:2",
        "gra912": "gpu:p100:2",
        "gra913": "gpu:p100:2",
        "gra914": "gpu:p100:2",
        "gra915": "gpu:p100:2",
        "gra916": "gpu:p100:2",
        "gra917": "gpu:p100:2",
        "gra918": "gpu:p100:2",
        "gra919": "gpu:p100:2",
        "gra920": "gpu:p100:2",
        "gra921": "gpu:p100:2",
        "gra922": "gpu:p100:2",
        "gra923": "gpu:p100:2",
        "gra924": "gpu:p100:2",
        "gra925": "gpu:p100:2",
        "gra926": "gpu:p100:2",
        "gra927": "gpu:p100:2",
        "gra928": "gpu:p100:2",
        "gra929": "gpu:p100:2",
        "gra930": "gpu:p100:2",
        "gra931": "gpu:p100:2",
        "gra932": "gpu:p100:2",
        "gra933": "gpu:p100:2",
        "gra934": "gpu:p100:2",
        "gra935": "gpu:p100:2",
        "gra936": "gpu:p100:2",
        "gra937": "gpu:p100:2",
        "gra938": "gpu:p100:2",
        "gra939": "gpu:p100:2",
        "gra940": "gpu:p100:2",
        "gra941": "gpu:p100:2",
        "gra942": "gpu:p100:2",
        "gra943": "gpu:p100:2",
        "gra944": "gpu:p100:2",
        "gra945": "gpu:p100:2",
        "gra946": "gpu:p100:2",
        "gra947": "gpu:p100:2",
        "gra948": "gpu:p100:2",
        "gra949": "gpu:p100:2",
        "gra950": "gpu:p100:2",
        "gra951": "gpu:p100:2",
        "gra952": "gpu:p100:2",
        "gra953": "gpu:p100:2",
        "gra954": "gpu:p100:2",
        "gra955": "gpu:p100:2",
        "gra956": "gpu:p100:2",
        "gra957": "gpu:p100:2",
        "gra958": "gpu:p100:2",
        "gra959": "gpu:p100:2",
        "gra960": "gpu:p100:2",
        "gra961": "gpu:p100:2",
        "gra962": "gpu:p100:2",
        "gra963": "gpu:p100:2",
        "gra964": "gpu:p100:2",
        "gra965": "gpu:p100:2",
        "gra966": "gpu:p100:2",
        "gra967": "gpu:p100:2",
        "gra968": "gpu:p100:2",
        "gra969": "gpu:p100:2",
        "gra970": "gpu:p100:2",
        "gra971": "gpu:p100:2",
        "gra972": "gpu:p100:2",
        "gra973": "gpu:p100:2",
        "gra974": "gpu:p100:2",
        "gra975": "gpu:p100:2",
        "gra976": "gpu:p100:2",
        "gra977": "gpu:p100:2",
        "gra978": "gpu:p100:2",
        "gra979": "gpu:p100:2",
        "gra980": "gpu:p100:2",
        "gra981": "gpu:p100:2",
        "gra982": "gpu:p100:2",
        "gra983": "gpu:p100:2",
        "gra984": "gpu:p100:2",
        "gra985": "gpu:p100:2",
        "gra986": "gpu:p100:2",
        "gra987": "gpu:p100:2",
        "gra1147": "gpu:v100:6",
        "gra1148": "gpu:v100:8",
        "gra1149": "gpu:v100:8",
        "gra1150": "gpu:v100:8",
        "gra1151": "gpu:v100:8",
        "gra1152": "gpu:v100:8",
        "gra1153": "gpu:v100:8",
        "gra1337": "gpu:v100:8",
        "gra1338": "gpu:v100:8",
        "gra1154": "gpu:t4:4",
        "gra1155": "gpu:t4:4",
        "gra1156": "gpu:t4:4",
        "gra1157": "gpu:t4:4",
        "gra1158": "gpu:t4:4",
        "gra1159": "gpu:t4:4",
        "gra1160": "gpu:t4:4",
        "gra1161": "gpu:t4:4",
        "gra1162": "gpu:t4:4",
        "gra1163": "gpu:t4:4",
        "gra1164": "gpu:t4:4",
        "gra1165": "gpu:t4:4",
        "gra1166": "gpu:t4:4",
        "gra1167": "gpu:t4:4",
        "gra1168": "gpu:t4:4",
        "gra1169": "gpu:t4:4",
        "gra1170": "gpu:t4:4",
        "gra1171": "gpu:t4:4",
        "gra1172": "gpu:t4:4",
        "gra1173": "gpu:t4:4",
        "gra1174": "gpu:t4:4",
        "gra1175": "gpu:t4:4",
        "gra1176": "gpu:t4:4",
        "gra1177": "gpu:t4:4",
        "gra1178": "gpu:t4:4",
        "gra1179": "gpu:t4:4",
        "gra1180": "gpu:t4:4",
        "gra1181": "gpu:t4:4",
        "gra1182": "gpu:t4:4",
        "gra1183": "gpu:t4:4",
        "gra1184": "gpu:t4:4",
        "gra1185": "gpu:t4:4",
        "gra1186": "gpu:t4:4",
        "gra1187": "gpu:t4:4",
        "gra1188": "gpu:t4:4",
        "gra1189": "gpu:t4:4",
        "gra1342": "gpu:a100:8",
        "gra1361": "gpu:a100:4",
        "gra1362": "gpu:a100:4",
        "gra1363": "gpu:a5000:4",
        "gra1364": "gpu:a5000:4",
        "gra1365": "gpu:a5000:4",
        "gra1366": "gpu:a5000:4",
        "gra1367": "gpu:a5000:4",
        "gra1368": "gpu:a5000:4",
        "gra1369": "gpu:a5000:4",
        "gra1370": "gpu:a5000:4",
        "gra1371": "gpu:a5000:4",
        "gra1372": "gpu:a5000:4",
        "gra1373": "gpu:a5000:4",
    },
}
if __name__ == "__main__":
    main()
