# ///
from __future__ import annotations

import asyncio
import enum
import functools
import logging
import subprocess
import tempfile
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import paramiko
import paramiko.config
from rich.panel import Panel
from rich.table import Table
from textual.widgets import DataTable

from sarc.client.job import SlurmState
from sarc.client.users.api import get_users
from sarc.config import MTL

from .common_utils import (
    CACHE_DIR,
    FilteringOptions,
    _CalledProcessError,
    _get_cache_file_name,
    cached,
    midnight,
    run_subprocess,
)
from .sarc_patches import get_clean_sarc_data

logger = logging.getLogger(__name__)


async def setup_torch_import_test(
    hostname: str, remote_dir: str = "$SCRATCH/torch_import_test"
):
    result = await run_subprocess(
        f"ssh {hostname} bash -l",
        input="\n".join(
            [
                f"mkdir -p {remote_dir}",
                f"uv --directory='{remote_dir}' init",
                f"uv --directory='{remote_dir}' add torch numpy",
            ]
        ),
    )
    print(result.stdout)
    print(result.stderr)
    result = await run_subprocess(f"ssh {hostname} 'ls {remote_dir}'")


async def get_torch_import_time(
    hostname: str, remote_dir: str = "$SCRATCH/torch_import_test"
) -> timedelta | None:
    # TODO: Can't for the life of me figure out how to make a login shell work over ssh with async.
    # The best I can do atm is to assume that UV is at ~/.local/bin/uv and check that it is.
    with tempfile.TemporaryFile(mode="w+") as temp_file:
        logger.info("Finding the `uv` executable on %s", hostname)
        _proc = await asyncio.create_subprocess_shell(
            f"ssh {hostname} bash -l which uv", stdout=temp_file
        )
        uv = await _proc.communicate()
        temp_file.seek(0)
        uv = temp_file.read().strip()
    if not uv:
        logger.warning(
            f"Could not find the `uv` executable on {hostname}. "
            "Please ensure that UV is installed and available in the PATH."
        )
        return None
    command = (
        f"ssh {hostname} '{uv} run --directory={remote_dir} "
        'python -c "import time; start=time.time(); import torch; print(time.time()-start)"\''
    )
    result = await run_subprocess(command)
    return timedelta(seconds=float(result.stdout.strip()))


def get_data(
    clusters: Sequence[str] = (),
    # users: Sequence[str] = (),
):
    midnight_tonight = midnight(datetime.now() + timedelta(days=1))
    return _cached_get_data(
        midnight_tonight,
        tuple(sorted(clusters)),  # tuple(sorted(users))
    )


@functools.lru_cache(maxsize=20)
def _cached_get_data(
    midnight_tonight: datetime,
    clusters: tuple[str, ...] = (),
    # users: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Cached function to get the data."""
    options = FilteringOptions(
        start=(midnight_tonight - timedelta(days=7)),
        end=midnight_tonight,
        user=(),
        clusters=(),
    )
    # This is cached as well:
    data = cached(get_clean_sarc_data)(options)
    if clusters:
        data = data[data["cluster_name"].isin(clusters)]
    # if users:
    #     data = data[data["user.mila.email"].isin(users)]
    return data


class Severity(enum.StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def colorize(self) -> str:
        """Colorize the severity level for display."""
        if self == Severity.CRITICAL:
            return "[blink red]CRITICAL[/blink red]"
        elif self == Severity.HIGH:
            return "[red]HIGH[/red]"
        elif self == Severity.MODERATE:
            return "[yellow]MODERATE[/yellow]"
        else:
            return f"[green]{self}[/green]"


class Alert(Protocol):
    """Protocol for an alert function."""

    async def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, str, Severity]:
        """Returns a DataFrame with jobs that should raise an alert, a description, and a severity level."""
        raise NotImplementedError


async def unused_gpus_alert(data: pd.DataFrame) -> tuple[pd.DataFrame, str, Severity]:
    """Returns a DataFrame with jobs that should raise an alert because GPUs are unused."""
    threshold = timedelta(hours=4)
    return (
        data.query(
            "elapsed_time > @threshold and gpu_utilization < 0.05",
            local_dict={"threshold": threshold},
        ),
        "GPU utilization has been <5% for more than 4 hours!",
        Severity.HIGH,
    )


alerts: list[Alert] = [
    unused_gpus_alert,
]


async def fill_alerts_table(table: DataTable, data: pd.DataFrame) -> None:
    # alerts_table = Table(title="Alerts")
    table.clear(columns=True)
    table.add_column("Alert Time")
    table.add_column("Job Period start")
    table.add_column("cluster")
    table.add_column("Job ID")
    table.add_column("User")
    table.add_column("severity")
    table.add_column("Description")

    stuff = await asyncio.gather(*(alert(data) for alert in alerts))
    jobs, descriptions, severities = zip(*stuff)

    for alert, jobs, description, severity in zip(
        alerts, jobs, descriptions, severities
    ):
        # jobs, description, severity = await alert(data)
        if jobs.empty:
            continue

        for user_email, group in jobs.groupby("user.mila.email"):
            # TODO: Check Jira for existing tickets about this user / job.
            # TODO: Group alerts by user, then sort by alert time (newest first?)
            alert_first_posted_time = datetime.now() - timedelta(hours=12)
            clusters = group["cluster_name"].unique()
            job_ids = group["job_id"].unique()
            start_time = group["start_time"].min()
            end_time = group["end_time"].min()

            table.add_row(
                str(alert_first_posted_time),
                f"{(datetime.now(tz=MTL).replace(microsecond=start_time.microsecond) - start_time)} ago",
                # + " "
                # + end_time.strftime("%Y-%m-%d %H:%M:%S"),
                clusters[0] if len(clusters) == 1 else ",".join(clusters),
                textwrap.shorten(
                    (
                        str(job_ids[0])
                        if len(job_ids) == 1
                        else ", ".join(map(str, job_ids))
                    ),
                    width=40,
                ),
                user_email,
                severity.colorize(),
                description + (f" (x {len(group)})" if len(group) > 1 else ""),
            )

    # table.add_row(
    #     str(datetime.now() - timedelta(hours=12)),
    #     "mila",
    #     "123123",
    #     "Bob@mila.quebec",
    #     "[blink red]CRITICAL",
    #     r"GPU utilization has been <5% for more than 4 hours!",
    # )


def fill_waste_overview_datatable(table: DataTable, data: pd.DataFrame) -> None:
    """Make a new table."""
    n_to_show = 50

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
        columns="rgu_equivalent_waste", n=n_to_show, keep="all"
    )
    gpu_util_stats = data.groupby(["cluster_name", "user.mila.email"]).aggregate(
        {"gpu_utilization": "describe"}
    )

    # table = Table(
    #     title=f"Most wasteful users (last 7 days) [{layout_iteration+1} / {n_layout_iterations}]",
    #     expand=True,
    # )
    table.clear(columns=True)
    table.add_column("#")
    table.add_column("User")
    table.add_column("Cluster")
    table.add_column("GPU Utilization")
    table.add_column("Job success rate")
    # table.add_column("Total allocated GPUs/RGUs", justify="right")
    table.add_column("Used/Wasted/Obstructed GPU days")
    table.add_column("U/W/Obs RGU*days")

    for i, (index, row) in enumerate(ordered_by_waste.iterrows(), start=1):
        assert isinstance(index, tuple) and len(index) == 2
        (cluster, user_email) = index
        used_gpus = row["allocated.gres_gpu"]
        gpu_util = gpu_util_stats.loc[index, "gpu_utilization"]
        table.add_row(
            f"{i}",
            user_email,
            cluster,
            f"{_colorize_utilization(gpu_util['mean'])} ± {gpu_util['std']:.1%}",
            f"{_colorize_utilization(row['job_success_rate'], red=0.1, orange=0.2)} (n={row['job_id']})",
            # f"{round(row['allocated.gres_gpu'])} / {round(row['allocated.gres_rgu'])}",
            f"[green]{row['gpu_equivalent_cost'].days}[/green] / [red]{row['gpu_equivalent_waste'].days}[/red] / [red]{row['gpu_overbilling_cost'].days}[/red]",
            f"[green]{row['rgu_equivalent_cost'].days}[/green] / [red]{row['rgu_equivalent_waste'].days}[/red] / [red]{row['rgu_overbilling_cost'].days}[/red]",
            # f"[red] {row['gpu_equivalent_waste'].days:.2f} / {row['rgu_equivalent_waste'].days:.2f}",
        )
    # return table


def fill_cluster_overview_table(table: DataTable, data: pd.DataFrame) -> None:
    table.clear(columns=True)

    total_mila_users = get_mila_students_in_period(
        start=midnight(datetime.now()) - timedelta(days=7),
        end=midnight(datetime.now()),
    )
    # total_mila_users = int(data["user.mila.email"].nunique())
    grouped_data = data.groupby("cluster_name").aggregate(
        {
            "job_id": "nunique",
            "user.mila.email": "nunique",
            "gpu_utilization": ["mean", "std"],
        }
    )
    gpu_cost_per_user_per_cluster = data.groupby(["cluster_name", "user.mila.email"])[
        ["gpu_equivalent_cost", "allocated.gres_gpu"]
    ].sum()
    # TODO: look into naganuma.hiroki@mila.quebec on Mila (800 gpus*days)
    gpus_per_user = gpu_cost_per_user_per_cluster.groupby("cluster_name").describe()

    # table = Table(title="Overview by cluster (last 7 days)", expand=True)
    # TODO: Show Min / Mean / Median / Max GPUs per user?
    table.add_column("Cluster")
    table.add_column("# of jobs")
    table.add_column("GPU Util")
    table.add_column("Mila students using this cluster")
    table.add_column("GPUs days per user")

    for index, row in grouped_data.iterrows():
        assert isinstance(index, str)
        cluster = index
        mila_users = int(row["user.mila.email"]["nunique"])
        # used_gpus_pct = avail_gpu / total_gpu
        num_jobs = row["job_id"]["nunique"]
        gpu_util_mean = row["gpu_utilization"]["mean"]
        gpu_util_std = row["gpu_utilization"]["std"]
        pct_of_mila_users = mila_users / total_mila_users

        gpudays_per_user_here = gpus_per_user.xs(cluster)["gpu_equivalent_cost"]
        gpus_per_user_str = (
            # f"[{gpus_per_user_here['min'].days} {gpus_per_user_here['max'].days}] "
            f"({gpudays_per_user_here['mean'].days:.1f}±{gpudays_per_user_here['std'].days:.1f})"
        )
        table.add_row(
            cluster,
            str(num_jobs),
            # f"{avail_gpu} / {total_gpu} ({used_gpus_pct:.2%})",
            f"{_colorize_utilization(gpu_util_mean)} ± {gpu_util_std:.1%}",
            f"{mila_users} / {total_mila_users} ({pct_of_mila_users:.2%})",
            gpus_per_user_str,
        )


async def fill_jobs_view_datatable(
    datatable: DataTable, data: pd.DataFrame, n_to_show: int = 50, reverse: bool = False
) -> None:
    # Mock data (TODO: replace)
    if reverse:
        jobs = data.nsmallest(
            n=n_to_show,
            columns="rgu_equivalent_waste",
            keep="all",
        )
    else:
        jobs = data.nlargest(
            n=n_to_show,
            columns="rgu_equivalent_waste",
            keep="all",
        )

    # Doesn't really work.
    # submit_lines = await _preload_submit_lines(most_wasteful_jobs, n=n_to_show)

    table = datatable
    table.clear(columns=True)
    table.add_column("#")
    table.add_column("Job ID")
    table.add_column("Cluster")
    table.add_column("User")
    table.add_column("Elapsed time")
    table.add_column("Avg GPU Util")
    table.add_column("Requested Ressources")
    # TODO: Have the interval be displayed with the unit selected dynamically instead (e.g. "days" or "hours")
    table.add_column("Used/Wasted/Obstructed GPU days")
    # table.add_column("Wasted GPU/RGU days", justify="right")
    # table.add_column("Obstructed GPU days", justify="right")
    table.add_column("SubmitLine")
    # table.add_column("Workdir", justify="left")
    # table.add_column("submit command", justify="left")

    for i, (_index, row) in list(enumerate(jobs.iterrows(), start=1)):
        requested_cols = [col for col in jobs.columns if col.startswith("requested.")]
        job_id = str(row["job_id"])
        cluster_name = row["cluster_name"]
        user = row["user"]
        job_id_link = None

        if cluster_name == "tamia":
            job_id_link = f"https://portail.{cluster_name}.ecpia.ca/secure/jobstats/{user}/{job_id}/"
        elif cluster_name != "mila":
            job_id_link = f"https://portail.{cluster_name}.calculquebec.ca/secure/jobstats/{user}/{job_id}/"

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
        mila_user = (
            row["user.mila.email"].removesuffix("@mila.quebec")
            if isinstance(row["user.mila.email"], str)
            else f"[red]{row['user']} (missing mila email)[/red]"
        )
        # TODO: Only fetch the submit line on a keypress event instead!
        # submit_line = await get_submit_line(job_id, cluster_name)
        if cluster_name != "mila":
            submit_line = "Press 'f' to fetch the submit line."
        else:
            # TODO: Slows up the UI way too much!
            submit_line = "Press 'f' to fetch the submit line."
            # submit_line = get_submit_line, job_id, cluster_name
            # )

            # submit_line = await asyncio.get_running_loop().run_in_executor(
            #     None, get_submit_line, job_id, cluster_name
            # )
        # todo: use the returned key to update the row on keypress.
        _key = table.add_row(
            f"{i}",
            f"[link={job_id_link}]{job_id}[/link]" if job_id_link else job_id,
            cluster_name,
            mila_user,
            f"{row['elapsed_time']}",
            _colorize_utilization(row["gpu_utilization"]),
            # _requested_table,
            " ".join(f"{k}={v}" for k, v in requested_resources.items() if v),
            f"{row['gpu_equivalent_cost'].days} / [red]{row['gpu_equivalent_waste'].days}[/] / [red]{row['gpu_overbilling_cost'].days}",
            # f"[red]{row['gpu_equivalent_waste'].days} / {row['rgu_equivalent_waste'].days}",
            # f"[red]{row['gpu_overbilling_cost'].days}",
            # submit_lines[i - 1],
            submit_line,
            # IDEA: Show it as a Syntax block:
            # rich.syntax.Syntax(submit_line, lexer="bash"),
            # key=f"{cluster_name}-{job_id}",
        )


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


def _header_panel():
    """Display header with clock."""

    class _Header:
        def __rich__(self) -> Panel:
            grid = Table.grid(expand=True)
            grid.add_column(justify="center", ratio=1)
            grid.add_column(justify="right")
            grid.add_row(
                "[b]SARC[/b] Waste Monitoring",
                datetime.now().ctime().replace(":", "[blink]:[/]"),
            )
            return Panel(grid)

    return _Header()


@functools.cache
def get_submit_line(job_id: int | str, cluster_name: str) -> str:
    # Do this only once (and then reuse the connection)
    assert CACHE_DIR and CACHE_DIR.exists() and CACHE_DIR.is_dir()
    cache_file = CACHE_DIR / _get_cache_file_name(get_submit_line, job_id, cluster_name)
    if cache_file.exists():
        logger.debug(f"Loading submit line from cache: {cache_file}")
        return cache_file.read_text().strip()

    if CLUSTER_DOWN.get(cluster_name):
        return f"[red]Unable to SSH to {cluster_name} (is the cluster down?)[/red]"

    control_path = _get_controlpath(cluster_name)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    # if cluster_name in REMOTES:
    #     login_node = REMOTES[cluster_name]
    # else:
    #     login_node = await RemoteV2.connect(cluster_name, control_path=control_path)
    # Need to first establish the multiplexed SSH connection to the cluster, in case it uses 2FA.
    # If we didn't and used a single command, the login banner / 2FA message on DRAC would be
    # also included in the output of the command.
    submit_line = subprocess.getoutput(
        f"ssh -o ControlMaster=auto -o 'ControlPath={control_path}' -o ControlPersist=yes {cluster_name} sacct -j {job_id} --noheader -o submitline%300"
    )
    # submit_line = await login_node.get_output_async(
    #     f"sacct -j {job_id} --noheader -o submitline%300",
    # )
    submit_line = submit_line.strip()
    cache_file.write_text(submit_line)
    logger.debug(f"Saved submit line to cache: {cache_file}")
    return submit_line
    # return subprocess.getoutput(
    #     f"ssh {multiplexing_args} {cluster_name} sacct -j {job_id} --noheader -o submitline%300"
    # ).strip()


@functools.cache
@cached
def get_mila_students_in_period(start: datetime, end: datetime) -> int:
    students = get_users(latest=True)
    return len(
        set(
            user.mila.email
            for user in students
            if user.mila and user.mila.email
            if (user.record_start and user.record_start <= end)
            and (user.record_end is None or user.record_end >= start)
        )
    )


async def setup_multiplexed_ssh_conection(hostname: str):
    control_path = _get_controlpath(hostname)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    multiplexing_args = (
        f"-o ControlMaster=auto -o 'ControlPath={control_path}' -o ControlPersist=yes"
    )
    # Need to first establish the multiplexed SSH connection to the cluster, in case it uses 2FA.
    # If we didn't and used a single command, the login banner / 2FA message on DRAC would be
    # also included in the output of the command.
    await run_subprocess(
        f"ssh {multiplexing_args} {hostname} echo OK",
        stdout=subprocess.DEVNULL,
    )
    return control_path


def _get_controlpath(hostname: str) -> Path:
    return Path(
        paramiko.config.SSHConfig.from_path(Path.home() / ".ssh" / "config")
        .lookup(hostname)
        .get("controlpath", Path.home() / ".cache" / "ssh" / "%r@%h:%p")
    ).expanduser()

    # df.loc[slice_during_rgu_time, "allocated.cpu"] /= 1000.0

    # df.loc[df["job_id"] == 48738025, "allocated.cpu"] /= 1000
    # is_narval = df["cluster_name"] == "narval"

    # Here we do it for all timeframes.
    # df.loc[is_drac & outrageous_num_of_cpus, "allocated.cpu"] /= 1000.0


async def _get_output(cmd: str):
    with (
        tempfile.TemporaryFile(mode="w+") as out_file,
        tempfile.TemporaryFile(mode="w+") as err_file,
    ):
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=out_file,
            stderr=err_file,
        )
        _out, _err = await proc.communicate()
        out_file.seek(0)
        stdout = out_file.read()
        err_file.seek(0)
        stderr = err_file.read()

        assert proc.returncode is not None
    if proc.returncode != 0:
        raise _CalledProcessError(
            returncode=proc.returncode,
            cmd=cmd,
            output=stdout if stdout else None,
            stderr=stderr if stderr else None,
        )
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        output=stdout if stdout else None,
        stderr=stderr if stderr else None,
    )
