from __future__ import annotations

import asyncio
import collections
import csv
import enum
import functools
import getpass
import logging
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Protocol, Sequence

import numpy as np
import pandas as pd
import paramiko
import paramiko.config
from rich.panel import Panel
from rich.table import Table
from textual import work
from textual.app import App, ComposeResult
from textual.containers import (
    Horizontal,
    HorizontalScroll,
    Vertical,
    VerticalScroll,
)  # noqa
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    RichLog,
)
from textual.worker import Worker
from textual_plotext import PlotextPlot

from sarc.client.job import SlurmState
from sarc.client.users.api import get_users
from sarc.config import MTL

from .common_utils import (
    CACHE_DIR,
    CLUSTER_DOWN,
    FilteringOptions,
    _CalledProcessError,
    _get_cache_file_name,
    cache_results_to_file,
    get_available_clusters,
    midnight,
    run_subprocess,
)
from .sarc_patches import get_clean_sarc_data

logger = logging.getLogger(__name__)

# alerts: list[Alert] = [
#     unused_gpus_alert,
# ]


class WasteMonitor(App):
    TITLE = "SARC Waste Monitoring"
    SUB_TITLE = "Data from the last 7 days. Last update: TODO"
    # For live editing the CSS:
    __CSS_PATH = Path(__file__).parent / "ui.tcss"
    CSS = """\
    Screen {
        align: center middle;
        padding: 1;
    }

    #buttons {
        height: 3;
        width: auto;
    }

    ContentSwitcher {
        border: round $primary;
        height: 1fr;
    }

    #cluster_overview {
        padding: 1;
        layout: grid;
        grid-size: 2;
    }

    #cluster_overview_table {
        column-span: 1;
        padding: 1;
    }
    #scratch_monitor {
        column-span: 1;
        padding: 1;
    }

    #overview_log {
        column-span: 2;
        align: center bottom;
    }
    """
    clusters: reactive[set[str]] = reactive(set())
    data: pd.DataFrame | None = None
    full_data: pd.DataFrame | None = None
    # alerts: reactive[Sequence[tuple[pd.DataFrame, str, Severity]]] = reactive(())
    # stuff = [alert(data) for alert in alerts]

    BINDINGS = [
        # ("f", "get_submit_line", "Get job submit line"),
    ]

    def compose(self) -> ComposeResult:
        # Todo: time doesn't update properly.
        # yield Static(_header_panel(), id="header_panel")
        yield Header(show_clock=True, id="header")
        with Horizontal(id="buttons"):
            yield Button("Overview", id="cluster_overview_button")
            yield Button("User View", id="user_view_button")
            yield Button("Worst Jobs", id="worst_jobs_button")
            yield Button("Best Jobs", id="best_jobs_button")
            with HorizontalScroll():
                clusters = [c.cluster_name for c in get_available_clusters()]
                for cluster in clusters:
                    yield Checkbox(
                        label=cluster.capitalize(),
                        value=True,
                        id=cluster,
                        name=f"{cluster}_checkbox",
                        tooltip=f"Include or Exclude data from the {cluster.capitalize()} cluster.",
                    )
                    # clusters = self.clusters | {cluster.cluster_name}
                other_clusters = [
                    c
                    for c in ["tamia", "rorqual", "fir", "nibi", "killarney", "vulcan"]
                    if c not in clusters
                ]
                for other_cluster in other_clusters:
                    yield Checkbox(
                        label=f"[strike]{other_cluster.capitalize()}[/]",
                        value=False,
                        id=other_cluster,
                        name=f"{other_cluster}_checkbox",
                        disabled=True,
                        tooltip=f"{other_cluster.capitalize()} cluster has not been added to SARC yet.",
                    )
                # self.clusters = clusters
            # yield Input(placeholder="User to query for")

        with ContentSwitcher(initial="cluster_overview"):
            # with VerticalScroll(id="cluster_overview"):
            with Vertical(id="cluster_overview"):
                # with Grid():  # with Horizontal():
                yield DataTable(
                    id="cluster_overview_table", name="Cluster Overview Table"
                )
                yield ScratchMonitorWidget(id="scratch_monitor")
                # todo: make this smaller (at the bottom of the screen)
                yield RichLog(
                    # max_lines=5,
                    highlight=True,
                    markup=True,
                    id="overview_log",
                )
                # yield DataTable(id="alerts_table")
            with VerticalScroll(id="user_view"):
                yield DataTable(id="user_view_table")
            with VerticalScroll(id="worst_jobs"):
                yield DataTable(id="worst_jobs_table")
            with VerticalScroll(id="best_jobs"):
                yield DataTable(id="best_jobs_table")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.query_one(ContentSwitcher).current = event.button.id.removesuffix(
            "_button"
        )

    async def action_get_submit_line(self, cluster_name: str, job_id: str) -> None:
        """Get the job submit line for the currently selected job."""
        current_panel = self.query_one(ContentSwitcher).current
        if current_panel not in {"worst_jobs", "best_jobs"}:
            return
        row_key = f"{cluster_name}_{job_id}"
        tables_to_update = [
            table for table in self.query(DataTable) if row_key in table.rows
        ]
        assert tables_to_update, "can't update a job that isnt in any of the tables!"
        current_submit_line = tables_to_update[0].get_row(row_key)[-1]

        if "submit_line" not in current_submit_line:
            # Submit line was already fetched for this job.
            return

        submit_line = await get_submit_line(cluster_name=cluster_name, job_id=job_id)
        self.query_one(RichLog).write(
            f"[{datetime.now()}] Retrieved submit_line for selected job: {cluster_name=}, {job_id=}: {submit_line!r}"
        )
        for table_to_update in tables_to_update:
            # Update the submit line in the table.
            table_to_update.update_cell(
                row_key=row_key,
                column_key="submit_line",
                value=submit_line,
                update_width=True,
            )

    def on_ready(self) -> None:
        self.clusters = set(c.cluster_name for c in get_available_clusters())
        self.update_jobs_dataframe()
        self.measure_scratch_torch_import_time()
        # self.populate_ui()
        self.set_interval(60, self.measure_scratch_torch_import_time)
        self.set_interval(5 * 60, self.update_jobs_dataframe)
        # assert self.full_data is not None
        # self.populate_ui(self.full_data)

    #     self.update_data()
    #     await self.populate_ui()
    #     # self.set_interval(5 * 60, self.update_data_and_ui)

    # def on_mount(self) -> None:
    #     # self.data = get_data()
    #     self.update_jobs_dataframe()
    #     # self.populate_ui()
    #     # self.update_alerts()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""
        self.log(event)
        # self.query_one(RichLog).write(
        #     f"{event.state.name}: ({event.worker})"
        #     # if event.state in (WorkerState.PENDING, WorkerState.RUNNING)
        #     # else event.state
        # )

    # @on(Checkbox.Changed)
    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        assert event.checkbox.id
        cluster = event.checkbox.id
        if event.value:
            # self.query_one(RichLog).write(f"Added {cluster} cluster")
            self.clusters = self.clusters | {cluster}
            self.notify(
                title="Cluster Selected",
                message=f"Selected {cluster} cluster.",
                timeout=5,
            )
        else:
            # self.query_one(RichLog).write(f"Unselected cluster {cluster}")
            self.clusters = self.clusters.difference({cluster})
            self.notify(
                title="Cluster Unselected",
                message=f"Unselected {cluster} cluster.",
                timeout=5,
            )
        # self.query_one(RichLog).write(rich.pretty.Pretty(sorted(self.clusters)))
        # self.sub_title = f"Data for clusters {list(self.clusters)}"
        # self.update_jobs_dataframe()
        assert self.full_data is not None
        self.data = self.full_data[self.full_data["cluster_name"].isin(self.clusters)]
        self.query_exactly_one("#worst_jobs_table", DataTable).clear()
        self.query_exactly_one("#best_jobs_table", DataTable).clear()
        self.populate_ui(self.data)
        # self.clear_notifications()

        # self.update_jobs_dataframe()

    @work(exclusive=True, group="data", thread=True)
    async def update_jobs_dataframe(self) -> None:
        full_data = get_data()
        self.call_from_thread(self.set_full_data, full_data)
        self.call_from_thread(self.populate_ui, full_data)
        # return self.data
        # self.data = data
        # await self.populate_ui(data)

    def set_full_data(self, data: pd.DataFrame) -> None:
        """Set the full data and update the UI."""
        self.full_data = data
        self.sub_title = f"Data from the last 7 days. Last update: {datetime.now()}"
        self.query_exactly_one(RichLog).write(
            f"[{datetime.now()}] - Updated SARC data."
        )

    # @work(exclusive=True)
    # async def update_alerts(self) -> None:
    #     self.alerts = await asyncio.gather(*(alert(self.data) for alert in alerts))

    def populate_ui(self, data: pd.DataFrame) -> None:
        self.data = data
        # TODO: The multiplexed SSH connections to all clusters should already be setup before this is launched
        # to avoid the SSH 2FA prompts messing up the UI.
        cluster_overview_table = self.query_exactly_one(
            "#cluster_overview_table", DataTable
        )
        fill_cluster_overview_table(cluster_overview_table, data)

        # await _setup_torch_import_test("mila")

        # alerts_table = self.query_exactly_one("#alerts_table", DataTable)
        # fill_alerts_table(alerts_table, alert_results=self.alerts, data=data)

        user_table = self.query_exactly_one("#user_view_table", DataTable)
        user_table.cursor_type = "row"
        fill_waste_overview_datatable(user_table, data)

        worst_jobs_table = self.query_exactly_one("#worst_jobs_table", DataTable)
        worst_jobs_table.cursor_type = "row"
        fill_jobs_view_datatable(worst_jobs_table, data)
        for row_key, row in list(worst_jobs_table.rows.items()):
            if row_key.value:
                cluster, _, job_id = row_key.value.partition("_")
                if cluster not in self.clusters:
                    worst_jobs_table.remove_row(row_key)
                # submit_line = get_submit_line(job_id=job_id, cluster=cluster)
                # worst_jobs_table.update_cell(
                #     row_key=row_key, column_key="submit_line", value=submit_line
                # )

        best_jobs_table = self.query_exactly_one("#best_jobs_table", DataTable)
        best_jobs_table.cursor_type = "row"
        fill_jobs_view_datatable(best_jobs_table, data, reverse=True)
        for row_key, row in list(best_jobs_table.rows.items()):
            if row_key.value:
                cluster, _, job_id = row_key.value.partition("_")
                if cluster not in self.clusters:
                    best_jobs_table.remove_row(row_key)

    @work(exclusive=True, group="scratch")
    async def measure_scratch_torch_import_time(self) -> None:
        """Measure the time it takes to import torch from the scratch directory."""
        # This is a placeholder for the actual implementation.
        # You would run a command like `time python -c "import torch"` in the scratch directory.
        # For now, we will just log a message.
        logger.debug("Measuring scratch torch import time...")
        scratch_widget = self.query_exactly_one(ScratchMonitorWidget)
        await scratch_widget.measure_and_save()

        self.refresh()

    # def on_mouse_move(self, event: events.MouseMove) -> None:
    #     # self.screen.query_one(RichLog).write(event)
    #     pass


previous_torch_import_time_results_file = "mila_scratch_torch_import_time.csv"


class ScratchMonitorWidget(Widget):
    """A widget to show the import time for torch."""

    import_time: reactive[float | None] = reactive(None)
    import_time_ema: reactive[float | None] = reactive(None)
    vals: reactive[collections.deque[tuple[datetime, float]]] = reactive(
        collections.deque(maxlen=100)
    )
    min_val: reactive[float | None] = reactive(None)
    max_val: reactive[float | None] = reactive(None)

    def compose(self):
        """Compose the widget."""
        yield PlotextPlot(id="scratch_torch_import_time_plot")

    period: ClassVar[timedelta] = timedelta(minutes=5)
    previous_results_file: ClassVar[str] = (
        "/network/scratch/n/normandf/torch_import_times.csv"
    )

    async def on_mount(self) -> None:
        previous_times = await self.get_previous_import_times()
        for dt, time, _who in previous_times:
            self.add_value(time, when=dt)
        if previous_times:
            self.update_ui()
        # self._plot(clear=False)

    @staticmethod
    def unzip[A, B, C](v: list[tuple[A, B, C]]) -> tuple[list[A], list[B], list[C]]:
        """Unzip a list of tuples into three lists."""
        if not v:
            return [], [], []
        tuples = zip(*v)
        return tuple(list(t) for t in tuples)  # type: ignore

    async def measure_and_save(self):
        previous_results = await self.get_previous_import_times(n=1)

        """
        Look at the entries for the past 10 minutes.
        The "person" (userid_programid) that wrote the most is the "writer".
        Everyone else is a reader. In a tie, use first alphabetically.
        """
        current_writer: str
        if previous_results:
            last_10_minutes: list[tuple[datetime, float, str]] = []
            ten_minutes_ago = datetime.now() - timedelta(minutes=10)
            for when, time, userid in previous_results:
                if when >= ten_minutes_ago:
                    last_10_minutes.append((when, time, userid))
            most_common_writers = collections.Counter(
                userid for _, _, userid in last_10_minutes
            ).most_common(2)
            most_common_count = most_common_writers[0][1]
            # re-sort a potential tie for first place in alphabetical order.
            current_writer = sorted(
                [
                    writer
                    for writer, count in most_common_writers
                    if count == most_common_count
                ]
            )[0]
        else:
            current_writer = self.user_id

        if current_writer != self.user_id:
            assert previous_results
            sample_datetime, measured_time, _who = previous_results[-1]
            logger.info(
                f"Not measuring torch import time, reusing sample from {current_writer} ago."
            )
            self.app.query_exactly_one(RichLog).write(
                f"[{sample_datetime}] - SCRATCH import time: {measured_time}s (read from shared file)"
            )
            self.add_value(measured_time, when=sample_datetime)
        else:
            if not previous_results:
                logger.info("No previous results found. Starting to write.")
            else:
                logger.info(
                    "This process is measuring and writing the torch import time for all other readers."
                )

            measured_time = await get_torch_import_time("mila")
            assert measured_time is not None
            sample_datetime = datetime.now()
            self.add_value(measured_time, when=sample_datetime)
            await self.save_result(measured_time, when=sample_datetime)
            self.app.query_exactly_one(RichLog).write(
                f"[{sample_datetime}] - SCRATCH import time: {measured_time}s (saved to shared file)"
            )
        self.update_ui()
        return (sample_datetime, measured_time)

    @functools.cached_property
    def user_id(self) -> str:
        # get the system user name and programid
        return f"{getpass.getuser()}_{os.getpid()}"

    async def get_previous_import_times(self, n: int = 100):
        previous_results_lines = csv.reader(
            (
                await _get_output(f"ssh mila tail -n {n} {self.previous_results_file}")
            ).splitlines(keepends=True)
        )
        results: list[tuple[datetime, float, str]] = []
        for row in previous_results_lines:
            dt = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f")
            time = float(row[1])
            userid = row[2].strip()
            results.append((dt, time, userid))
        return results

    async def save_result(self, time: float, when: datetime | None = None) -> None:
        """Save the results to a shared file."""
        if when is None:
            when = datetime.now()
        line_to_add = f"{when.strftime('%Y/%m/%d %H:%M:%S.%f')},{time},{self.user_id}"
        await run_subprocess(
            f"""ssh mila 'echo "{line_to_add}" >> {self.previous_results_file}'"""
        )

    def add_value(self, time: float, when: datetime | None = None) -> None:
        """Add a value to `self.vals` and update the running statistics."""
        if when is None:
            when = datetime.now()

        if self.import_time is None:
            assert self.import_time_ema is None
            self.import_time = time
            self.import_time_ema = time
        else:
            assert self.import_time_ema is not None
            # Exponential moving average
            alpha = 0.1
            self.import_time_ema = self.import_time_ema * (1 - alpha) + time * alpha
            self.import_time = time

        self.vals.append((when, time))
        _, times = zip(*self.vals)

        # Update in a way that preserves the min and max over all time, not just last 100
        if self.min_val is None and self.max_val is None:
            self.min_val = min(times)
            self.max_val = max(times)
        else:
            assert self.min_val is not None and self.max_val is not None
            self.min_val = min(min(times), self.min_val)
            self.max_val = max(max(times), self.max_val)
        # todo: unsure if it is a good idea to do this for every new value.
        # self.vals = self.vals.copy()

    def replot(self) -> None:
        """Set up the plot."""
        self._plot(clear=True)
        self.refresh()

    def _plot(self, clear: bool = True):
        plt = self.query_one(PlotextPlot).plt
        if clear:
            plt.clear_data()
        plt.date_form(input_form="Y/m/d H:M:S", output_form="H:M:S")
        times: tuple[datetime, ...]
        vals: tuple[float, ...]
        if self.vals:
            times, vals = zip(*self.vals)
            stimes = [t.strftime("%Y/%m/%d %H:%M:%S") for t in times]
            plt.bar(stimes, vals, label="$SCRATCH (Mila)", minimum=self.min_val)
            plt.title(
                f"Torch import time (min={self.min_val:.2f}, max={self.max_val:.2f}) (WIP: hover to see updated values)"
            )
        else:
            plt.title(
                "Torch import time: loading... (WIP: hover in a few seconds to see updated values)"
            )

    def update_ui(self) -> None:
        assert self.vals
        now, time = self.vals[-1]
        # now = datetime.now()
        # self.add_value(time, when=now)
        _when, times = zip(*self.vals)

        std = np.std(times)
        assert self.min_val is not None
        assert self.max_val is not None

        if time > self.min_val + std:
            self.app.query_exactly_one(RichLog).write(
                f"[{now}] - [yellow]$SCRATCH is slow on the Mila cluster! {time=:.2f}s vs {self.min_val=:.2f}s[/yellow]"
            )
            self.notify(
                title="$SCRATCH is slow!",
                message=(
                    f"$SCRATCH on Mila cluster is slower than usual: "
                    f"{time=:.2f}s vs {self.min_val=:.2f}s"
                ),
                severity="warning",
                timeout=300,
            )

        logger.info(f"Torch import time: {self.import_time}")
        logger.info(f"EMA: {self.import_time_ema}")
        logger.info(f"Average: {np.mean(times)}")
        self.vals = self.vals.copy()  # to trigger a reactive update.
        self.replot()


async def setup_torch_import_time_project(
    hostname: str, remote_dir: str = "$SCRATCH/torch_import_test"
):
    uv = await get_uv_path(hostname)
    if not uv:
        logger.warning(
            f"Could not find the `uv` executable on {hostname}. "
            "Please ensure that UV is installed and available in the PATH."
        )
        return
    proc = await run_subprocess(
        f"ssh {hostname} '{uv} init {remote_dir} --python=3.12'", check=False
    )
    if proc.returncode != 0:
        if "Project is already initialized" in proc.stderr:
            logger.info(f"Project {remote_dir} already exists on {hostname}.")
        else:
            raise _CalledProcessError.from_completed(proc)
    await run_subprocess(
        f"ssh {hostname} '{uv} add --project={remote_dir} torch numpy'", check=False
    )


async def get_torch_import_time(
    hostname: str, remote_dir: str = "$SCRATCH/torch_import_test"
) -> float | None:
    # TODO: Can't for the life of me figure out how to make a login shell work over ssh with async.
    # The best I can do atm is to assume that UV is at ~/.local/bin/uv and check that it is.
    uv = await get_uv_path(hostname)
    if not uv:
        logger.warning(
            f"Could not find the `uv` executable on {hostname}. "
            "Please ensure that UV is installed and available in the PATH."
        )
        return None
    command = (
        f"ssh {hostname} '{uv} run --project={remote_dir} --directory={remote_dir} "
        'python -c "import time; start=time.time(); import torch; print(time.time()-start)"\''
    )
    result = await run_subprocess(command)
    return float(result.stdout.strip())


async def get_uv_path(hostname: str) -> str | None:
    with tempfile.TemporaryFile(mode="w+") as temp_file:
        logger.info("Finding the `uv` executable on %s", hostname)
        _proc = await asyncio.create_subprocess_shell(
            f"ssh {hostname} bash -l which uv", stdout=temp_file
        )
        uv = await _proc.communicate()
        temp_file.seek(0)
        uv = temp_file.read().strip()

    return uv


def get_data(
    clusters: Sequence[str] = (),
    # users: Sequence[str] = (),
):
    midnight_tonight = midnight(datetime.now()) + timedelta(days=1)
    return _cached_get_data(
        midnight_tonight=midnight_tonight,
        clusters=tuple(sorted(clusters)),  # tuple(sorted(users))
    )


@functools.lru_cache(maxsize=16)
def _cached_get_data(
    midnight_tonight: datetime,
    period: timedelta = timedelta(days=7),
    clusters: tuple[str, ...] = (),
    # users: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Cached function to get the data."""
    options = FilteringOptions(
        start=(midnight_tonight - period),
        end=midnight_tonight,
        user=(),
        clusters=tuple(sorted(set(clusters))),
    )
    # This is cached as well:
    data = cache_results_to_file(get_clean_sarc_data)(options)
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


def fill_alerts_table(
    table: DataTable,
    alert_results: Sequence[tuple[pd.DataFrame, str, Severity]],
    data: pd.DataFrame,
) -> None:
    # alerts_table = Table(title="Alerts")
    table.clear(columns=True)
    table.add_column("Alert Time")
    table.add_column("Job Period start")
    table.add_column("cluster")
    table.add_column("Job ID")
    table.add_column("User")
    table.add_column("severity")
    table.add_column("Description")

    jobs, descriptions, severities = zip(*alert_results)

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


def fill_jobs_view_datatable(
    table: DataTable,
    data: pd.DataFrame,
    n_to_show: int = 50,
    reverse: bool = False,
) -> None:
    # Mock data (TODO: replace)
    if reverse:
        jobs = data.nsmallest(
            n=n_to_show,
            columns="rgu_equivalent_waste",
            # keep="all",
        )
    else:
        jobs = data.nlargest(
            n=n_to_show,
            columns="rgu_equivalent_waste",
            # keep="all",
        )

    # Doesn't really work.
    # submit_lines = await _preload_submit_lines(most_wasteful_jobs, n=n_to_show)

    # table = table.clear(columns=True)
    column_key_to_labels = {
        "index": "#",
        "job_id": "Job ID",
        "cluster_name": "Cluster",
        "user": "User",
        "elapsed_time": "Elapsed time",
        "gpu_utilization": "Avg GPU Util",
        "requested.gres": "Requested Ressources",
        "gpu_days": "Used/Wasted/Obstructed GPU days",
        "submit_line": "SubmitLine",
    }
    # TODO: Have the interval be displayed with the unit selected dynamically instead (e.g. "days" or "hours")

    if not table.columns:
        column_keys = [
            table.add_column(label, key=key)
            for key, label in column_key_to_labels.items()
        ]
    else:
        column_keys = column_key_to_labels.keys()
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
        submit_line = f"[@click=app.get_submit_line('{cluster_name}','{job_id}')]Click to get submit line[/]"
        # if cluster_name != "mila":
        # submit_line = "Press 'f' to fetch the submit line."
        # else:
        # TODO: Slows up the UI way too much!
        # submit_line = "Press 'f' to fetch the submit line."
        # submit_line = get_submit_line, job_id, cluster_name
        # )

        # submit_line = await asyncio.get_running_loop().run_in_executor(
        #     None, get_submit_line, job_id, cluster_name
        # )
        row_key = f"{cluster_name}_{job_id}"

        row_data = [
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
        ]

        if row_key in table.rows:
            existing_row_data = table.get_row(row_key)
            assert len(existing_row_data) == len(row_data) == len(column_keys)
            for col, old_data, new_data in zip(
                column_keys, existing_row_data, row_data
            ):
                if col == "submit_line":
                    if "submit line" in new_data:
                        continue  # keep old data with actual submit line!
                if old_data != new_data:
                    logger.debug(
                        f"Updating {col} for {row_key}: {old_data} -> {new_data}"
                    )
                    table.update_cell(row_key, column_key=col, value=new_data)
        else:
            #     # datatable.get_row_index(row_key)
            #     table.remove_row(row_key)
            # todo: use the returned key to update the row on keypress.
            _key = table.add_row(
                *row_data,
                # IDEA: Show it as a Syntax block:
                # rich.syntax.Syntax(submit_line, lexer="bash"),
                key=row_key,
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


async def get_submit_line(job_id: int | str, cluster_name: str) -> str:
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
    proc = await run_subprocess(
        f"ssh -o ControlMaster=auto -o 'ControlPath={control_path}' -o ControlPersist=yes {cluster_name} sacct -j {job_id} --noheader -o submitline%300"
    )
    # submit_line = await login_node.get_output_async(
    #     f"sacct -j {job_id} --noheader -o submitline%300",
    # )
    submit_line = proc.stdout.strip()
    cache_file.write_text(submit_line)
    logger.debug(f"Saved submit line to cache: {cache_file}")
    return submit_line
    # return subprocess.getoutput(
    #     f"ssh {multiplexing_args} {cluster_name} sacct -j {job_id} --noheader -o submitline%300"
    # ).strip()


@functools.cache
@cache_results_to_file
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
    return stdout.strip()
