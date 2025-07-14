import asyncio
import collections
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
import rich
import rich.pretty
from textual import work
from textual.app import App, ComposeResult
from textual.containers import (
    Grid,
    Horizontal,
    HorizontalScroll,
    Vertical,
    VerticalScroll,
)
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
from textual.worker import Worker, WorkerState, get_current_worker
from textual_plotext import PlotextPlot

from examples.waste_monitor.sarc_patches import CLUSTER_DOWN, setup_sarc_connection

from .common_utils import get_available_clusters
from .waste_utils import (
    Alert,
    Severity,
    fill_alerts_table,
    fill_cluster_overview_table,
    fill_jobs_view_datatable,
    fill_waste_overview_datatable,
    get_data,
    get_submit_line,
    get_torch_import_time,
    setup_multiplexed_ssh_conection,
    unused_gpus_alert,
)

logger = logging.getLogger(__name__)

alerts: list[Alert] = [
    unused_gpus_alert,
]


class WasteMonitor(App):
    TITLE = "SARC Waste Monitoring"
    SUB_TITLE = "Data from the last 7 days. Last update: TODO"
    # For live editing the CSS:
    # CSS_PATH = Path(__file__).parent / "waste_monitor.tcss"
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
        width: 90%;
        height: 1fr;
    }

    RichLog {
    }
    
    #cluster_overview {
        padding: 2 4;
    }

    #cluster_overview_table {
        align: center middle;
        padding: 1 2;
    }

    #alerts_table {
        padding: 1 2;
    }
    """
    clusters: reactive[set[str]] = reactive(
        set(c.cluster_name for c in get_available_clusters())
    )
    data: pd.DataFrame | None = None
    full_data: pd.DataFrame | None = None
    # alerts: reactive[Sequence[tuple[pd.DataFrame, str, Severity]]] = reactive(())
    # stuff = [alert(data) for alert in alerts]

    BINDINGS = [
        ("f", "get_submit_line", "Get job submit line"),
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
                for cluster in get_available_clusters():
                    yield Checkbox(
                        label=cluster.cluster_name.capitalize(),
                        value=True,
                        id=cluster.cluster_name,
                        name=f"{cluster.cluster_name}_checkbox",
                    )
                    # clusters = self.clusters | {cluster.cluster_name}
                # self.clusters = clusters
            # yield Input(placeholder="User to query for")

        with ContentSwitcher(initial="cluster_overview"):
            # with VerticalScroll(id="cluster_overview"):
            with VerticalScroll(id="cluster_overview"):
                with Grid():  # with Horizontal():
                    yield DataTable(id="cluster_overview_table")
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

    def action_get_submit_line(self) -> None:
        """Get the job submit line for the currently selected job."""
        current_panel = self.query_one(ContentSwitcher).current
        if current_panel not in {"worst_jobs", "best_jobs"}:
            return

        table = self.query_one(DataTable)
        # if not table.cursor_row:
        #     self.query_one(RichLog).write("No job selected.")
        #     return
        index = table.cursor_row
        row_key, col_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        self.query_one(RichLog).write(
            f"Current selected job: {row_key.value}, {col_key.value}"
        )
        return  # TODO: implement the rest of this here.
        assert row_key.value
        job_id, _, cluster = row_key.value.partition("_")
        submit_line = get_submit_line(job_id=job_id, cluster=cluster)
        table.update_cell(row_key=row_key, column_key="submit_line", value=submit_line)
        # if "submit_line" in job:
        # self.query_one(RichLog).write(f"Job submit line: {job['submit_line']}")
        # else:
        # self.query_one(RichLog).write("No submit line available for this job.")

    def on_ready(self) -> None:
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
        time = await get_torch_import_time("mila")
        assert time is not None
        scratch_widget = self.query_exactly_one(ScratchMonitorWidget)
        scratch_widget.update(time)

    # def on_mouse_move(self, event: events.MouseMove) -> None:
    #     # self.screen.query_one(RichLog).write(event)
    #     pass


class ScratchMonitorWidget(Widget):
    """A widget to show the import time for torch."""

    import_time: reactive[timedelta | None] = reactive(None)
    import_time_ema: reactive[timedelta | None] = reactive(None)
    vals: reactive[collections.deque[tuple[datetime, float]]] = reactive(
        collections.deque(maxlen=100)
    )

    def compose(self):
        """Compose the widget."""
        yield PlotextPlot()

    def on_mount(self) -> None:
        plt = self.query_one(PlotextPlot).plt
        plt.date_form("d/m/Y H:M:S")
        if self.vals:
            times: tuple[datetime, ...]
            times, vals = zip(*self.vals)
            stimes = [t.strftime("%d/%m/%Y %H:%M:%S") for t in times]
            plt.scatter(stimes, vals, marker="*", label="$SCRATCH (Mila)")
        plt.title("Torch import time")

    def replot(self) -> None:
        """Set up the plot."""
        plt = self.query_one(PlotextPlot).plt
        plt.clear_data()
        plt.date_form("d/m/Y H:M:S")
        if self.vals:
            times: tuple[datetime, ...]
            times, vals = zip(*self.vals)
            stimes = [t.strftime("%d/%m/%Y %H:%M:%S") for t in times]
            plt.scatter(stimes, vals, marker="*", label="$SCRATCH (Mila)")
        self.refresh()

    def update(self, time: timedelta) -> None:
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
        if time > self.import_time_ema:
            self.notify(
                title="$SCRATCH is slow!",
                message=(
                    f"$SCRATCH on Mila cluster is slower than usual: "
                    f"{time.total_seconds():.2}s vs {self.import_time_ema.total_seconds():.2f}s"
                ),
                severity="warning",
                timeout=300,
            )

        self.vals.append((datetime.now(), time.total_seconds()))
        logger.info(f"Torch import time: {self.import_time}")
        logger.info(f"EMA: {self.import_time_ema}")
        logger.info(f"Average: {self.import_time_ema}")
        self.replot()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""
        self.log(event)
        assert self.parent
        self.parent.query_one(RichLog).write(
            f"{event.state.name}: ({event.worker})"
            # if event.state in (WorkerState.PENDING, WorkerState.RUNNING)
            # else event.state
        )

    def on_ready(self) -> None:
        """Called when the widget is ready."""
        # self.query_exactly_one("#scratch_monitor_table", DataTable).add_columns(
        #     "Timestamp", "Torch import time"
        # )
        self.notify("Hello, from Textual!", title="Welcome")
        self.log(f"Import time: {self.measure_scratch_torch_import_time()}")
        self.set_interval(60, self.measure_scratch_torch_import_time)

    @work(exclusive=True)
    async def measure_scratch_torch_import_time(self) -> None:
        """Measure the time it takes to import torch from the scratch directory."""
        # This is a placeholder for the actual implementation.
        # You would run a command like `time python -c "import torch"` in the scratch directory.
        # For now, we will just log a message.
        logger.debug("Measuring scratch torch import time...")
        time = await get_torch_import_time("mila")
        assert time is not None
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
        logger.info(f"Torch import time: {self.import_time:T}")
        logger.info(f"Torch import time EMA: {self.import_time_ema:T}")
