import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
import rich
import rich.panel
import rich.pretty
import rich.text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Checkbox, ContentSwitcher, DataTable, RichLog
from textual.worker import Worker, WorkerState, get_current_worker

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
    TITLE = f"[b]SARC[/b] Waste Monitoring - {datetime.now().ctime().replace(':', '[blink]:[/]')}"
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
            with Vertical(id="cluster_overview"):
                # with Horizontal():
                yield DataTable(id="cluster_overview_table")
                yield ScratchMonitorWidget(id="scratch_monitor")
                # yield DataTable(id="alerts_table")
            with VerticalScroll(id="user_view"):
                yield DataTable(id="user_view_table")
            with VerticalScroll(id="worst_jobs"):
                yield DataTable(id="worst_jobs_table")
            with VerticalScroll(id="best_jobs"):
                yield DataTable(id="best_jobs_table")

        yield RichLog(highlight=True, markup=True, id="overview_log")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.query_one(ContentSwitcher).current = event.button.id.removesuffix(
            "_button"
        )

    def action_get_submit_line(self) -> None:
        """Get the job submit line for the currently selected job."""
        current_panel = self.query_one(ContentSwitcher).current
        if current_panel != "worst_jobs" and current_panel != "best_jobs":
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
        self.query_one(RichLog).write(
            f"{event.state.name}: ({event.worker})"
            # if event.state in (WorkerState.PENDING, WorkerState.RUNNING)
            # else event.state
        )

    # @on(Checkbox.Changed)
    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        assert event.checkbox.id
        if event.value:
            self.query_one(RichLog).write(f"Added {event.checkbox.id} cluster")
            self.clusters = self.clusters | {event.checkbox.id}
        else:
            self.query_one(RichLog).write(f"Unselected cluster {event.checkbox.id}")
            self.clusters = self.clusters.difference({event.checkbox.id})
        # self.query_one(RichLog).write(rich.pretty.Pretty(sorted(self.clusters)))
        # self.sub_title = f"Data for clusters {list(self.clusters)}"
        # self.update_jobs_dataframe()
        assert self.full_data is not None
        self.data = self.full_data[self.full_data["cluster_name"].isin(self.clusters)]
        self.query_one(RichLog).write(
            rich.pretty.Pretty(sorted(self.data["cluster_name"].unique()))
        )
        self.query_exactly_one("#worst_jobs_table", DataTable).clear()
        self.query_exactly_one("#best_jobs_table", DataTable).clear()
        self.populate_ui(self.data)
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


class ValidateApp(App):
    # CSS_PATH = "validate01.tcss"
    CSS = """\
    #buttons {
        dock: top;
        height: auto;
    }
    """
    count = reactive(0)

    def validate_count(self, count: int) -> int:
        """Validate value."""
        if count < 0:
            count = 0
        elif count > 10:
            count = 10
        return count

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Button("+1", id="plus", variant="success"),
            Button("-1", id="minus", variant="error"),
            id="buttons",
        )
        yield RichLog(highlight=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "plus":
            self.count += 1
        else:
            self.count -= 1
        self.query_one(RichLog).write(f"count = {self.count}")


class ScratchMonitorApp(App):
    # def compose(self):
    #     yield ScratchMonitor()
    import_time: reactive[timedelta | None] = reactive(None)
    import_time_ema: reactive[timedelta | None] = reactive(None)

    # def compose(self) -> ComposeResult:
    #     """Compose the widget."""
    # yield DataTable(id="scratch_monitor_table")

    def render(self):
        """Render the widget."""
        if self.import_time is None:
            assert self.import_time_ema is None
            return rich.panel.Panel(
                "No import time measured yet.", title="Scratch Monitor"
            )
        assert self.import_time_ema is not None
        return rich.panel.Panel(
            f"Import time: {self.import_time.total_seconds():.2f} seconds (EMA: {self.import_time_ema.total_seconds():.2f} seconds)",
            title="Scratch Monitor",
        )

    def on_ready(self) -> None:
        """Called when the widget is ready."""
        # self.query_exactly_one("#scratch_monitor_table", DataTable).add_columns(
        #     "Timestamp", "Torch import time"
        # )
        self.measure_scratch_torch_import_time()
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
        logger.info(f"Torch import time: {self.import_time}")
        logger.info(f"Torch import time EMA: {self.import_time_ema}")


class ScratchMonitorWidget(Widget):
    """A widget to show the import time for torch."""

    import_time: reactive[timedelta | None] = reactive(None)
    import_time_ema: reactive[timedelta | None] = reactive(None)

    # def compose(self) -> ComposeResult:
    #     """Compose the widget."""
    # yield DataTable(id="scratch_monitor_table")

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
        logger.info(f"Torch import time: {self.import_time}")
        logger.info(f"Torch import time EMA: {self.import_time_ema}")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""
        self.log(event)
        assert self.parent
        self.parent.query_one(RichLog).write(
            f"{event.state.name}: ({event.worker})"
            # if event.state in (WorkerState.PENDING, WorkerState.RUNNING)
            # else event.state
        )

    def render(self):
        """Render the widget."""
        if self.import_time is None:
            assert self.import_time_ema is None
            return rich.text.Text("No torch import time measured yet.")
        assert self.import_time_ema is not None
        return rich.text.Text(
            f"Torch import time on $SCRATCH: {self.import_time.total_seconds():.2f} seconds (EMA: {self.import_time_ema.total_seconds():.2f} seconds)"
        )

    def on_ready(self) -> None:
        """Called when the widget is ready."""
        # self.query_exactly_one("#scratch_monitor_table", DataTable).add_columns(
        #     "Timestamp", "Torch import time"
        # )
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
