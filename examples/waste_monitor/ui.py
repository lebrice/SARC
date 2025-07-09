import logging
from datetime import datetime, timedelta

import pandas as pd
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Checkbox, ContentSwitcher, DataTable, RichLog

from examples.waste_monitor.sarc_client import get_available_clusters
from examples.waste_monitor.waste_utils import (
    fill_alerts_table,
    fill_cluster_overview_table,
    fill_jobs_view_datatable,
    fill_waste_overview_datatable,
    get_data,
    get_torch_import_time,
)

logger = logging.getLogger(__name__)

class ValidateApp(App):
    CSS_PATH = "validate01.tcss"

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

    def render(self) -> str:
        """Render the widget."""
        if self.import_time is None:
            assert self.import_time_ema is None
            return "No import time measured yet."
        assert self.import_time_ema is not None
        return f"Import time: {self.import_time.total_seconds():.2f} seconds (EMA: {self.import_time_ema.total_seconds():.2f} seconds)"


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
            self.import_time_ema = (
                self.import_time_ema * (1-alpha) + time * alpha
            )
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

    def render(self) -> str:
        """Render the widget."""
        if self.import_time is None:
            assert self.import_time_ema is None
            return "No import time measured yet."
        assert self.import_time_ema is not None
        return f"Import time: {self.import_time.total_seconds():.2f} seconds (EMA: {self.import_time_ema.total_seconds():.2f} seconds)"


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
            self.import_time_ema = (
                self.import_time_ema * (1-alpha) + time * alpha
            )
            self.import_time = time
        logger.info(f"Torch import time: {self.import_time:T}")
        logger.info(f"Torch import time EMA: {self.import_time_ema:T}")


class RichLogApp(App):
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
    clusters: reactive[set[str]] = reactive(set())

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
                    self.clusters = self.clusters | {cluster.cluster_name}
            # yield Input(placeholder="User to query for")

        with ContentSwitcher(initial="cluster_overview"):
            # with VerticalScroll(id="cluster_overview"):
            with Vertical(id="cluster_overview"):
                with Horizontal():
                    yield DataTable(id="cluster_overview_table")
                    yield ScratchMonitorWidget(id="scratch_monitor")
                yield DataTable(id="alerts_table")
                yield RichLog(highlight=True, markup=True, id="overview_log")
            with VerticalScroll(id="user_view"):
                yield DataTable(id="user_view_table")
            with VerticalScroll(id="worst_jobs"):
                yield DataTable(id="worst_jobs_table")
            with VerticalScroll(id="best_jobs"):
                yield DataTable(id="best_jobs_table")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.query_one(ContentSwitcher).current = event.button.id.removesuffix(
            "_button"
        )

    def on_ready(self) -> None:
        self.update_data_and_ui()
        self.set_interval(5 * 60, self.update_data_and_ui)

    def on_mount(self) -> None:
        self.update_data_and_ui()

    # @on(Checkbox.Changed)
    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        text_log = self.query_one(RichLog)
        assert event.checkbox.id
        if event.value:
            self.clusters = self.clusters | {event.checkbox.id}
        else:
            self.clusters = self.clusters.difference({event.checkbox.id})
        # TODO: update the UI following changes to the data in an efficient way.
        text_log.write(rich.pretty.Pretty(sorted(self.clusters)))
        self.sub_title = f"Data for clusters {list(self.clusters)}"
        self.update_data_and_ui()

    @work(exclusive=True)
    async def update_data_and_ui(self) -> None:
        """Update the data and the UI."""
        data = get_data(list(self.clusters), ())
        await self.populate_ui(data)

    async def populate_ui(self, data: pd.DataFrame):
        # TODO: The multiplexed SSH connections to all clusters should already be setup before this is launched
        # to avoid the SSH 2FA prompts messing up the UI.
        cluster_overview_table = self.query_exactly_one(
            "#cluster_overview_table", DataTable
        )
        fill_cluster_overview_table(cluster_overview_table, data)

        # await _setup_torch_import_test("mila")

        alerts_table = self.query_exactly_one("#alerts_table", DataTable)
        await fill_alerts_table(alerts_table, data)

        user_table = self.query_exactly_one("#user_view_table", DataTable)
        user_table.cursor_type = "row"
        fill_waste_overview_datatable(user_table, data)

        worst_jobs_table = self.query_exactly_one("#worst_jobs_table", DataTable)
        worst_jobs_table.cursor_type = "row"
        await fill_jobs_view_datatable(worst_jobs_table, data)

        best_jobs_table = self.query_exactly_one("#best_jobs_table", DataTable)
        best_jobs_table.cursor_type = "row"
        await fill_jobs_view_datatable(best_jobs_table, data, reverse=True)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        # self.screen.query_one(RichLog).write(event)
        pass        
        