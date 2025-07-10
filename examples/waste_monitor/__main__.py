# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "rich",
#     "sarc",
#     "textual",
# ]
#
# [tool.uv.sources]
# sarc = { path = "../../" }
# ///
import logging
import os
import subprocess

import rich.logging
import textual.logging

# os.environ["PATH"] = os.environ["PATH"] + os.path.dirname(__file__)
from .common_utils import get_available_clusters
from .sarc_patches import CLUSTER_DOWN, setup_sarc_connection
from .ui import WasteMonitor
from .waste_utils import get_data, logger, setup_multiplexed_ssh_conection


def _setup_logging(verbose: int):
    logging.basicConfig(
        handlers=[
            rich.logging.RichHandler(show_time=False),
            textual.logging.TextualHandler(),
        ],
        format="%(message)s",
        level=logging.ERROR,
        force=True,
    )
    logging.getLogger("sarc").setLevel(logging.WARNING)
    logging.getLogger("examples").setLevel(logging.WARNING)

    if verbose == 0:
        logger.setLevel("WARNING")
    elif verbose == 1:
        logger.setLevel("INFO")
    else:
        logger.setLevel("DEBUG")


_setup_logging(verbose=2)
app = WasteMonitor()


async def main():
    _setup_logging(verbose=2)
    # print(await get_torch_import_time("mila"))
    app = WasteMonitor()
    await app.run_async()  # _data = get_data()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
