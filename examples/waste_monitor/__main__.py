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
import sys
from pathlib import Path

import rich.logging
import textual.logging

# Get the absolute path of the directory containing the current script
current_dir = Path(__file__).resolve().parent

# Add the project's root directory (or any other relevant directory) to sys.path
# This allows importing modules from that added directory as if they were top-level packages.
project_root = current_dir.parent  # Adjust based on your project structure
sys.path.append(str(project_root))
from waste_monitor.common_utils import get_available_clusters
from waste_monitor.sarc_patches import CLUSTER_DOWN, setup_sarc_connection
from waste_monitor.ui import WasteMonitor
from waste_monitor.waste_utils import get_data, logger, setup_multiplexed_ssh_conection


def _setup_logging(verbose: int):
    logging.basicConfig(
        handlers=[
            rich.logging.RichHandler(show_time=False),
            textual.logging.TextualHandler(stderr=False),
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
# app = WasteMonitor()


async def setup_connections():
    await setup_sarc_connection()
    for cluster in get_available_clusters():
        if CLUSTER_DOWN.get(cluster.cluster_name):
            logger.info(f"Skipping {cluster} cluster which is supposedly down.")
            continue
        try:
            await setup_multiplexed_ssh_conection(cluster.cluster_name)
        except subprocess.CalledProcessError as err:
            logger.error(
                f"Failed to setup multiplexed SSH connection to {cluster.cluster_name}: {err}"
            )
            CLUSTER_DOWN[cluster.cluster_name] = True
        else:
            logger.info(
                f"Successfully set up multiplexed SSH connection to {cluster.cluster_name}"
            )


async def app():
    _setup_logging(verbose=2)
    await setup_connections()
    app = WasteMonitor()
    await app.run_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(app())
