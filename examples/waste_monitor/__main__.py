# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "rich",
#     "sarc",
#     "textual",
#     "textual_plotext",
# ]
#
# [tool.uv.sources]
# sarc = { path = "../../" }
# ///
import asyncio
import logging
import subprocess

import rich.logging
import textual.logging

from .common_utils import CLUSTER_DOWN, get_available_clusters, setup_sarc_connection
from .ui import (
    WasteMonitor,
    get_data,
    logger,
    setup_multiplexed_ssh_conection,
    setup_torch_import_time_project,
)


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
    logger.info(
        "Successfully set up multiplexed SSH connection to sarc01-dev with port forwarding."
    )
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


async def async_main():
    _setup_logging(verbose=2)
    await setup_connections()
    get_data([cluster.cluster_name for cluster in get_available_clusters()])
    await setup_torch_import_time_project("mila")
    app = WasteMonitor()
    await app.run_async()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
