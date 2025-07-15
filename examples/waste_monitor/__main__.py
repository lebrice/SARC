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

from .common_utils import (
    CLUSTER_DOWN,
    get_available_clusters,
    run_subprocess,
    setup_sarc_connection,
)
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

    clusters = [c.cluster_name for c in get_available_clusters()]
    already_connected = await asyncio.gather(
        *[check_for_existing_multiplexed_connection(c) for c in clusters]
    )
    already_connected = dict(zip(clusters, already_connected))
    if any(already_connected.values()):
        logger.info(
            "Already connected to "
            + ", ".join(c for c, connected in already_connected.items() if connected)
        )
    if not all(already_connected.values()):
        logger.info(
            "Creating a new SSH connection (with ControlMaster) to the following clusters: "
            + ", ".join(
                c for c, connected in already_connected.items() if not connected
            )
        )

    # note: can't do all of them at once with asyncio.gather because of the 2FA prompts.
    for cluster in [c for c in clusters if not already_connected[c]]:
        if CLUSTER_DOWN.get(cluster):
            logger.info(f"Skipping {cluster} cluster which is supposedly down.")
            continue
        try:
            await setup_multiplexed_ssh_conection(cluster)
        except subprocess.CalledProcessError as err:
            logger.error(
                f"Failed to setup multiplexed SSH connection to {cluster}: {err}"
            )
            CLUSTER_DOWN[cluster] = True
        else:
            logger.info(f"Successfully set up multiplexed SSH connection to {cluster}")


async def check_for_existing_multiplexed_connection(hostname: str):
    proc = await run_subprocess(f"ssh -O check {hostname}", check=False)
    return proc.returncode == 0


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
