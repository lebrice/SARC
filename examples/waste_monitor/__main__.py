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
import logging
import subprocess

import rich.logging
import textual.logging

from .sarc_client import get_available_clusters
from .sarc_patches import CLUSTER_DOWN, setup_sarc_connection
from .ui import WasteMonitor
from .waste_utils import logger, setup_multiplexed_ssh_conection


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


async def main():
    _setup_logging(verbose=2)
    # print(await get_torch_import_time("mila"))

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

    # Calling the function so the results are saved in memory (thanks to functools.lru_cache).
    # _data = get_data()
    # for cluster in get_available_clusters():
    #     _cluster_data = get_data(clusters=[cluster.cluster_name])
    #     if not (_cluster_data.empty or CLUSTER_DOWN.get(cluster.cluster_name)):
    #         asyncio.run(fill_jobs_view_datatable(unittest.mock.Mock(), _cluster_data))

    app = WasteMonitor()
    await app.run_async()    # _data = get_data()

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())