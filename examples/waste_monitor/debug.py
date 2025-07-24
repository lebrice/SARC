# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "sarc",
#     "rich",
# ]
#
# [tool.uv.sources]
# sarc = { git = "https://www.github.com/mila-iqia/SARC", rev = "457a66805975baab0dc85bd5893f1dd7661c1138" }
# ///
import dataclasses
import datetime
import functools
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Callable

import gifnoc
import pandas as pd
import rich
import rich.logging
import rich.pretty
import simple_parsing

from sarc.client.series import (
    compute_cost_and_waste,
    load_job_series,
    update_job_series_rgu,
)
from sarc.config import MTL

logger = logging.getLogger(__name__)
# Need to point to the sarc-client.yaml file. Might need to modify this path for your machine.
sarc_client_config_file = Path(__file__).parent / "config/sarc-client.yaml"
gifnoc.set_sources(sarc_client_config_file)

tomorrow = (datetime.datetime.now(tz=MTL) + datetime.timedelta(days=1)).replace(
    hour=0, minute=0, second=0, microsecond=0
)


@dataclasses.dataclass
class QueryOptions:
    start: str | None = None
    end: str | datetime.datetime = tomorrow
    job_id: int | None = None
    cluster: str | None = None
    user: str | None = None
    user_mila_email: str | None = None


def cache[**P, Out](func: Callable[P, Out]) -> Callable[P, Out]:
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        path = tempfile.gettempdir() / Path(
            (
                func.__name__
                + "_"
                + "-".join(map(str, args))
                + "-".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            ).replace(" ", "_")
        ).with_suffix(".pkl")
        if path.exists():
            logger.info(f"Loading previous result from {path}")

            with open(path, "rb") as file:
                return pickle.load(file)
        result = func(*args, **kwargs)
        with open(path, "wb") as file:
            logger.info(f"Save result to {path}")
            pickle.dump(result, file)
        return result

    return _wrapped


def _setup_logging(verbose: int):
    logging.basicConfig(
        handlers=[rich.logging.RichHandler()],
        format="%(message)s",
        level=logging.INFO,
        force=True,
    )

    if verbose == 0:
        logger.setLevel("WARNING")
    elif verbose == 1:
        logger.setLevel("INFO")
    else:
        logger.setLevel("DEBUG")


def main():
    _setup_logging(2)
    options = simple_parsing.parse(QueryOptions)
    # period = options.end - options.start if options.start else datetime.timedelta(days=90)

    jobs = cache(load_job_series)(
        start=options.start,
        end=options.end,
        job_id=options.job_id,
        user=options.user,
        cluster=options.cluster,
    )
    if options.user_mila_email:
        jobs = jobs.query(f"`user.mila.email` == '{options.user_mila_email}'")

    # Get rid of weird "lost" jobs
    jobs = jobs.query("(start_time - submit_time).dt.days <= 90")
    jobs = jobs.query("(end_time - start_time).dt.days <= 90")
    # remove jobs that started way before the start date of the query.
    if options.start:
        td = datetime.timedelta(days=30)
        end = (
            options.end
            if isinstance(options.end, datetime.datetime)
            else datetime.datetime.strptime(options.end, "%Y-%m-%d").astimezone(MTL)
        )
        # Remove jobs that started more than 30 days before the query start date. (weird lost jobs).
        jobs = jobs[
            (end - jobs["start_time"])  # type: ignore
            < min(td * 2, td + datetime.timedelta(days=30))
        ]

    jobs = update_job_series_rgu(jobs)
    # This makes it much easier to understand the compute times later (gpu days for instance).
    jobs = jobs.assign(elapsed_time=pd.to_timedelta(jobs["elapsed_time"], unit="s"))
    jobs = compute_cost_and_waste(jobs)
    # Add "requested.gres_rgu", also makes sense to have.
    rgu_per_gpu = jobs["allocated.gres_rgu"] / jobs["allocated.gres_gpu"]
    jobs = jobs.assign(
        **{
            "requested.gres_rgu": jobs["requested.gres_gpu"] * rgu_per_gpu,
            # Add the same cost/waste stats for RGUs.
            "rgu_cost": jobs["gpu_cost"] * rgu_per_gpu,
            "rgu_waste": jobs["gpu_waste"] * rgu_per_gpu,
            "rgu_equivalent_cost": jobs["gpu_equivalent_cost"] * rgu_per_gpu,
            "rgu_equivalent_waste": jobs["gpu_equivalent_waste"] * rgu_per_gpu,
            "rgu_overbilling_cost": jobs["gpu_overbilling_cost"] * rgu_per_gpu,
        }
    )
    average = jobs.aggregate(
        {
            "cpu_utilization": "mean",
            "gpu_utilization": "mean",
            "gpu_sm_occupancy": "mean",
            "gpu_power": "mean",
            "requested.gres_gpu": "sum",
            "elapsed_time": "sum",
            "job_id": "nunique",
            "cluster_name": "unique",
            **{
                key: "sum"
                for compute_type in [
                    "cpu",
                    "gpu",
                    "rgu",
                ]
                for key in [
                    f"{compute_type}_cost",
                    f"{compute_type}_waste",
                    f"{compute_type}_equivalent_cost",
                    f"{compute_type}_equivalent_waste",
                    f"{compute_type}_overbilling_cost",
                ]
            },
        }
    )
    print("Aggregated job statistics:")
    rich.pretty.pprint(average.to_dict())

    # uncomment to show the jobs:
    # for job in jobs.to_dict(orient="records"):
    #     rich.pretty.pprint(job)


def _show_first_entry(df: pd.DataFrame):
    return rich.pretty.pretty_repr(df.to_dict(orient="records")[0])


if __name__ == "__main__":
    main()
