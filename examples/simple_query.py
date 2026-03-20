import datetime
import logging
from datetime import timedelta
from pathlib import Path

import gifnoc
import rich.logging
import simple_parsing
from cache_utils import FilteringOptions, cache_results_to_file

from sarc.client.series import compute_cost_and_waste, load_job_series

args = simple_parsing.parse(
    FilteringOptions,
    default=FilteringOptions(
        start=datetime.datetime(2025, 1, 1),
        end=datetime.datetime(2026, 1, 1),
        clusters=(),  # all clusters
        user=(),  # all users
    ),
)

logging.basicConfig(
    level=logging.INFO if args.verbose else logging.WARNING,
    handlers=[rich.logging.RichHandler()],
    force=True,
    format="%(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(
    logging.DEBUG
    if args.verbose > 1
    else logging.INFO
    if args.verbose
    else logging.WARNING
)
# Ugly, but we need to tell SARC which config to use (doesn't default to the client one).
gifnoc.set_sources(Path(__file__).parent.parent / "config/sarc-client.yaml")

seconds_in_year = timedelta(days=365.25).total_seconds()
kwargs: dict = dict(start=args.start, end=args.end)
assert not args.user and not args.clusters, "no filtering by user or cluster for now"
logger.info(f"Querying job series with the following filters: {kwargs}")

df = cache_results_to_file(load_job_series)(**kwargs)
logger.debug(f"Number of entries: {len(df)}")
df = compute_cost_and_waste(df)

# Group jobs by user
grouped_by_user = df.groupby(["cluster_name"])
stats = grouped_by_user.aggregate(
    {
        "gpu_utilization": "mean",
        "cpu_utilization": "mean",
        "cpu_cost": lambda c: c.sum() / seconds_in_year,
        "gpu_cost": lambda c: c.sum() / seconds_in_year,
    }
)
# Compute the total amount of compute time used, wasted and overbilled by user.
# Print from worst offender to best usage.
print(stats.sort_values(ascending=False, by="gpu_utilization").to_markdown())
