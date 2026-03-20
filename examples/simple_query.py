import datetime
from datetime import timedelta
from pathlib import Path

import gifnoc
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

# Ugly, but we need to tell SARC which config to use (doesn't default to the client one).
gifnoc.set_sources(Path(__file__).parent.parent / "config/sarc-client.yaml")

seconds_in_year = timedelta(days=365.25).total_seconds()
df = cache_results_to_file(load_job_series)(start=args.start, end=args.end)
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
