import datetime
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

import gifnoc
import pandas as pd
import rich.logging
import rich.pretty
import simple_parsing
import tqdm
from cache_utils import FilteringOptions, cache_results_to_file

from sarc.client.series import (
    compute_cost_and_waste,
    load_job_series,
    update_job_series_rgu,
)
from sarc.core.models.users import UserData
from sarc.core.models.validators import DateMatchError
from sarc.users.db import get_users

MTL = ZoneInfo("America/Montreal")


def midnight(dt: datetime.datetime, tz: ZoneInfo = MTL) -> datetime.datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz)


START_DATE = datetime.datetime(2025, 4, 1, tzinfo=MTL)
END_DATE = midnight(datetime.datetime.today())
USER_EMAILS = (Path(__file__).parent / "ivado_users.txt").read_text().splitlines()

args = simple_parsing.parse(
    FilteringOptions,
    default=FilteringOptions(
        start=START_DATE,
        end=END_DATE,
        clusters=(),  # all clusters
        user=(),  # all users
    ),
)

logging.basicConfig(
    level=logging.INFO if args.verbose else logging.WARNING,
    format="%(message)s",
    handlers=[rich.logging.RichHandler()],
    force=True,
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


all_users = get_users()
email_to_user_or_prof: dict[str, UserData] = {}
uuid_to_user = {u.uuid: u for u in all_users}
for user in get_users({"email": {"$in": USER_EMAILS}}):
    email_to_user_or_prof[user.email] = user
    supervisors = user.supervisor.values_in_range(START_DATE, END_DATE)
    for supervisor_uuid in supervisors:
        supervisor = uuid_to_user[supervisor_uuid]
        email_to_user_or_prof[supervisor.email] = supervisor

# All the 'users' (students + profs) that we care about.
# NOTE: We also include the compute usage of the profs supervising the students in the query,
# even if they aren't in the USER_EMAILS list.

df = cache_results_to_file(load_job_series)(**kwargs)
df = df[df["user.email"].isin(email_to_user_or_prof)]
# df = df[df["user.email"].isin(USER_EMAILS)]  # would ignore profs submitting jobs themselves.
profs_not_in_query = set(email_to_user_or_prof) - set(USER_EMAILS)
if profs_not_in_query:
    rich.print(
        "NOTE: The following users (probably profs) will also have their compute usage shown, "
        "even if they were not in the list of emails to query for (they supervised one or more "
        "researchers in the provided list of emails)."
    )
    rich.pretty.pprint(profs_not_in_query)
logger.debug(f"Number of users in the query: {len(email_to_user_or_prof)}")

df = compute_cost_and_waste(df)
df = update_job_series_rgu(df)


def _add_cost_waste_rgu(df: pd.DataFrame) -> pd.DataFrame:
    gpu_rgu_ratio = df["allocated.gres_rgu"] / df["allocated.gres_gpu"]
    return df.assign(
        rgu_cost=df["gpu_cost"] * gpu_rgu_ratio,
        rgu_waste=df["gpu_waste"] * gpu_rgu_ratio,
        rgu_equivalent_cost=df["gpu_equivalent_cost"] * gpu_rgu_ratio,
        rgu_equivalent_waste=df["gpu_equivalent_waste"] * gpu_rgu_ratio,
        rgu_overbilling_cost=df["gpu_overbilling_cost"] * gpu_rgu_ratio,
    )


df = _add_cost_waste_rgu(df)

# Have the same ordering as in the USER_EMAILS list.
# users = sorted(users, key=lambda u: USER_EMAILS.index(u.email))
# email_to_user = dict(zip(USER_EMAILS, users))


# Need to add the supervisor email column to the dataframe. This is tricky to compute:
# We have to go job by job, and find the supervisor of the associated user at the time the job ran.
def add_supervisor_email_column(df: pd.DataFrame) -> pd.DataFrame:
    job_supervisor_emails: list[str] = []
    most_recent_supervisor_for_user: dict[str, UUID] = {}

    for index, job_row in tqdm.tqdm(
        df.iterrows(), total=len(df), disable=not sys.stdout.isatty()
    ):
        job_start = job_row["start_time"]
        assert isinstance(job_start, datetime.datetime) and job_start.tzinfo is not None
        job_user_email = job_row["user.email"]
        job_user = email_to_user_or_prof[job_user_email]
        # Might get a DateMatchError if the user doesn't have a known supervisor at that time.
        try:
            supervisor_uuid = job_user.supervisor.get_value(job_start)
            most_recent_supervisor_for_user[job_user_email] = supervisor_uuid
        except DateMatchError:
            # "bill" this job to the user itself (i.e. consider them as their own supervisor).
            # This bills the compute done by a prof to themselves.
            if job_user_email not in most_recent_supervisor_for_user:
                logger.info(
                    f"User {job_user_email} doesn't have a known supervisor at {job_start}! Using themselves as their own supervisor."
                )
                # TODO: Check if this might do weird things with students that become a prof.
                supervisor_uuid = job_user.uuid
                most_recent_supervisor_for_user[job_user_email] = supervisor_uuid
            else:
                # logger.debug(
                #     f"Using the most recent known supervisor for {job_user_email} instead: {uuid_to_user[supervisor_uuid].email}"
                # )
                supervisor_uuid = most_recent_supervisor_for_user[job_user_email]

        supervisor_user = uuid_to_user[supervisor_uuid]
        supervisor_email = supervisor_user.email
        job_supervisor_emails.append(supervisor_email)
    df = df.assign(**{"supervisor.email": job_supervisor_emails})
    return df


df = add_supervisor_email_column(df)

# Group jobs by user, supervisor, and cluster.
grouped_by_user = df.groupby(["supervisor.email", "user.email", "cluster_name"])
stats = grouped_by_user.aggregate(
    {
        "cpu_cost": lambda c: c.sum() / seconds_in_year,
        "gpu_cost": lambda c: c.sum() / seconds_in_year,
        "rgu_cost": lambda c: c.sum() / seconds_in_year,
    }
)
# Compute the total amount of compute time used, wasted and overbilled by user.
# Print from worst offender to best usage.
result = stats.sort_values(ascending=False, by="rgu_cost")
print(result.to_markdown())
result.to_csv(f"ivado_query_{date.today()}.csv")
