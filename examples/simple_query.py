import datetime
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import gifnoc
import pandas as pd
import plotly.express as px
import rich.logging
import rich.pretty
import rich.progress
import rich.progress_bar
import simple_parsing
import tqdm
import tqdm.rich
from cache_utils import FilteringOptions, cache_results_to_file

from sarc.client.series import (
    compute_cost_and_waste,
    load_job_series,
    update_job_series_rgu,
)
from sarc.core.models.users import UserData
from sarc.core.models.validators import DateMatchError
from sarc.users.db import get_users

logger = logging.getLogger(__name__)
MTL = ZoneInfo("America/Montreal")

# Ugly, but we need to tell SARC which config to use (doesn't default to the client one).
gifnoc.set_sources(Path(__file__).parent.parent / "config/sarc-client.yaml")


def midnight(dt: datetime.datetime, tz: ZoneInfo = MTL) -> datetime.datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz)


START_DATE = datetime.datetime(2025, 4, 1, tzinfo=MTL)
END_DATE = midnight(datetime.datetime.today())
seconds_in_year = timedelta(days=365.25).total_seconds()


def _add_cost_waste_rgu(df: pd.DataFrame) -> pd.DataFrame:
    gpu_rgu_ratio = df["allocated.gres_rgu"] / df["allocated.gres_gpu"]
    return df.assign(
        rgu_cost=df["gpu_cost"] * gpu_rgu_ratio,
        rgu_waste=df["gpu_waste"] * gpu_rgu_ratio,
        rgu_equivalent_cost=df["gpu_equivalent_cost"] * gpu_rgu_ratio,
        rgu_equivalent_waste=df["gpu_equivalent_waste"] * gpu_rgu_ratio,
        rgu_overbilling_cost=df["gpu_overbilling_cost"] * gpu_rgu_ratio,
    )


def main():
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

    kwargs: dict = dict(start=args.start, end=args.end)
    if args.clusters:
        kwargs["cluster_name"] = {"$in": args.clusters}

    logger.info(f"Querying job series with the following filters: {kwargs}")

    # NOTE: we get the data from all users and then filter by user email.
    # This is better then trying to filter by cluster username (can't filter by user email directly).
    df = cache_results_to_file(load_job_series)(**kwargs)

    user_emails = args.get_users()
    # TODO: Re-enable if we want to filter the results by user email.
    if user_emails := args.get_users():
        df = df[df["user.email"].isin(user_emails)]
    if args.clusters:
        assert df["cluster_name"].isin(args.clusters).all(), (
            "Some clusters in the dataframe are not in the provided list of clusters to query for. This should not happen since we filter by cluster name in the query, but just in case..."
        )

    df = compute_cost_and_waste(df)
    df = update_job_series_rgu(df)
    df = _add_cost_waste_rgu(df)

    # Show a two pie charts:
    # - One with the number of unique clusters used by users overall, for example
    #   37% of users use one cluster, 21 % used 2 clusters, etc etc.
    # - Another with the number of users that are using each cluster.

    # temporarily overwrite the user.email column.
    df = temporarily_overwrite_user_email(args, df)

    clusters_used_by_user = df.groupby("user.email").aggregate(
        num_clusters=pd.NamedAgg("cluster_name", "nunique"),
        clusters=pd.NamedAgg("cluster_name", lambda x: set(x.unique())),
    )

    # Show, for users that only use 1 cluster, which clusters they use.
    for i in range(1, 3):
        single_cluster_users_data = (
            clusters_used_by_user.query(f"num_clusters == {i}")["clusters"]
            .explode()
            .value_counts()
        )
        s = "s" if i > 1 else ""
        print(single_cluster_users_data.to_markdown())
        fig = px.pie(
            single_cluster_users_data,
            values=single_cluster_users_data.values,
            names=single_cluster_users_data.index,
            title=f"Cluster{s} used by users that used exactly {i} cluster{s}",
        )
        fig.show()

    # Show, for users that only use 2 clusters, which pairs of clusters they use.
    num_clusters_used_plot_data = (
        df.groupby("user.email")["cluster_name"].nunique().value_counts().sort_index()
    )
    # Create a figure with two plots side by side.

    # Display a table showing users using 1+ clusters, 2+ clusters, etc.
    # Cumulative sum of users using at least x clusters.
    total_users = num_clusters_used_plot_data.sum()
    for i in range(1, num_clusters_used_plot_data.index.max() + 1):
        num_users_using_at_least_i_clusters = num_clusters_used_plot_data[
            num_clusters_used_plot_data.index >= i
        ].sum()
        logger.info(
            f"{num_users_using_at_least_i_clusters} users used at least {i} cluster(s) ({num_users_using_at_least_i_clusters / total_users:.1%})"
        )

    # TODO: To show the users that used '0' clusters, we have to use the
    # get_users() function, since they won't have entries in the job dataframe!
    # all_users = get_users()
    # user_ids_in_dataframe = df["user.uuid"].unique()
    # active_users = [u for u in all_users if user_is_active(u, args.start, args.end)]
    # print(f"{len(active_users)=}, {len(user_ids_in_dataframe)=}")
    # active_users_without_jobs = [
    #     u for u in active_users if u.uuid not in user_ids_in_dataframe
    # ]
    # num_clusters_used_plot_data[0] = len(active_users_without_jobs)
    # num_clusters_used_plot_data = num_clusters_used_plot_data.sort_index()

    fig1 = px.pie(
        num_clusters_used_plot_data,
        values=num_clusters_used_plot_data.values,
        names=[f"{v} clusters" for v in num_clusters_used_plot_data.index],
        title="Number of unique clusters used per user",
    )
    fig1.show()

    num_users_per_cluster_used_plot_data = (
        df.groupby("cluster_name")["user.email"].nunique().sort_index()
    )
    # num_clusters_used_plot_data["other/none"] = len(active_users_without_jobs)

    fig2 = px.pie(
        num_users_per_cluster_used_plot_data,
        values=num_users_per_cluster_used_plot_data.values,
        names=num_users_per_cluster_used_plot_data.index,
        title="Number of users using each cluster",
    )
    fig2.show()
    return
    df = add_responsible_for_compute_column(df, new_column_name="supervisor.email")
    make_awesome_sunburst_plot(
        df, filter=args, compute_type="rgu", supervisor_key="supervisor.email"
    ).show()
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
    result.to_csv(f"query_{date.today()}.csv")


def temporarily_overwrite_user_email(
    args: FilteringOptions, df: pd.DataFrame
) -> pd.DataFrame:
    all_users = get_users()
    user_id_to_user = {u.uuid: u for u in all_users}
    mila_username_to_user: dict[str, UserData] = {}
    drac_username_to_user: dict[str, UserData] = {}
    for user in all_users:
        if mila_creds := user.associated_accounts.get("mila"):
            if accounts := mila_creds.values_in_range(args.start, args.end):
                mila_username_to_user[accounts[0]] = user
        if drac_creds := user.associated_accounts.get("drac"):
            if accounts := drac_creds.values_in_range(args.start, args.end):
                drac_username_to_user[accounts[0]] = user
    user_emails: list[str | None] = []
    _warned = set()
    for _key, row in tqdm.rich.tqdm(
        df.iterrows(),
        total=len(df),
        disable=not sys.stdout.isatty(),
        leave=False,
        unit="rows",
    ):
        # Userid is in the dataframe, use it.
        user_id = row["user.uuid"]
        if user_id and (user := user_id_to_user.get(user_id)):
            user_emails.append(user.email)
            continue

        username = row["user"]
        cluster = row["cluster_name"]

        # Find the UserData with this username on that cluster.
        if cluster == "mila":
            user = mila_username_to_user.get(username)
        else:
            user = drac_username_to_user.get(username)
        if user is None:
            # if username not in warned:
            #     job_id = row["job_id"]
            #     logger.warning(
            #         f"No data for user with username {username} on cluster {cluster} (job id {job_id})!"
            #     )
            #     warned.add(username)
            user_emails.append(None)
        else:
            user_emails.append(user.email)
    df = df.assign(**{"user.email": user_emails})
    return df


def user_is_active(
    user: UserData, start_date: datetime.datetime, end_date: datetime.datetime
) -> bool:
    """Whether this user has an active account during this period."""
    for credentials in user.associated_accounts.values():
        if credentials.values_in_range(start_date, end_date):
            return True
    return False


def add_responsible_for_compute_column(
    df: pd.DataFrame, new_column_name: str = "supervisor.email"
) -> pd.DataFrame:
    """Add a new column to the dataframe with the person _responsible_ for the job's compute.

    - If a student has a supervisor, the supervisor's email is used.
    - If a student has both a supervisor and co-supervisors, only the supervisor's email is set.
    - If a student doesn't have a supervisor, but has a co-supervisor, one of the co-supervisors' email is used.
    - If a user doesn't have a supervisor or co-supervisor at the time of the job (e.g. a prof or staff), their email is used.
    """
    # This is tricky to compute:
    #     We have to go job by job, and find the supervisor of the associated user
    #     at the time the job started.

    # This is the list of supervisor emails for each job that we will add as a column to the dataframe.
    job_supervisor_email: list[str] = []
    # job_co_supervisor_email: list[str] = []

    uuid_to_user = {u.uuid: u for u in get_users()}
    # Build a map from email string to UserData for the users.
    # email_to_user: dict[str, UserData] = {user.email: user for user in all_users}
    # for user in all_users:
    #     email_to_user[user.email] = user
    #     supervisors = user.supervisor.values_in_range(START_DATE, END_DATE)
    #     for supervisor_uuid in supervisors:
    #         supervisor = uuid_to_user[supervisor_uuid]
    #         email_to_user[supervisor.email] = supervisor

    for index, job_row in tqdm.tqdm(
        df.iterrows(), total=len(df), disable=not sys.stdout.isatty()
    ):
        job_start = job_row["start_time"]
        assert isinstance(job_start, datetime.datetime) and job_start.tzinfo is not None
        user_id = job_row["user.uuid"]
        user = uuid_to_user[user_id]

        # Might get a DateMatchError if the user doesn't have a known supervisor at that time.
        try:
            supervisor_id = user.supervisor.get_value(job_start)
        except DateMatchError:
            # No supervisor at the given date.
            # TODO: Could also use the co-supervisors if there is one?
            try:
                co_supervisor_ids = user.co_supervisors.get_value(job_start)
                # just take one of the co-supervisors if there are multiple.
                # TODO: Could sort somehow, for example based on status at Mila?
                supervisor_id = co_supervisor_ids.pop()
            except DateMatchError, KeyError:
                # "bill" this job to the user itself (i.e. consider them as their own supervisor).
                # This bills the compute done by a prof to themselves.
                supervisor_id = user_id

        supervisor = uuid_to_user[supervisor_id]
        supervisor_email = supervisor.email

        job_supervisor_email.append(supervisor_email)
    df = df.assign(**{new_column_name: job_supervisor_email})
    return df


def make_awesome_sunburst_plot(
    plot_data: pd.DataFrame | pd.Series,
    filter: FilteringOptions,
    compute_type: Literal["cpu", "gpu", "rgu"] = "rgu",
    show_prof_type: bool = True,
    supervisor_key: str = "supervisor.email",
):
    # all_clusters = cluster_type_to_show == "all"
    # if not all_clusters:
    #     sarc_data = sarc_data[sarc_data["cluster_type"] == cluster_type_to_show]
    clusters = plot_data.index.get_level_values("cluster_name").unique().tolist()
    data_start_date = plot_data["compute_start_time"].min().date()
    data_end_date = plot_data["compute_end_time"].max().date()
    multiple_clusters = len(clusters) > 1
    s = "s" if len(clusters) > 1 else ""
    total_compute_values = plot_data[
        ["rgu_equivalent_years", "gpu_equivalent_years", "cpu_equivalent_years"]
    ].sum()
    print(total_compute_values.to_markdown())
    days_in_period = (filter.end - filter.start).days

    # TODO: Add a table in the hover instead of badly formatted floats.
    fig = px.sunburst(
        plot_data.reset_index(),
        path=(["prof_type"] if show_prof_type else [])
        + (
            ["cluster_name", supervisor_key, "user.mila.email"]
            if multiple_clusters
            else [supervisor_key, "user.mila.email"]
        ),
        values=f"{compute_type}_equivalent_years",
        color="gpu_utilization",
        hover_data=[
            "gpu_equivalent_years",
            "rgu_equivalent_years",
            "cpu_equivalent_years",
            "gpu_years",
            "rgu_years",
            "cpu_years",
            "gpu_utilization",
            "cpu_utilization",
            "number_of_jobs",
            "average_job_length_seconds",
            "job_completion_rate",
            "compute_start_time",
            "compute_end_time",
        ],
        title=(
            f"Usage on {clusters[0] if len(clusters) == 1 else clusters} cluster{s} "
            f"between {filter.start.date()} and {filter.end.date()} ({days_in_period} days)"
        ),
        subtitle=(
            "Compute is expressed in GPU/RGU days (1 GPU*day := a GPU used for a full day)<br>"
            "(RGU are a unit or GPU power. A100-80G --> 4.8 RGUs, H100 --> 12.2 RGUs)<br>"
            f"SARC shows {plot_data['number_of_jobs'].sum()} jobs between {data_start_date} and {data_end_date}.<br>"
            # TODO: would be nice to show the maximum possible number of GPUs available full-time during this period.
            f"Number of days in period: {days_in_period}<br>"
            + (
                f"Total compute used: "
                f"{total_compute_values['gpu_equivalent_years']:.0f} GPU years, "
                f"{total_compute_values['rgu_equivalent_years']:.0f} RGU years, "
                f"{total_compute_values['cpu_equivalent_years']:.0f} CPU years, "
                "<br>"
            )
            + (
                f"Equivalent to "
                f"{total_compute_values['gpu_equivalent_years']:.0f} GPUs, "
                f"{total_compute_values['rgu_equivalent_years']:.0f} RGUs, and "
                f"{total_compute_values['cpu_equivalent_years']:.0f} CPUs "
                f"used full time during this period.<br>"
            )
        ),
        # color_continuous_scale="RdBu",
        color_continuous_scale=[
            [0.0, "rgb(255, 0, 0)"],
            [0.5, "rgb(255, 255, 255)"],
            [1.0, "rgb(0, 255, 0)"],
        ],
        color_continuous_midpoint=0.5,
        branchvalues="total",
    )
    # https://community.plotly.com/t/labeling-percentage-on-each-sector-in-sunburst-chart/32129/4
    fig.update_traces(
        textinfo="label+text+value+percent root+percent parent",
        texttemplate=(
            "%{label}<br>"
            + ("%{value:.2s} " + compute_type.upper() + " years<br>")
            + (
                "%{percentParent:.0%} of parent / %{percentRoot:.0%} of total compute"
                if not multiple_clusters
                else (
                    "parent/selection/total:<br>"
                    "%{percentParent:.0%} / %{percentEntry:.0%} / %{percentRoot:.0%}"
                )
            )
        ),
    )
    return fig


if __name__ == "__main__":
    main()
