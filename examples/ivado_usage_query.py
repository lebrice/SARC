import functools
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px

from examples.waste_monitor.__main__ import _setup_logging
from examples.waste_monitor.sarc_patches import (
    FilteringOptions,
    get_clean_sarc_data,
)
from sarc.client.users.api import User, get_users
from sarc.config import MTL

logger = logging.getLogger(__name__)

# Need to have this secret file with the emails of the core profs.
CORE_PROF_EMAILS = Path("core_profs.txt").read_text().splitlines()
DRAC_CLUSTERS = ["narval", "beluga", "cedar", "graham", "rorqual", "fir", "nibi"]
PAICE_CLUSTERS = ["tamia", "killarney", "vulcan"]


@functools.cache
# @cache_results_to_file
def get_active_mila_user_records_during_period(
    start: datetime, end: datetime
) -> tuple[User, ...]:
    # NOTE: Might be multiple records for the same student if they changed status or something during the period!
    students = get_users(latest=False)
    students_in_period: list[User] = []
    for student in students:
        if student.mila is None or student.mila.email is None:
            continue
        r_start = student.record_start
        r_end = student.record_end
        if not (r_start is None or r_start.astimezone(MTL) <= end.astimezone(MTL)):
            continue  # record starts after `end`, ignore.
        if not (r_end is None or r_end.astimezone(MTL) >= start.astimezone(MTL)):
            continue  # record ends before `start`, ignore.
        students_in_period.append(student)
    return tuple(students_in_period)


# @cache_results_to_file
def get_group_students(
    prof_email: str,
    start: datetime,
    end: datetime,
) -> list[User]:
    """Get list of student emails supervised by a professor."""
    student_records = get_active_mila_user_records_during_period(start=start, end=end)

    return sorted(
        [
            s
            for s in student_records
            if any(
                s.mila_ldap.get(k) == prof_email
                for k in ["supervisor", "co_supervisor"]
            )
        ],
        key=lambda v: v.name,
    )


def is_prof(user: User, all_users: Iterable[User]) -> bool:
    """A user is considered a professor if they don't have a supervisor (and no co-supervisor) and at least one student."""
    if any(user.mila_ldap.get(k) for k in ["supervisor", "co_supervisor"]):
        return False
    user_email = user.mila.email
    return any(
        other_user.mila_ldap.get("supervisor") == user_email
        or other_user.mila_ldap.get("co_supervisor") == user_email
        for other_user in all_users
    )


def is_core_prof(user: User, all_users: Iterable[User]) -> bool:
    """A user is considered a professor if they don't have a supervisor (and no co-supervisor) and at least one student."""
    return is_prof(user, all_users) and user.mila.email in CORE_PROF_EMAILS


supervisor_key = "user.mila_ldap.supervisor"


def main():
    """
    Simplifying for now: attributing all the compute usage to only the supervisor instead of:
    a) splitting between supervisor and co-supervisor, or
    b) duplicating the values by counting it for both the supervisor or co-supervisor.
    """

    _setup_logging(verbose=2)
    logger.setLevel(logging.DEBUG)
    # 1er avril 2025 au 31 juillet 2025
    filter = FilteringOptions(
        start=datetime(2025, 4, 1, tzinfo=MTL), end=datetime(2025, 7, 31, tzinfo=MTL)
    )

    sarc_data = get_clean_sarc_data(filter)
    assert sarc_data.query("job_id == 16").empty
    # Chop up into monthly frames to keep the portion of jobs
    # that started before the start or ended after the end.
    sarc_data = sarc_data.assign(
        start_time=sarc_data["start_time"].dt.tz_convert(MTL),
        end_time=sarc_data["end_time"].dt.tz_convert(MTL),
    )

    # TODO: Jobs that started before the start, or ended before the end are also included in the output.
    # Ideally we'd like to split them into sections to only consider their usage during the period.
    # clip_time here (instead of as an argument to `load_job_series`).
    jobs_that_will_be_clipped = sarc_data.query(
        "((start_time < @filter.start) & (end_time > @filter.start)) | "
        "((start_time < @filter.end) & (end_time > @filter.end))"
    )
    assert jobs_that_will_be_clipped.empty

    _users_in_period = get_active_mila_user_records_during_period(
        start=filter.start, end=filter.end
    )
    _profs_emails = set(
        supervisor
        for key in ["supervisor", "co_supervisor"]
        for user in _users_in_period
        if (supervisor := user.mila_ldap.get(key))
    )
    _professors = [
        user
        for user in _users_in_period
        if user.mila is not None and user.mila.email in _profs_emails
    ]
    prof_emails = {prof.mila.email for prof in _professors}

    is_prof = sarc_data["user.mila.email"].isin(prof_emails)
    is_student = sarc_data[supervisor_key].notna()
    is_staff = sarc_data[supervisor_key].isna() & ~is_prof
    # Need to be in exactly one of these three categories.
    assert (_t := (is_student ^ is_prof ^ is_staff)).all(), (
        sarc_data[_t]["user.mila.email"].unique().tolist()
    )

    sarc_data = sarc_data.assign(user_type="other")
    sarc_data.loc[is_student, "user_type"] = "student"
    sarc_data.loc[is_prof, "user_type"] = "prof"
    sarc_data.loc[is_staff, "user_type"] = "other"

    # Mark profs as their own supervisor to make it easier to count their compute towards their own group.
    sarc_data.loc[is_prof, supervisor_key] = sarc_data.loc[is_prof, "user.mila.email"]
    sarc_data.loc[is_staff, supervisor_key] = "No supervisor"

    is_core = (is_student | is_prof) & sarc_data["user.mila_ldap.supervisor"].isin(
        CORE_PROF_EMAILS
    )
    sarc_data = sarc_data.assign(prof_type="")
    sarc_data.loc[is_core, "prof_type"] = "core prof"
    sarc_data.loc[~is_staff & ~is_core, "prof_type"] = "non-core prof"
    sarc_data.loc[is_staff, "prof_type"] = "Staff/Industry/Other"

    sarc_data = sarc_data.assign(
        cluster_type=sarc_data["cluster_name"].map(
            {
                "mila": "mila",
                **{c: "drac" for c in DRAC_CLUSTERS},
                **{c: "paice" for c in PAICE_CLUSTERS},
            }
        )
    )
    grouped_data = sarc_data.groupby(
        ["cluster_type", "cluster_name", "prof_type", supervisor_key, "user.mila.email"]
    )
    plot_data = grouped_data.aggregate(
        dict(
            **{
                c: "mean"
                for c in sarc_data.columns
                if c.endswith("_mean") or "utilization" in c
            },
        ),
    )
    # Add the total [cpu/gpu/rgu] cost in days.
    plot_data = plot_data.assign(
        # Add columns with the total compute in days.
        **{
            f"{c.removesuffix('_cost')}_days": (
                grouped_data[c].sum().div(timedelta(days=1).total_seconds())
            )
            for c in sarc_data.columns
            if c.endswith("_cost")
        },
        # Add other nice metrics.
        number_of_jobs=grouped_data["job_id"].nunique(),
        average_job_length_seconds=grouped_data["elapsed_time"].mean(),
        job_completion_rate=grouped_data["job_state"].apply(
            lambda s: (s == "COMPLETED").sum() / s.shape[0] if s.shape[0] > 0 else 0
        ),  # type: ignore
        # When they started using compute.
        compute_start_time=grouped_data["start_time"].min(),
        # When they stopped using compute.
        compute_end_time=grouped_data["end_time"].max(),
    )

    # plot_data = plot_data.sort_values("rgu_equivalent_cost_days", ascending=False)
    plot_data.to_csv(f"compute_usage_{filter.start.date()}_{filter.end.date()}.csv")

    make_awesome_sunburst_plot(
        plot_data.xs("mila", level="cluster_type"),
        # cluster_type_to_show="mila",
        filter=filter,
        show_rgu=True,
    )
    make_awesome_sunburst_plot(
        plot_data.xs("drac", level="cluster_type"),
        # cluster_type_to_show="drac",
        filter=filter,
        show_rgu=True,
    )
    # make_awesome_sunburst_plot(
    #     sarc_data, cluster_type_to_show="drac", filter=filter, show_rgu=False
    # )
    # make_awesome_sunburst_plot(
    #     sarc_data, cluster_type_to_show="paice", filter=filter, show_rgu=False
    # )
    # make_awesome_sunburst_plot(
    #     sarc_data, cluster_type_to_show="all", filter=filter, show_rgu=False
    # )


def make_awesome_sunburst_plot(
    plot_data: pd.DataFrame | pd.Series,
    filter: FilteringOptions,
    show_rgu: bool = False,
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
        [
            "rgu_equivalent_days",
            "gpu_equivalent_days",
            "cpu_equivalent_days",
        ]
    ].sum()
    print(total_compute_values.to_markdown())
    days_in_period = (filter.end - filter.start).days

    # TODO: Add a table in the hover instead of badly formatted floats.
    fig = px.sunburst(
        plot_data.reset_index(),
        path=(
            ["prof_type", "cluster_name", supervisor_key, "user.mila.email"]
            if multiple_clusters
            else ["prof_type", supervisor_key, "user.mila.email"]
        ),
        values="rgu_equivalent_days" if show_rgu else "gpu_equivalent_days",
        color="gpu_utilization",
        hover_data=[
            "gpu_equivalent_days",
            "rgu_equivalent_days",
            "cpu_equivalent_days",
            "gpu_days",
            "rgu_days",
            "cpu_days",
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
                f"{total_compute_values['gpu_equivalent_days']:.0f} GPU days, "
                f"{total_compute_values['rgu_equivalent_days']:.0f} RGU days, "
                f"{total_compute_values['cpu_equivalent_days']:.0f} CPU days, "
                "<br>"
            )
            + (
                f"Equivalent to "
                f"{total_compute_values['gpu_equivalent_days'] / days_in_period:.0f} GPUs, "
                f"{total_compute_values['rgu_equivalent_days'] / days_in_period:.0f} RGUs, and "
                f"{total_compute_values['cpu_equivalent_days'] / days_in_period:.0f} CPUs "
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
            + ("%{value:.2s} " + ("RGU" if show_rgu else "GPU") + " days<br>")
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
    fig.show("browser")

    # mila_data = sarc_data.query("cluster_type=='mila'")
    # drac_data = sarc_data.query("cluster_type=='drac'")
    # paice_data = sarc_data.query("cluster_type=='paice'")

    # _, fig = usage_plot_mila_or_drac(mila_data)
    # fig.show("browser")


def _usage_plot_mila_or_drac(sarc_data: pd.DataFrame):
    """Make a pie chart of the resource usage for different categories of users/groups.

    Groups the data for profs that are not "core" profs in a single entry.
    Same for all users that aren't profs or students, we assume they are staff/industry/other
    and place them in a single entry.
    """

    usage_data = sarc_data.groupby("user.mila_ldap.supervisor").aggregate(
        dict(
            **{c: "sum" for c in sarc_data.columns if c.endswith("_cost")},
            **{c: "mean" for c in sarc_data.columns if c.endswith("_mean")},
        )
    )

    # Note: Seems like renaming np.nan in index to something else doesn't work.
    # This is why we opt for setting the supervisor to "Staff/Industry/Other" above instead of keeping NaNs.
    # usage_data = usage_data.rename(index={np.nan: "Staff/Industry/Other"})
    is_non_core_prof = ~(
        usage_data.index.isin(CORE_PROF_EMAILS)
        | (usage_data.index == "Staff/Industry/Other")
    )
    usage_data = pd.concat(
        [
            pd.DataFrame(
                [usage_data[is_non_core_prof].sum(axis=0)],
                index=["Non-core profs"],
                columns=usage_data.columns,
            ),
            usage_data[~is_non_core_prof],
        ]
    )
    s = "s" if len(usage_data) > 1 else ""
    print(usage_data.to_markdown())
    clusters = sarc_data["cluster_name"].unique().tolist()
    start_date = sarc_data["start_time"].min().date()
    end_date = sarc_data["end_time"].max().date()
    fig = px.pie(
        usage_data,
        values="rgu_equivalent_cost",
        names=usage_data.index.values,
        title=(
            f"Usage on {clusters[0] if len(clusters) == 1 else clusters} cluster{s} between {start_date} and {end_date}"
        ),
    )

    return usage_data, fig
    # sarc_data =


if __name__ == "__main__":
    main()
