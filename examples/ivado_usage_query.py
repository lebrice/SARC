import functools
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import plotly.express as px

from examples.waste_monitor.__main__ import _setup_logging
from examples.waste_monitor.sarc_patches import (
    FilteringOptions,
    get_clean_sarc_data,
)
from sarc.client.users.api import User, get_users
from sarc.config import MTL

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

    _setup_logging(verbose=1)
    # 1er avril 2025 au 31 juillet 2025
    filter = FilteringOptions(start=datetime(2025, 4, 1), end=datetime(2025, 7, 31))
    users_in_period = get_active_mila_user_records_during_period(
        start=filter.start, end=filter.end
    )
    all_profs_emails = set(
        supervisor
        for key in ["supervisor", "co_supervisor"]
        for user in users_in_period
        if (supervisor := user.mila_ldap.get(key))
    )
    professors = [
        user
        for user in users_in_period
        if user.mila is not None and user.mila.email in all_profs_emails
    ]
    sarc_data = get_clean_sarc_data(filter)
    assert isinstance(sarc_data, pd.DataFrame)
    # For jobs where users don't have a supervisor, and the user itself is not a prof,
    # we set the "supervisor" to "Staff/Industry/Other"
    prof_emails = {prof.mila.email for prof in professors}

    is_student = sarc_data[supervisor_key].notna()
    is_prof = sarc_data["user.mila.email"].isin(prof_emails)
    is_staff = sarc_data[supervisor_key].isna() & ~is_prof
    # Need to be in exactly one of these three categories.
    assert (_t := (is_student ^ is_prof ^ is_staff)).all(), (
        sarc_data[_t]["user.mila.email"].unique().tolist()
    )

    sarc_data = sarc_data.assign(user_type="")
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
    # make_awesome_sunburst_plot(sarc_data, cluster_type_to_show="all", filter=filter)
    make_awesome_sunburst_plot(sarc_data, cluster_type_to_show="mila", filter=filter)
    make_awesome_sunburst_plot(sarc_data, cluster_type_to_show="drac", filter=filter)
    make_awesome_sunburst_plot(sarc_data, cluster_type_to_show="paice", filter=filter)


def make_awesome_sunburst_plot(
    sarc_data: pd.DataFrame,
    cluster_type_to_show: Literal["mila", "drac", "paice", "all"],
    filter: FilteringOptions,
):
    sarc_data = sarc_data[sarc_data["cluster_type"] == cluster_type_to_show]
    clusters = sarc_data["cluster_name"].unique().tolist()
    data_start_date = sarc_data["start_time"].min().date()
    data_end_date = sarc_data["end_time"].max().date()
    multiple_clusters = len(clusters) > 1
    plot_data = sarc_data.groupby(
        ["cluster_name", "prof_type", supervisor_key, "user.mila.email"]
    ).aggregate(
        dict(
            **{c: "sum" for c in sarc_data.columns if c.endswith("_cost")},
            **{
                c: "mean"
                for c in sarc_data.columns
                if c.endswith("_mean") or "utilization" in c
            },
            job_id="count",
        )
    )
    plot_data = plot_data.assign(
        **{
            c: plot_data[c].div(pd.Timedelta(days=1))
            for c in plot_data.columns
            if c.endswith("_cost")
        }
    )
    s = "s" if len(clusters) > 1 else ""
    total_compute_values = (
        sarc_data[["rgu_equivalent_cost", "gpu_equivalent_cost", "cpu_equivalent_cost"]]
        .div(pd.Timedelta(days=1))
        .sum()
    )
    days_in_period = (filter.end - filter.start).days

    # TODO: Add a table in the hover instead of badly formatted floats.
    fig = px.sunburst(
        plot_data.reset_index(),
        path=(
            [
                # "cluster_type",
                "prof_type",
                "cluster_name",
                supervisor_key,
                "user.mila.email",
            ]
            if multiple_clusters
            else ["prof_type", supervisor_key, "user.mila.email"]
        ),
        values="rgu_equivalent_cost",
        color="gpu_utilization",
        hover_data=[
            "gpu_equivalent_cost",
            "rgu_equivalent_cost",
            "cpu_equivalent_cost",
            "gpu_utilization",
            "cpu_utilization",
        ],
        title=(
            f"Usage on {clusters[0] if len(clusters) == 1 else clusters} cluster{s} "
            f"between {filter.start.date()} and {filter.end.date()} ({days_in_period} days)"
        ),
        subtitle=(
            "Compute is expressed in GPU/RGU days (1 GPU*day := a GPU used for a full day)<br>"
            "(RGU are a unit or GPU power. A100-80G --> 4.8 RGUs, H100 --> 12.2 RGUs)<br>"
            f"SARC shows {sarc_data['job_id'].nunique()} jobs between {data_start_date} and {data_end_date}.<br>"
            # TODO: would be nice to show the maximum possible number of GPUs available full-time during this period.
            f"Number of days in period: {days_in_period}<br>"
            + (
                f"Total compute used: "
                f"{total_compute_values['gpu_equivalent_cost']:.0f} GPU days, "
                f"{total_compute_values['rgu_equivalent_cost']:.0f} RGU days, "
                f"{total_compute_values['cpu_equivalent_cost']:.0f} CPU days, "
                "<br>"
            )
            + (
                f"Equivalent to "
                f"{total_compute_values['gpu_equivalent_cost'] / days_in_period:.0f} GPUs, "
                f"{total_compute_values['rgu_equivalent_cost'] / days_in_period:.0f} RGUs, and "
                f"{total_compute_values['cpu_equivalent_cost'] / days_in_period:.0f} CPUs "
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
        secondary_y=False,
        textinfo="label+text+value+percent root+percent parent",
        # texttemplate="%{y:.1f} days",
        texttemplate=(
            "%{label}<br>"
            "%{value:.2s} RGU days<br>"
            + (
                "%{percentParent:.0%} of parent, %{percentRoot:.0%} of total compute<br>"
                if not multiple_clusters
                else (
                    "%{percentParent:.0%} of parent / %{percentEntry:.0%} of selection<br>"
                    "%{percentRoot:.0%} of total compute"
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
