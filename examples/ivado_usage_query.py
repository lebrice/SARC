import functools
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px

from examples.waste_monitor.__main__ import _setup_logging
from examples.waste_monitor.sarc_patches import (
    FilteringOptions,
    get_clean_sarc_data,
)
from sarc.client.users.api import User, get_users
from sarc.config import MTL

CORE_PROF_EMAILS = Path("core_profs.txt").read_text().splitlines()


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


def main():
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

    # Simplifying for now: attributing all the compute usage to only the supervisor instead of:
    # a) splitting between supervisor and co-supervisor, or
    # b) duplicating the values by counting it for both the supervisor or co-supervisor.
    usage_data, fig = usage_plot_mila_or_drac(
        filter=filter,
        professors=professors,
        sarc_data=sarc_data,
        mila_only=True,
    )
    print(usage_data.to_markdown())
    fig.show(renderer="browser")

    usage_data, fig = usage_plot_mila_or_drac(
        filter=filter,
        professors=professors,
        sarc_data=sarc_data,
        mila_only=False,
        drac_only=True,
    )
    print(usage_data.to_markdown())
    fig.show(renderer="browser")

    usage_data, fig = usage_plot_mila_or_drac(
        filter=filter,
        professors=professors,
        sarc_data=sarc_data,
        mila_only=False,
        drac_only=False,
    )
    print(usage_data.to_markdown())
    fig.show(renderer="browser")

    fig.write_image("usage_per_prof_on_mila_cluster.png")
    # plt.show(block=True)


def usage_plot_mila_or_drac(
    filter: FilteringOptions,
    professors: Sequence[User],
    sarc_data: pd.DataFrame,
    mila_only: bool = True,
    drac_only: bool = False,
):
    sarc_data = sarc_data.copy()
    if mila_only:
        sarc_data = sarc_data.query("cluster_name=='mila'")
    elif drac_only:
        sarc_data = sarc_data.query("cluster_name!='mila'")
        sarc_data = sarc_data[
            ~sarc_data["cluster_name"].isin(["tamia", "killarney", "vulcan"])
        ]
    else:
        # PAICE only.
        sarc_data = sarc_data[
            sarc_data["cluster_name"].isin(["tamia", "killarney", "vulcan"])
        ]
    # For jobs where users don't have a supervisor, and the user itself is not a prof, we set the "supervisor" to "Staff/Industry/Other
    prof_emails = {prof.mila.email for prof in professors}

    is_prof = sarc_data["user.mila.email"].isin(prof_emails)
    is_staff = sarc_data["user.mila_ldap.supervisor"].isna() & ~is_prof

    # TODO: Need to count the compute of profs towards themselves, even though they don't have a supervisor field!
    sarc_data.loc[is_prof, "user.mila_ldap.supervisor"] = sarc_data.loc[
        is_prof, "user.mila.email"
    ]
    sarc_data.loc[is_staff, "user.mila_ldap.supervisor"] = "Staff/Industry/Other"

    usage_data = (
        sarc_data.groupby("user.mila_ldap.supervisor")[
            ["gpu_cost", "gpu_equivalent_cost", "rgu_cost", "rgu_equivalent_cost"]
        ]
        .sum()
        .div(pd.Timedelta(days=1))  # convert from datetime to float (gpu/rgu days).
    )

    # TODO: renaming nan in index to something else doesn't seem to work.
    # usage_data = usage_data.rename(index={np.nan: "Staff/Industry/Other"})
    is_non_core_prof = ~(
        usage_data.index.isin(CORE_PROF_EMAILS)
        | (usage_data.index == "Staff/Industry/Other")
    )
    usage_data = pd.concat(
        [
            usage_data[~is_non_core_prof],
            pd.DataFrame(
                usage_data[is_non_core_prof].sum(axis=0),
                index=["Non-core profs"],
                columns=usage_data.columns,
            ),
        ]
    )
    print(usage_data.to_markdown())
    fig = px.pie(
        usage_data,
        values="rgu_equivalent_cost",
        names=usage_data.index.values,
        title=f"Usage on {'Mila' if mila_only else 'DRAC' if drac_only else 'PAICE'} cluster between {filter.start.date()} and {filter.end.date()}",
    )

    return usage_data, fig
    # sarc_data =


if __name__ == "__main__":
    main()
