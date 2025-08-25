import functools
from datetime import datetime

import pandas as pd
import plotly.express as px

from examples.waste_monitor.__main__ import _setup_logging
from examples.waste_monitor.sarc_patches import (
    FilteringOptions,
    get_clean_sarc_data,
)
from sarc.client.users.api import User, get_users
from sarc.config import MTL


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
    sarc_data = get_clean_sarc_data(filter)

    # Simplifying for now: attributing all the compute usage to only the supervisor instead of:
    # a) splitting between supervisor and co-supervisor, or
    # b) duplicating the values by counting it for both the supervisor or co-supervisor.

    usage_per_prof_on_mila_cluster = (
        sarc_data.query("cluster_name=='mila'")
        .groupby("user.mila_ldap.supervisor")[
            ["gpu_cost", "gpu_equivalent_cost", "rgu_cost", "rgu_equivalent_cost"]
        ]
        .sum()
        .div(pd.Timedelta(days=1))  # convert from datetime to float (gpu/rgu days).
        # .plot.pie(
        #     # y="rgu_equivalent_cost",
        #     subplots=True,
        #     figsize=(12, 8),
        #     legend=False,
        #     title="Usage on mila cluster",
        # )
    )
    fig = px.pie(
        usage_per_prof_on_mila_cluster,
        values="rgu_equivalent_cost",
        names=usage_per_prof_on_mila_cluster.index.values,
        title=f"Usage on mila cluster between {filter.start.date()} and {filter.end.date()}",
    )
    fig.show(renderer="browser")
    fig.write_image("usage_per_prof_on_mila_cluster.png")
    # plt.show(block=True)
    print(usage_per_prof_on_mila_cluster.to_markdown())
    # sarc_data =


if __name__ == "__main__":
    main()
