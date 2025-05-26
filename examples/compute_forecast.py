import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Generic, Mapping, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import rich.layout
import rich.logging
import rich.panel
import rich.pretty
import rich.prompt
import rich.text
import simple_parsing
import yaml
from typing_extensions import Self

os.environ["SARC_CONFIG"] = "config/sarc-client.yaml"

from sarc.client.job import JobStatistics
from sarc.client.series import (
    compute_cost_and_waste,
    load_job_series,
)
from sarc.client.users.api import User, get_users
from sarc.config import MTL, ClusterConfig
from sarc.jobs.series import (
    update_cluster_job_series_rgu,
)

logger = logging.getLogger(__name__)
pd.options.display.max_colwidth = 300
pd.options.display.max_rows = 1000
pd.options.display.float_format = lambda x: f"{x:.3f}"

ALL_CLUSTERS = ["mila", "narval", "beluga", "cedar", "graham"]

seconds_in_a_year = timedelta(days=365.242374).total_seconds()

_gpu_name_mapping = {
    "gpu:tesla_v100-sxm2-16gb:4": "v100-16gb",
    "p100": "p100-12gb",
    "gpu:p100:4": "p100-12gb",
    "gpu:p100:2": "p100-12gb",
    "gpu:p100l:4": "p100-16gb",
    "v100": "v100-16gb",
    "gpu:v100:6": "v100-16gb",
    "gpu:v100:8": "v100-16gb",
    "gpu:v100l:4": "v100-32gb",
    "gpu:t4:4": "t4-16gb",
    "4g.20gb": "a100-40gb-4g.20gb",
    "3g.20gb": "a100-40gb-3g.20gb",
    "a100_4g.20gb": "a100-40gb-4g.20gb",
    "gpu:a100_4g.20gb:4": "a100-40gb-4g.20gb",
    "a100_3g.20gb": "a100-40gb-3g.20gb",
    "gpu:a100_3g.20gb:4": "a100-40gb-3g.20gb",
    "a100": "a100-40gb",
    "gpu:a100:4": "a100-40gb",
    "gpu:a100:8": "a100-40gb",
    "gpu:a100_4g.20gb:4,gpu:a100_3g.20gb:4": "a100-mixup",
    "gpu:a100l:4": "a100-80gb",
    "gpu:a100l:8": "a100-80gb",
    "gpu:a6000:8": "a6000",
    "gpu:rtx8000:8": "rtx8000-48gb",
    "gpu:h100:8": "h100-80gb",
    "NVIDIA A100-SXM4-40GB": "a100-40gb",
    "NVIDIA A100-80GB PCIe": "a100-80gb",
    "NVIDIA A100 80GB PCIe": "a100-80gb",
    "NVIDIA A100-SXM4-80GB": "a100-80gb",
    "NVIDIA H100 80GB HBM3": "h100-80gb",
    "NVIDIA L40S": "l40s",
    "NVIDIA RTX A6000": "a6000",
    "gpu:l40s:4": "l40s",
    "a100_2g.10gb": "a100-40gb-2g.10gb",
    "2g.10gb": "a100-40gb-2g.10gb",
    "2g.20gb": "a100-80gb-2g.20gb",
    "3g.40gb": "a100-80gb-3g.40gb",
    "4g.40gb": "a100-80gb-4g.40gb",
    "Tesla V100-SXM2-16GB": "v100-16gb",
    "Tesla V100-SXM2-32GB": "v100-32gb",
    "Tesla V100-SXM2-32GB-LS": "v100-32gb",
    "NVIDIA V100-SXM2-32GB-LS": "v100-32gb",
    "Quadro RTX 8000": "rtx8000-48gb",  # Dummy
    "gpu:a5000:4": "a5000-24gb",
    # NOTE: Added for narval. Might be fixed with `get_node_to_gpu`, unclear.
    "a100_1g.5gb": "a100-weird",
    "1g.5gb": "a100-weird",
}

_gpu_ram = {
    "p100-12gb": 12,
    "p100-16gb": 16,
    "t4-16gb": 16,
    "v100-16gb": 16,
    "v100-32gb": 32,
    "a100-40gb": 40,
    "a100-mixup": 40,
    "a100-40gb-2g.10gb": 10,
    "a100-40gb-4g.20gb": 20,
    "a100-40gb-3g.20gb": 20,
    "rtx8000-48gb": 48,
    "a5000-24gb": 24,
    "a6000": 48,  # Dummy
    "a100-80gb-4g.40gb": 40,
    "a100-80gb-3g.40gb": 40,
    "a100-80gb-2g.20gb": 20,
    "a100-80gb": 80,
    "h100-80gb": 80,
    "l40s": 48,
    # NOTE: Added for narval. Might be fixed with `get_node_to_gpu`, unclear.
    "a100-weird": 5,
}

_RGUS = {
    "p100-12gb": 1,
    "p100-16gb": 1.1,
    "t4-16gb": 1.3,
    "v100-16gb": 2.2,
    "v100-32gb": 2.6,
    "a100-40gb": 4,
    "a100-mixup": 4,
    "a100-40gb-4g.20gb": 2.3,
    "a100-40gb-3g.20gb": 2,
    "a100-40gb-2g.10gb": 1,
    "rtx8000-48gb": 2.81,  # dummy
    "a5000-24gb": 2.6,  # dummy
    "a100-80gb": 4.8,
    "a100-80gb-2g.20gb": 4.8 * 2 / 7,
    "a100-80gb-3g.40gb": 4.8 * 3 / 7,
    "a100-80gb-4g.40gb": 4.8 * 4 / 7,
    "a6000": 4.93,
    "h100-80gb": 12.2,
    "l40s": 10.4,
}


def _midnight(dt: datetime) -> datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _get_survey_answers_csv(google_sheets_url: str) -> pd.DataFrame:
    """IDEA: Fetches the CSV data from the given Google Sheets URL.

    Pretty unnecessary, just a nice-to-have. The difficulty is that the URL is only accessible
    after logging in with Google authentication.
    """
    raise NotImplementedError("TODO")


@functools.total_ordering
@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class Options:
    """Configuration options for this script."""

    start: datetime = simple_parsing.field(
        default=(_midnight(datetime.now(tz=MTL)) - timedelta(days=30)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ Start date. """

    end: datetime = simple_parsing.field(
        default=_midnight(datetime.now(tz=MTL)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ End date. """

    user: list[str] = dataclasses.field(default_factory=list)
    """ Which user(s) to query information for. Leave blank to get a global compute profile."""

    users_file: Path | None = dataclasses.field(default=None, repr=False)

    clusters: list[str] = dataclasses.field(default_factory=list)
    """ Which clusters to query information for. Leave blank to get data from all clusters."""

    cache_dir: Path = dataclasses.field(
        default=Path(os.environ.get("SCRATCH", tempfile.gettempdir())),
        hash=False,
        repr=False,
    )
    """ Directory where temporary files will be stored."""
    verbose: int = simple_parsing.field(
        alias=["-v", "--verbose"], action="count", default=0, hash=False, repr=False
    )

    def get_users(self, assume_mila_email: bool = False) -> list[str]:
        if self.users_file:
            assert not self.user, "can't use both user_file and users!"
            users = sorted(set(self.users_file.read_text().splitlines(keepends=False)))
        else:
            users = self.user
        user_emails = []
        for user in users:
            if "@" in user:
                user_emails.append(user.strip())
            elif assume_mila_email:
                user_emails.append(user.strip() + "@mila.quebec")
            else:
                raise ValueError(
                    f"User '{user}' does not contain an email address. "
                    "Please provide a valid email address or set `assume_mila_email=True`."
                )
        return user_emails

    def unique_path(self, label: str = "", extension: str = ".pkl") -> Path:
        user_emails = self.get_users()
        user_portion = (
            hashlib.md5("+".join(sorted(user_emails)).encode()).hexdigest()
            if user_emails is not None and len(user_emails)
            else "all"
        )
        # cluster_portion = "-".join(self.clusters) if self.clusters else "all"
        start_portion = (
            self.start.strftime("%Y-%m-%d")
            if self.start == _midnight(self.start)
            else str(self.start).replace(" ", "_")
        )
        end_portion = (
            self.end.strftime("%Y-%m-%d")
            if self.end == _midnight(self.end)
            else str(self.end).replace(" ", "_")
        )
        return (
            self.cache_dir
            / f"compute_profile-{user_portion}-{start_portion}-{end_portion}-{label}"
        ).with_suffix(extension)

    def __eq__(self, other: object) -> bool:
        """Returns whether this filter is equal to the other."""
        if not isinstance(other, Options):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and set(self.user) == set(other.user)
            and set(self.clusters) == set(other.clusters)
        )

    def __lt__(self, other: Self) -> bool:
        """Returns whether this filter is strictly more restrictive than the other."""
        if not isinstance(other, Options):
            return NotImplemented
        self_users = self.get_users(assume_mila_email=True)
        other_users = other.get_users(assume_mila_email=True)
        if self_users == [] and other_users != []:
            # This filter matches all users while the other one doesn't.
            return False
        if self_users != [] and other_users == []:
            # Pretend that the other filter matches one more user than this one,
            # to make the comparison below cleaner.
            other_users = self_users + ["some_random_user_that_isnt_in_self"]
        self_clusters = sorted(set(self.clusters))
        other_clusters = sorted(set(other.clusters))
        if self_clusters == [] and other_clusters != []:
            # This filter matches all clusters while the other one doesn't.
            return False
        if self_clusters != [] and other_clusters == []:
            # Pretend that the other filter matches one more cluster than this one,
            # to make the comparison below cleaner.
            other_clusters = self_clusters + ["some_random_cluster_that_isnt_in_self"]

        return (
            (other.start < self.start)
            and (self.end < other.end)
            and (set(self_users) < set(other_users))
            and (set(self.clusters) < set(other.clusters))
        )


def _setup_logging(verbose: int):
    logging.basicConfig(
        handlers=[rich.logging.RichHandler()],
        format="%(message)s",
        level=logging.ERROR,
    )
    logging.getLogger("sarc").setLevel(logging.WARNING)

    if verbose == 0:
        logger.setLevel("WARNING")
    elif verbose == 1:
        logger.setLevel("INFO")
    else:
        logger.setLevel("DEBUG")


T = TypeVar("T", float, timedelta)


@dataclasses.dataclass(frozen=True)
class Estimate(Generic[T]):
    min: T
    max: T

    @property
    def mean(self) -> T:
        return (self.min + self.max) / 2

    def __repr__(self) -> str:
        return f"{self.mean} ± {(self.max - self.min) / 2}  [{self.min}, {self.max}]"

    def __add__(self, other: Self) -> "Estimate[T]":
        return Estimate(self.min + other.min, self.max + other.max)


def main():
    options = simple_parsing.parse(
        Options, default=Options(user=["blake.richards@mila.quebec"], verbose=1)
    )
    _setup_logging(verbose=options.verbose)
    prof = options.user[0]

    students = get_group_students(prof)
    print(f"Students supervised by {prof}: {[s.name for s in students]}")

    group_usage_per_student = get_group_usage_by_student(prof)

    print(f"Usage by students supervised by {prof}:")
    k = 5
    for year in sorted(group_usage_per_student["year"].unique()):
        mask = group_usage_per_student["year"] == year
        print(f"{k} students that used the most compute in {prof}'s group in {year}:")
        print(group_usage_per_student[mask].nlargest(5, "gpu_years"))

    group_usage_old = get_group_usage(prof)

    # TODO: Compare the "old" vs this potential "new" way to get the group usage (by summing across students).
    group_usage_new = (
        group_usage_per_student.groupby("year")
        # TODO: Shouldn't sum all metrics! Only the gpu_years and cpu_years.
        .sum()
        .assign(students=group_usage_per_student.groupby("year")["user"].nunique())
        .reset_index()  # add back the 'year' as a column
    )
    _print_like_form_shows(group_usage_old)
    _print_like_form_shows(group_usage_new)
    return

    usage_projections = get_group_usage_projections(prof)
    _print_like_form_shows(pd.concat([group_usage_old, usage_projections]))

    # print(usage.to_markdown())


def get_group_students(prof_email: str) -> list[User]:
    """Get list of student emails supervised by a professor.
    For now, returns random fake emails for testing.
    """
    prof_students = [
        user
        for user in get_users()
        if user.mila_ldap.get("supervisor") == prof_email
        or user.mila_ldap.get("co_supervisor") == prof_email
    ]
    return sorted(prof_students, key=lambda v: v.name)


def get_group_usage(
    prof_email: str,
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2025, 1, 1),
) -> pd.DataFrame:
    """Returns the total compute usage for a prof's group in the given period."""

    students = get_group_students(prof_email)
    logger.info(f"{prof_email} has apparently {len(students)} students.")

    options = Options(
        start=start.astimezone(MTL),
        end=end.astimezone(MTL),
        user=[s.mila.email for s in students],
    )
    sarc_data = _get_cleaned_df(options)
    usage_stats = _get_stats(sarc_data, options, frame_size="YS")
    gpu_job_stats = usage_stats[usage_stats["requested.gres_gpu"] > 0]
    cpu_job_stats = usage_stats[usage_stats["requested.gres_gpu"] == 0]

    unknown_gpu = gpu_job_stats["allocated.gpu_type"] == "unknown"
    assert not any(unknown_gpu), gpu_job_stats[unknown_gpu][
        ["job_id", "cluster_name", "allocated.gres_gpu", "nodes"]
    ]

    # Note: good to know: allocated.gres_gpu takes into account the "effective" # of gpus used.
    # For example, if you use all the CPUs on a node, you get billed for all the gpus.

    assert not gpu_job_stats["allocated.gpu_type"].isna().any(), gpu_job_stats[
        "allocated.gpu_type"
    ].unique()

    # Create two new columns for the CPU and GPU memory usage in gigabytes.
    gpu_job_stats = gpu_job_stats.assign(
        gpu_mem_gb=(
            gpu_job_stats["gpu_memory"]
            * gpu_job_stats["allocated.gpu_type"].map(_gpu_ram)
            # note: don't multiply by # of gpus.
            # * gpu_job_stats["allocated.gres_gpu"]
        ),
        cpu_mem_gb=(
            # system_memory is a percentage, allocated.mem is in MB (I think).
            gpu_job_stats["system_memory"] * (gpu_job_stats["allocated.mem"] // 1024)
        ),
    )
    cpu_job_stats = cpu_job_stats.assign(
        cpu_mem_gb=(
            cpu_job_stats["system_memory"] * (cpu_job_stats["allocated.mem"] // 1024)
        ),
    )
    grouped_gpu_stats = gpu_job_stats.groupby(["timestamp"])
    gpu_sum_metrics_years = (
        grouped_gpu_stats[["rgu_equivalent_cost", "cpu_equivalent_cost"]].sum()
        / seconds_in_a_year
    )
    gpu_mean_stats = grouped_gpu_stats[["gpu_utilization", "gpu_mem_gb"]].mean()
    gpu_max_stats = grouped_gpu_stats[["gpu_mem_gb", "cpu_mem_gb"]].max()

    grouped_cpu_stats = cpu_job_stats.groupby(["timestamp"])
    cpu_sum_metrics_years = (
        grouped_cpu_stats[["cpu_equivalent_cost"]].sum() / seconds_in_a_year
    )
    cpu_mean_stats = grouped_cpu_stats[["cpu_mem_gb"]].mean()
    cpu_max_stats = grouped_cpu_stats[["cpu_mem_gb"]].max()

    n_students_per_year = usage_stats.groupby(["timestamp"])["user"].nunique()
    logger.info(f"Number of students with slurm jobs per year: {n_students_per_year}")

    years = sorted(usage_stats["timestamp"].dt.year.unique().astype(int))

    data = {
        "year": years,
        "students": n_students_per_year,
        "gpu_years": gpu_sum_metrics_years["rgu_equivalent_cost"],
        "gpu_mem_mean": gpu_mean_stats["gpu_mem_gb"],
        "gpu_mem_max": gpu_max_stats["gpu_mem_gb"],
        "gpu_util_mean": gpu_mean_stats["gpu_utilization"],
        "gpu_cpu_years": gpu_sum_metrics_years["cpu_equivalent_cost"],
        "gpu_cpu_mem_mean": gpu_mean_stats["gpu_mem_gb"],
        "gpu_cpu_mem_max": gpu_max_stats["cpu_mem_gb"],
        "cpu_years": cpu_sum_metrics_years["cpu_equivalent_cost"],
        "cpu_mem_mean": cpu_mean_stats["cpu_mem_gb"],
        "cpu_mem_max": cpu_max_stats["cpu_mem_gb"],
    }
    data = pd.DataFrame(data)
    # Change the `year` column to have int dtype:
    data = data.astype({"year": int})
    return data


def get_group_usage_by_student(
    prof_email: str,
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2025, 1, 1),
) -> pd.DataFrame:
    """Returns the total compute usage for a prof's group in the given period."""

    students = get_group_students(prof_email)
    logger.info(f"{prof_email} has apparently {len(students)} students.")

    options = Options(
        start=start.astimezone(MTL),
        end=end.astimezone(MTL),
        user=[s.mila.email for s in students],
    )
    sarc_data = _get_cleaned_df(options)
    usage_stats = _get_stats(sarc_data, options, frame_size="YS")
    gpu_job_stats = usage_stats[usage_stats["requested.gres_gpu"] > 0]
    cpu_job_stats = usage_stats[usage_stats["requested.gres_gpu"] == 0]
    # Create two new columns for the CPU and GPU memory usage in gigabytes.
    gpu_job_stats = gpu_job_stats.assign(
        gpu_mem_gb=(
            gpu_job_stats["gpu_memory"]
            * gpu_job_stats["allocated.gpu_type"].map(_gpu_ram)
            # note: don't multiply by # of gpus.
            # * gpu_job_stats["allocated.gres_gpu"]
        ),
        cpu_mem_gb=(
            # system_memory is a percentage, allocated.mem is in MB (I think).
            gpu_job_stats["system_memory"] * (gpu_job_stats["allocated.mem"] // 1024)
        ),
    )
    cpu_job_stats = cpu_job_stats.assign(
        cpu_mem_gb=(
            cpu_job_stats["system_memory"] * (cpu_job_stats["allocated.mem"] // 1024)
        ),
    )
    grouped_gpu_stats = gpu_job_stats.groupby(["timestamp", "user"])
    gpu_sum_metrics_years = (
        grouped_gpu_stats[["rgu_equivalent_cost", "cpu_equivalent_cost"]].sum()
        / seconds_in_a_year
    )
    gpu_mean_stats = grouped_gpu_stats[["gpu_utilization", "gpu_mem_gb"]].mean()
    gpu_max_stats = grouped_gpu_stats[["gpu_mem_gb", "cpu_mem_gb"]].max()

    grouped_cpu_stats = cpu_job_stats.groupby(["timestamp", "user"])
    cpu_sum_metrics_years = (
        grouped_cpu_stats[["cpu_equivalent_cost"]].sum() / seconds_in_a_year
    )
    cpu_mean_stats = grouped_cpu_stats[["cpu_mem_gb"]].mean()
    cpu_max_stats = grouped_cpu_stats[["cpu_mem_gb"]].max()

    years = sorted(usage_stats["timestamp"].dt.year.astype(int).unique())

    values: list[dict] = []
    index: list[tuple[int, str]] = []

    def _slice_and_get_value(
        df: pd.DataFrame,
        column: str,
        default: float,
        timestamp: pd.Timestamp,
        user: str,
    ) -> float:
        return df.xs(timestamp, level="timestamp")[column].get(user, default)

    timestamps = sorted(usage_stats["timestamp"].unique())
    assert len(years) == len(timestamps)
    for year, timestamp in zip(years, timestamps):
        for student in students:
            user = student.mila.username
            index.append((year, student.mila.email))
            _slice = functools.partial(
                _slice_and_get_value, timestamp=timestamp, user=user, default=0.0
            )
            user_year_values = {
                "user": user,
                "year": year,
                "gpu_years": _slice(gpu_sum_metrics_years, "rgu_equivalent_cost"),
                "gpu_mem_mean": _slice(gpu_mean_stats, "gpu_mem_gb"),
                "gpu_mem_max": _slice(gpu_max_stats, "gpu_mem_gb"),
                "gpu_util_mean": _slice(gpu_mean_stats, "gpu_utilization"),
                "gpu_cpu_years": _slice(gpu_sum_metrics_years, "cpu_equivalent_cost"),
                "gpu_cpu_mem_mean": _slice(gpu_mean_stats, "gpu_mem_gb"),
                "gpu_cpu_mem_max": _slice(gpu_max_stats, "cpu_mem_gb"),
                "cpu_years": _slice(cpu_sum_metrics_years, "cpu_equivalent_cost"),
                "cpu_mem_mean": _slice(cpu_mean_stats, "cpu_mem_gb"),
                "cpu_mem_max": _slice(cpu_max_stats, "cpu_mem_gb"),
            }
            values.append(user_year_values)
    # Could also make a multiindex, but makes it a bit harder to work with.
    # return pd.DataFrame.from_records(
    #     values, index=pd.MultiIndex.from_tuples(index, names=["year", "user"])
    # )
    df = pd.DataFrame.from_records(values)
    return df


def get_group_usage_projections(prof_email: str) -> pd.DataFrame:
    """Dummy function that returns random projection data for years 2025-2026."""
    group_usage = get_group_usage(prof_email)
    n_predictions = 2
    next_two_years = group_usage["year"].max() + np.arange(1, 1 + n_predictions)

    extrapolations = extrapolate_linear(group_usage, next_two_years).clip(lower=0)
    # Note: round students to the nearest integer? (small detail perhaps)
    extrapolations = extrapolations.astype({"year": int}).assign(
        students=extrapolations["students"].round()
    )
    return extrapolations


def extrapolate_linear(group_usage: pd.DataFrame, new_x: list[int]) -> pd.DataFrame:
    # Linear extrapolation function
    x = group_usage["year"].to_numpy().astype(int)
    result: list[np.ndarray] = []
    for col in group_usage.columns:
        if col == "year":
            result.append(np.asarray(new_x))
            continue
        y = group_usage[col].values
        coeffs = np.polyfit(x, y, 1)  # Linear fit
        extrapolated_vals = np.poly1d(coeffs)(new_x)
        result.append(extrapolated_vals)
    extrapolated_df = pd.DataFrame(
        np.vstack(result).T, index=new_x, columns=group_usage.columns
    )
    return extrapolated_df


def _print_like_form_shows(df: pd.DataFrame):
    print("Year," + ",".join(df["year"].astype(str).tolist()))
    # print("Students," + ",".join(df["students"].astype(str).tolist()))
    columns = [
        "students",
        "gpu_years",
        "gpu_mem_mean",
        "gpu_mem_max",
        "gpu_util_mean",
        "gpu_cpu_years",
        "gpu_cpu_mem_mean",
        "gpu_cpu_mem_max",
        "cpu_years",
        "cpu_mem_mean",
        "cpu_mem_max",
    ]
    for column in columns:
        vals = df[column]
        print(column + ", " + ", ".join(vals.map(lambda x: f"{x:.3f}").tolist()))


def _compare_survey_answers_with_SARC():
    """Unused atm: Compares the survey answers with Sarc data and displays a plot."""
    # Set to `True` to enable interactive mode to annotate the survey results manually.
    interactive = False

    # Can use the command-line flags to filter the sarc (and survey) data for a specific user.
    # Optionally filter the survey data (and sarc data) for a given user. Here we just reuse the Options class.
    filtering_options = simple_parsing.parse(Options)
    _setup_logging(filtering_options.verbose)
    filtering_user_emails = filtering_options.get_users(assume_mila_email=True)

    survey_answers_csv = (
        Path(__file__).parent / "Sondages - Compute Forecast - answers.csv"
    )
    # TODO: get the survey answers programmatically (need to setup google Oauth)
    # import gspread
    # gc = gspread.oauth()
    # sh = gc.open(options.papers)
    # papers = pd.DataFrame(sh.worksheet("answers").get_all_records())

    survey_data, raw_survey_data = _load_survey_data(survey_answers_csv)
    # Gets the sarc data to cover all users and data ranges and clusters mentioned in the survey.
    overall_survey_period_options = _get_options_that_cover_survey_period(survey_data)
    rich.print("Overall survey period, users, and clusters: ")
    rich.pretty.pprint(overall_survey_period_options)
    all_sarc_data = _get_cleaned_df(overall_survey_period_options)

    if filtering_user_emails:
        raw_survey_data = _filter_survey_data_by_users(
            raw_survey_data, user_emails=filtering_user_emails
        )
        survey_data = _filter_survey_data_by_users(
            survey_data, user_emails=filtering_user_emails
        )

    survey_entries = survey_data.to_dict(orient="records")
    raw_survey_entries = raw_survey_data.to_dict(orient="records")
    assert len(survey_entries) and len(survey_entries) == len(raw_survey_entries), (
        len(survey_entries),
        len(raw_survey_entries),
    )
    kept_survey_entries: list[dict] = []
    dropped_survey_entries: list[dict] = []
    annotated_gpu_hour_estimates_per_user_per_entry: list[dict[str, Estimate]] = []
    survey_entry_filters: list[Options] = []
    for i, (answers_dict, _raw_survey_entry) in enumerate(
        zip(survey_entries, raw_survey_entries)
    ):
        survey_entry_df = survey_data.iloc[[i]]
        # Get a filter based on the survey entry content, that can be used to query (or filter) the SARC data.
        survey_entry_period_options = _get_options_that_cover_survey_period(
            survey_entry_df
        )

        # Display the data nicely so that it can be read as part of the interactive prompt below.
        if interactive:
            (f"Survey entry #{i}")
            _display_survey_entry(answers_dict)

        # To avoid having to re-enter some previously annotated data, we use the cache dir and maybe
        # a flag to clear the cache.

        # todo: The start / end dates are sometimes missing for a given survey entry.
        # TODO: Select some start data from either SARC or the survey data.
        if pd.isna(survey_entry_period_options.start):
            logger.warning(f"Missing a start date for survey entry {i}!")
            survey_entry_period_options = dataclasses.replace(
                survey_entry_period_options, start=overall_survey_period_options.start
            )
        if pd.isna(survey_entry_period_options.end):
            logger.warning(f"Missing an end date for survey entry {i}!")
            survey_entry_period_options = dataclasses.replace(
                survey_entry_period_options, end=overall_survey_period_options.end
            )
        logger.debug(f"Survey entry period: {survey_entry_period_options}")

        gpu_hours_estimates = _extract_gpu_hours_from_survey_entry(answers_dict)
        if interactive:
            print(
                f"Estimated gpu*hours extracted from the survey answers: {gpu_hours_estimates}"
            )
        if gpu_hours_annotation := _get_existing_annotation(
            answers_dict, cache_dir=filtering_options.cache_dir
        ):
            if interactive:
                print(f"Previous annotation: {gpu_hours_annotation}")
            if interactive and not rich.prompt.Confirm.ask(
                "Keep the existing annotation?"
            ):
                gpu_hours_annotation = _get_estimate_from_user(
                    answers_dict, gpu_hours_annotation
                )
        elif interactive and rich.prompt.Confirm.ask("Adjust the value manually?"):
            gpu_hours_annotation = _get_estimate_from_user(
                answers_dict, previous_annotation=gpu_hours_annotation
            )

        if gpu_hours_annotation is not None:
            _save_annotation(
                answers_dict,
                annotation=gpu_hours_annotation,
                cache_dir=filtering_options.cache_dir,
            )
        else:
            # Use the value extracted from the survey answers.
            gpu_hours_annotation = gpu_hours_estimates

        sarc_data_for_this_paper = _filter_sarc_data(
            all_sarc_data,
            filtering_options=survey_entry_period_options,
        )

        if not len(sarc_data_for_this_paper):
            logger.error(
                RuntimeError(
                    f"There is apparently no data in SARC covering the survey "
                    f"entry #{i} for the paper '{answers_dict['Paper Title']}'!\n"
                    f"(Filter used for that entry: {survey_entry_period_options})"
                ),
                extra={"style": "bold red"},
            )
            dropped_survey_entries.append(answers_dict)
            continue
            # breakpoint()

        kept_survey_entries.append(answers_dict)
        survey_entry_filters.append(survey_entry_period_options)
        # sarc_data_per_survey_entry.append(resource_hours_by_user_and_workdir)
        annotated_gpu_hour_estimates_per_user_per_entry.append(gpu_hours_annotation)

    logger.info(
        f"{len(kept_survey_entries)} out of {len(survey_entries)} survey answers had associated data in SARC"
    )
    if dropped_survey_entries:
        rich.print("Survey answers with no SARC data and their filters:")
        rich.pretty.pprint(
            {
                dropped_entry["Paper Title"]: _get_options_that_cover_survey_period(
                    survey_data.iloc[[survey_entries.index(dropped_entry)]]
                )
                for dropped_entry in dropped_survey_entries
            }
        )

    # Idea: Annotate the plots with the data from the survey. (TODO: How?)
    # sarc_data_per_entry = pd.concat(sarc_data_per_survey_entry)

    comparison_df_data: dict[str, pd.DataFrame] = {}
    for i, (answers_dict, entry_filter, estimated_usage_from_answers) in enumerate(
        zip(
            kept_survey_entries,
            survey_entry_filters,
            # sarc_data_per_survey_entry,
            annotated_gpu_hour_estimates_per_user_per_entry,
        )
    ):
        sarc_data_for_this_paper = _filter_sarc_data(all_sarc_data, entry_filter)
        usage_stats = _get_stats(sarc_data_for_this_paper, entry_filter)

        usage_by_user = (
            usage_stats.groupby(["user.mila.email"])[
                [
                    # "cpu_billed",
                    "cpu_cost",
                    # "cpu_equivalent_cost",
                    # "gpu_billed",
                    "gpu_cost",
                    # "gpu_equivalent_cost",
                    # "rgu_equivalent_cost",
                ]
            ]
            .sum()
            .divide(3600)
        )
        # print(f"What they say they used:")
        usage_by_user = usage_by_user.assign(
            survey_gpuhours_min=pd.Series(
                {k: v.min for k, v in estimated_usage_from_answers.items()}
            ),
            survey_gpuhours_max=pd.Series(
                {k: v.max for k, v in estimated_usage_from_answers.items()}
            ),
        )
        comparison_df_data[answers_dict["Paper Title"]] = usage_by_user

        logger.debug(
            "SARC data vs survey answers for entry #%s} (NOTE: SARC data may include other projects!):\n%s",
            i,
            usage_by_user.to_markdown(),
        )

        # TODO: Ask users which directory (given a list) they
        k = 5
        logger.info(
            "Top %s directories where most compute was allocated for that entry period:\n%s",
            k,
            usage_stats.groupby(["user.mila.email", "work_dir"])[
                ["cpu_cost", "gpu_cost"]
            ]
            .sum()
            .divide(3600)
            .nlargest(k, "gpu_cost"),
        )
        logger.debug(
            f"Top {k} job names where most compute was allocated for that entry period:"
        )
        logger.debug(
            usage_stats.groupby(["user.mila.email", "name"])[["cpu_cost", "gpu_cost"]]
            .sum()
            .divide(3600)
            .nlargest(k, "gpu_cost")
            .to_markdown()
        )

    comparison_df = pd.concat(
        comparison_df_data, names=["Paper Title", "user.mila.email"]
    )
    comparison_df = comparison_df.rename(
        {"cpu_cost": "cpu_hours", "gpu_cost": "gpu_hours"}, axis=1
    )
    # comparison_df.groupby("Paper Title").sum().drop(columns="cpu_hours").plot.hist()
    _plot_comparison(comparison_df)
    plt.show()
    comparison_df.to_csv("comparison.csv")


def _plot_comparison(comparison_df):
    # Assuming `comparison_df` is your DataFrame
    grouped = comparison_df.groupby("Paper Title").sum()

    # Extract data
    papers = grouped.index
    actual_gpu_hours = grouped["gpu_hours"]
    survey_min = grouped["survey_gpuhours_min"]
    survey_max = grouped["survey_gpuhours_max"]

    # Display the minimum and maximum as a range of some sort, and the actual value alongside it.
    fig, ax = plt.subplots(figsize=(10, 6))
    # For each paper (x axis), display a histogram with three bars: min, max, and actual.
    # Use a log scale for the y axis, and annotate the bars with the actual values.
    bar_width = 0.2
    x = np.arange(len(papers))
    ax.bar(x - bar_width, survey_min, width=bar_width, label="Survey Min")
    ax.bar(x, actual_gpu_hours, width=bar_width, label="Actual")
    ax.bar(x + bar_width, survey_max, width=bar_width, label="Survey Max")
    ax.set_xticks(x)
    ax.set_xticklabels(papers, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("GPU Hours (log scale)")
    ax.set_title("GPU Hours Comparison")
    ax.legend()


def _get_estimate_from_user(
    survey_entry: dict, previous_annotation: dict[str, Estimate] | None
) -> dict[str, Estimate]:
    previous_annotation = previous_annotation or {}
    # todo: Need to set a value for each of the authors.
    authors = sorted(
        set(
            _author
            for k, v in survey_entry.items()
            if "Email" in k and (_author := _none_if_nan(v)) is not None
        )
    )
    annotations_per_author: dict[str, Estimate] = {}
    for author in authors:
        existing_estimate = previous_annotation.get(author)

        min_gpu_hours = rich.prompt.FloatPrompt(
            "What is the [bold]minimum[/bold] GPU*hour given the above data?"
        )(default=existing_estimate.min if existing_estimate else None)
        assert isinstance(min_gpu_hours, float)

        max_gpu_hours = rich.prompt.FloatPrompt(
            "What is the [bold]maximum[/bold] GPU*hour given the above data?"
        )(default=existing_estimate.max if existing_estimate else min_gpu_hours)
        assert isinstance(max_gpu_hours, float)
        assert max_gpu_hours >= min_gpu_hours

        annotation = Estimate(min=min_gpu_hours, max=max_gpu_hours)

        assert author not in annotations_per_author
        annotations_per_author[author] = annotation
    return annotations_per_author


def _get_existing_annotation(
    survey_entry: dict, cache_dir: Path
) -> dict[str, Estimate] | None:
    """Gets the existing annotation for a given survey entry.

    The annotation is stored in a file named after the survey entry's hash.
    """
    annotation_file = _get_annotation_cache_file(survey_entry, cache_dir)
    if annotation_file.exists():
        with open(annotation_file, "r") as f:
            data = yaml.safe_load(f)
            return {k: Estimate(**v) for k, v in data.items()}
    return None


def _save_annotation(
    survey_entry: dict, annotation: dict[str, Estimate], cache_dir: Path
) -> None:
    """Saves the annotation for a given survey entry.

    The annotation is stored in a file named after the survey entry's hash.
    """
    annotation_file = _get_annotation_cache_file(survey_entry, cache_dir)
    with open(annotation_file, "w") as f:
        yaml.safe_dump({k: dataclasses.asdict(v) for k, v in annotation.items()}, f)


def _serialize_survey_entry(survey_entry: dict) -> dict:
    return {
        k: (str(v) if isinstance(v, pd.Timestamp) else v if not pd.isna(v) else None)
        for k, v in survey_entry.items()
    }


def _get_annotation_cache_file(survey_entry: dict, cache_dir: Path):
    serialized_survey_entry = _serialize_survey_entry(survey_entry)
    for k, v in serialized_survey_entry.items():
        try:
            json.dumps(v)
        except TypeError:
            breakpoint()
    survey_entry_hash = hashlib.md5(
        json.dumps(serialized_survey_entry).encode()
    ).hexdigest()
    annotation_file = cache_dir / f"annotation_{survey_entry_hash}.yaml"
    return annotation_file


def _display_survey_entry(survey_entry: dict):
    values = _drop_na_values(survey_entry)
    rich.print(
        rich.panel.Panel(
            rich.pretty.pretty_repr(
                {
                    k: v
                    if isinstance(v, str) and len(v) >= 30
                    else rich.pretty.pretty_repr(v)
                    for k, v in values.items()
                }
            )
        )
    )


def _extract_gpu_hours_from_survey_entry(survey_entry: dict) -> dict[str, Estimate]:
    """Gets the estimated GPU*hours from a survey entry.

    Returns the minimum and maximum values for the product of the
    "Number of slurm jobs", "Average running time per job",
    "Number of GPUs per experiment", and "Average GPU-utilization" columns
    for each author.
    """
    usage_per_user: dict[str, Estimate] = {}

    number_of_jobs_map: dict[str, tuple[float, float]] = {
        "100 or less": (0, 100),
        "(100, 500]": (100, 500),
        "(500, 1000]": (500, 1000),
        "(1000, 2000]": (1000, 2000),
        "(2000, 5000]": (2000, 5000),
        "(5000, 10000]": (5000, 10_000),
        "more than 10000": (
            10_000,
            20_000,  # todo: what to use as a maximum estimate in this case?
        ),
    }
    running_time_map: dict[str, tuple[timedelta, timedelta]] = {
        "3h or less": (timedelta(0), timedelta(hours=3)),
        "(3h, 12h]": (timedelta(hours=3), timedelta(hours=12)),
        "(12h, 24h]": (timedelta(hours=12), timedelta(hours=24)),
        "(24h, 48h]": (timedelta(hours=24), timedelta(hours=48)),
        "(2d,  4d]": (timedelta(days=2), timedelta(days=4)),
        # TODO: Need to invent a maximum here.
        "more than 4d": (timedelta(days=4), timedelta(days=7)),
    }
    gpus_per_job_map: dict[str, tuple[int, int]] = {
        "0": (0, 0),
        "1": (1, 1),
        "2": (2, 2),
        "(2, 4]": (2, 4),
        "(5, 8]": (5, 8),
        "more than 8": (8, 16),  # TODO: Need to invent a maximum here.
    }

    section_suffixes = ["", *[f".{i}" for i in range(1, 10)]]
    for section_index, section_suffix in enumerate(section_suffixes):
        # note: Seems to always be filled, with `nan` when missing (not None)
        author: str | None = _none_if_nan(
            survey_entry[
                f"Email of the co-author who ran these experiments{section_suffix}"
            ]
        )
        number_of_jobs: str | None = _none_if_nan(
            survey_entry[f"Number of slurm jobs{section_suffix}"]
        )
        running_time: str | None = _none_if_nan(
            survey_entry[f"Average running time per slurm job{section_suffix}"]
        )
        gpus_per_job: str | None = _none_if_nan(
            survey_entry[f"Number of GPUs per experiment{section_suffix}"]
        )

        average_gpu_util: str | None = _none_if_nan(
            survey_entry[f"Average GPU-utilization{section_suffix}"]
        )

        if any([author, number_of_jobs, running_time, gpus_per_job, average_gpu_util]):
            logger.debug(
                f"Raw answers for section {section_index}: "
                f"{author=}, {number_of_jobs=}, {running_time=}, {gpus_per_job=}, {average_gpu_util=}"
            )

        if (
            number_of_jobs is not None
            and running_time is not None
            and gpus_per_job is not None
        ):
            # Assume that if the "author" field in a group isn't filled, it's the user that is answering the form.
            _author = author or survey_entry["Email address"]
            _gpus_per_job_min, _gpus_per_job_max = gpus_per_job_map[gpus_per_job]
            _number_of_jobs_min, _number_of_jobs_max = number_of_jobs_map[
                number_of_jobs
            ]
            _running_time_min, _running_time_max = running_time_map[running_time]
            if average_gpu_util is None:
                # todo: use the user's average util from SARC maybe?
                _average_gpu_util = 0.5
            else:
                _average_gpu_util = float(average_gpu_util) / 10.0

            minimum_gpu_hours = (
                _number_of_jobs_min
                * _gpus_per_job_min
                * (_running_time_min.total_seconds() / 3600)
                * _average_gpu_util
            )
            maximum_gpu_hours = (
                _number_of_jobs_max
                * _gpus_per_job_max
                * (_running_time_max.total_seconds() / 3600)
                * _average_gpu_util
            )
            if existing_entry := usage_per_user.get(_author):
                usage_per_user[_author] = existing_entry + Estimate(
                    min=minimum_gpu_hours, max=maximum_gpu_hours
                )
            else:
                usage_per_user[_author] = Estimate(
                    min=minimum_gpu_hours, max=maximum_gpu_hours
                )
        else:
            # dont' allow sparse entries for now.
            assert (
                number_of_jobs is None and running_time is None and gpus_per_job is None
            ), (author, number_of_jobs, running_time, gpus_per_job)
    return usage_per_user


def _none_if_nan(v: T) -> T | None:
    try:
        float_v = float(v)  # type: ignore
        if math.isnan(float_v):
            return None
    except ValueError:
        pass
    return v


def _drop_na_values(d: Mapping):
    return {k: v for k, v in d.items() if not pd.isna(v)}


def _filter_survey_data_by_users(
    survey_data: pd.DataFrame, user_emails: list[str]
) -> pd.DataFrame:
    user_emails = list(map(_check_is_email_and_lower, user_emails))

    email_columns = [c for c in survey_data.columns if "Email" in c]
    mask = np.zeros(len(survey_data), dtype=bool)
    for col in email_columns:
        mask |= (
            survey_data[col]
            .where(pd.notna, other="")
            .map(_check_is_email_and_lower)
            .isin(user_emails)
        )
    return survey_data[mask]


def _filter_sarc_data(
    all_sarc_data_cleaned: pd.DataFrame, filtering_options: Options
) -> pd.DataFrame:
    users = filtering_options.get_users()
    users = list(map(_check_is_email_and_lower, users))
    df = all_sarc_data_cleaned[all_sarc_data_cleaned["user.mila.email"].isin(users)]
    df = df[df["cluster_name"].isin(filtering_options.clusters)]
    df = df[df["start_time"].between(filtering_options.start, filtering_options.end)]
    return df


def _check_is_email_and_lower(v: str):
    if not v:
        return v
    if "@" not in v:
        raise ValueError(f"'{v}' is not a valid email address.")
    return v.lower()

    # logger.debug(options)

    # users = options.get_users()
    # df = get_cleaned_df(options)


def _get_stats(
    sarc_data: pd.DataFrame, options: Options, frame_size: timedelta | str | None = None
) -> pd.DataFrame:
    stats = compute_time_frames(
        sarc_data,
        ["gpu_cost", "cpu_cost", "cpu_equivalent_cost", "gpu_equivalent_cost"],
        start=options.start,
        end=options.end,
        frame_size=(
            frame_size
            if frame_size is not None
            else "MS"
            if (_period := (options.end - options.start)) > timedelta(days=90)
            else timedelta(days=7)
            if _period > timedelta(days=30)
            else timedelta(days=1)
        ),
    )
    stats = stats.assign(
        rgu_equivalent_cost=(
            stats["gpu_equivalent_cost"] * stats["allocated.gpu_type_rgu"]
        )
    )

    return stats


def _load_survey_data(survey_data_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sd = pd.read_csv(survey_data_csv, header=1)
    raw_data = sd
    date_columns = [c for c in sd.columns if "Start date" in c or "End date" in c]
    sd = sd.assign(
        User=sd["Email address"].map(lambda v: v.removesuffix("@mila.quebec")),
        **{
            c: pd.to_datetime(sd[c], format="%d/%m/%Y")
            .dt.tz_localize("UTC")  # this gymnastic seems needed to get a datetime
            .dt.tz_convert(MTL)
            for c in date_columns
        },
    )
    clusters_used_columns = [c for c in sd.columns if "Which clusters did you use" in c]
    new_used_cluster_columns: dict[str, np.ndarray[tuple[int], np.dtype[np.bool]]] = {
        c: np.zeros(len(sd), dtype=bool) for c in ALL_CLUSTERS
    }
    for cluster in ALL_CLUSTERS:
        used_cluster = np.stack(
            [
                sd[column].map(
                    lambda v: cluster in v.lower() if isinstance(v, str) else False
                )
                for column in clusters_used_columns
            ]
        ).any(0)
        new_used_cluster_columns[cluster] = used_cluster  # type: ignore
    sd = sd.assign(
        **{
            f"{cluster}_used": cluster_was_used
            for cluster, cluster_was_used in new_used_cluster_columns.items()
        }
    )
    return sd, raw_data


def _get_options_that_cover_survey_period(survey_data: pd.DataFrame) -> Options:
    """Return the filter to use to fetch SARC data for the given the survey answer(s)."""
    email_columns = [c for c in survey_data.columns if "Email" in c]
    users = set()
    for email_column in email_columns:
        _users = set(survey_data[email_column].dropna().tolist())
        users |= _users
    user_emails = sorted(users)

    clusters_used_columns = [
        c
        for c in survey_data.columns
        if c.startswith("Which clusters did you use for these experiments?")
    ]

    start_date_columns = [c for c in survey_data.columns if c.startswith("Start date")]
    end_date_columns = [c for c in survey_data.columns if c.startswith("End date")]

    earliest_start_date: datetime = min(
        survey_data[c].dropna().min() for c in start_date_columns
    )
    latest_end_date: datetime = max(
        survey_data[c].dropna().max() for c in end_date_columns
    )

    # note: can contain things like 'mila, beluga' and 'other clusters' etc.
    known_clusters_used: set[str] = set()
    for column in clusters_used_columns:
        all_clusters_used = survey_data[column].dropna().unique()
        for cluster_used_entry in all_clusters_used:
            for cluster in ALL_CLUSTERS:
                if cluster in cluster_used_entry.lower():
                    known_clusters_used.add(cluster)
    # clusters
    return Options(
        user=user_emails,
        start=earliest_start_date,
        end=latest_end_date,
        clusters=list(known_clusters_used),
    )


def _get_cleaned_df(options: Options) -> pd.DataFrame:
    """Gets "cleaned" SARC data for a given period, including *lots* of patches."""
    cache_file = options.unique_path()
    _user_emails = options.get_users(assume_mila_email=True)
    assert all(map(_check_is_email_and_lower, _user_emails))
    users = [user.partition("@")[0] for user in _user_emails]

    logger.info(
        f"Looking up for data between {options.start} and {options.end} for users: {users or 'all'} and clusters {options.clusters or 'all'}"
    )
    if cache_file.exists():
        logger.info(f"Reading previous data from {cache_file}.")
        df = pd.read_pickle(cache_file)
        assert isinstance(df, pd.DataFrame)
    elif (
        options.user
        and (
            all_users_cache_file := dataclasses.replace(
                options, user=[], users_file=None
            ).unique_path()
        ).exists()
    ):
        logger.info(
            f"Reusing and filtering previous data for all users at {all_users_cache_file}."
        )
        df = pd.read_pickle(all_users_cache_file)
        assert isinstance(df, pd.DataFrame)
        df = df[df["user"].isin(users)]
    else:
        logger.info(
            f"Did not find previous results at {cache_file}. Fetching job data."
        )
        df = load_job_series(
            start=options.start,
            end=options.end,
            user=(
                {"$in": users}
                if (isinstance(users, list) and users)
                else users
                if users
                else None
            ),  # support querying for multiple users.
            clip_time=False,  # True,
        )
        logger.info(f"Saving data to {cache_file}")
        df.to_pickle(cache_file)

    for time_column in ["submit_time", "start_time", "end_time"]:
        # df[time_column] = df[time_column].dt.tz_localize("UTC").dt.tz_convert(MTL)
        df[time_column] = df[time_column].dt.tz_convert(MTL)

    _validate_gpu_ram()

    # Clusters we want to compare
    if options.clusters:
        # Filter clusters
        df = df[df["cluster_name"].isin(options.clusters)]

    df.fillna({"requested.gres_gpu": 0, "allocated.gres_gpu": 0}, inplace=True)
    df = _fix_lost_jobs(df)
    df = _fix_unaligned_cache(df, options.start, options.end)
    df = _remove_old_nodes(df)
    df = _replace_outlier_stats_with_na(df)
    df = _fix_missing_gpu_type(df)

    _fix_rgu_discrepencies_inplace(df)
    # todo: double-check if this is still needed here.
    df.fillna({"requested.gres_gpu": 0, "allocated.gres_gpu": 0}, inplace=True)

    _fix_allocated_cpus_drac_inplace(df)

    # todo: Do we want to get the averages from the other jobs of the same user on other clusers?
    # Or from the average utilization on that same cluster by different users?
    # IF so, we might need to reload the data for all users here
    # if users := options.get_users():
    #     all_users_data_for_same_period = _get_cleaned_df(
    #         dataclasses.replace(options, user=[], user_file=None)
    #     )
    df = _fill_missing_metrics_using_means(df)
    df = compute_cost_and_waste(df)

    # Sanity checks.
    assert (df["start_time"] != 0).all()
    assert (_requested_gres_gpu := df["requested.gres_gpu"]).notnull().all() and (
        _requested_gres_gpu >= 0
    ).all()

    df = _set_cpu_gpu_billed(df)

    df, missing_users = _find_missing_user_to_mila_emails(df)
    if missing_users:
        print(f"Missing the mila email for these users: {sorted(missing_users)}")

    return df


def _find_missing_user_to_mila_emails(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing_mila_email = df["user.mila.email"].isna()
    missing_mila_email_users = df[missing_mila_email]["user"].unique()

    n_missing = missing_mila_email.sum()
    if not n_missing:
        return df, []

    N = df.shape[0]
    logger.warning(
        f"'user.mila.email' is missing in {n_missing} jobs ({n_missing / N:.2%}) "
        f"from {len(missing_mila_email_users)} users."
    )

    # Find jobs from the same users, where the email is not missing.
    missing_user_to_emails = df[
        df["user"].isin(missing_mila_email_users) & df["user.mila.email"].notna()
    ][["user", "user.mila.email"]]

    # NOTE: edge case here: Is it be possible for the same user to have different emails?
    # If so, here calling `dict` will use the last email found.
    user_to_email_dict = dict(
        list(missing_user_to_emails.drop_duplicates().itertuples(index=False))
    )
    # Use those to fill the missing entries.
    df.loc[missing_mila_email, "user.mila.email"] = df.loc[
        missing_mila_email, "user"
    ].map(user_to_email_dict)

    still_missing_mila_email = df["user.mila.email"].isna()
    still_missing_mila_email_users = df[still_missing_mila_email]["user"].unique()

    n_still_missing = still_missing_mila_email.sum()
    n_users_fixed = len(
        set(missing_mila_email_users) - set(still_missing_mila_email_users)
    )
    n_fixed = n_missing - n_still_missing
    logger.info(
        f"Able add the missing 'user.mila.email' for the {n_fixed} jobs from {n_users_fixed} users "
        f"({(n_fixed / n_missing):.2%} of the jobs with a missing value)."
    )
    return df, sorted(still_missing_mila_email_users)


def _replace_outlier_stats_with_na(df: pd.DataFrame):
    """`load_job_series` only removes the H100 outliers for gpu_utilization.

    Here we do the rest.
    """
    # Shouldn't really be necessary anymore, but still.
    df = df.assign(
        **{
            col: df[col].where(lambda v: (0 <= v) & (v <= 1), pd.NA)
            for col in [
                "gpu_utilization",
                *(f"gpu_utilization_fp{bits}" for bits in [16, 32, 64]),
                "gpu_memory",
                "gpu_sm_occupancy",
            ]
        },
        gpu_power=df["gpu_power"].where(lambda v: (0 <= v) & (v <= 10e10), pd.NA),
    )
    # df.loc[df["gpu_utilization"] > 1, "gpu_utilization"] = pd.NA
    # for bits in [16, 32, 64]:
    #     df.loc[df[f"gpu_utilization_fp{bits}"] > 1, f"gpu_utilization_fp{bits}"] = pd.NA
    # df.loc[df["gpu_memory"] > 1, "gpu_memory"] = pd.NA
    # df.loc[df["gpu_sm_occupancy"] > 1, "gpu_sm_occupancy"] = pd.NA
    # df.loc[df["gpu_power"] > 10e10, "gpu_power"] = pd.NA
    return df


def _fill_missing_metrics_using_means(
    df: pd.DataFrame, all_users_data: pd.DataFrame | None = None
):
    """Fill in missing JobStatistics metrics using average of available data."""
    stat_columns = list(JobStatistics.__fields__.keys())
    clusters = df["cluster_name"].unique()

    # no_na = df.dropna(subset=stat_columns, how="any")
    # assert no_na.shape[0] > 0

    across_cluster_means = {col: df[col].dropna().mean() for col in stat_columns}
    logger.debug(
        f"Mean of stats across all clusters: {_get_stats_str(across_cluster_means)}"
    )

    # todo: cpu_utilization and system_memory should be there for all jobs, right?
    gpu_columns = [col for col in stat_columns if col.startswith("gpu")]
    cpu_system_stats_columns = list(set(stat_columns) - set(gpu_columns))

    assert df["allocated.gres_gpu"].notna().all()

    # Create some masks
    has_gpu = df["allocated.gres_gpu"] > 0
    is_missing_gpu_stats = has_gpu & df[gpu_columns].isna().any(axis="columns")
    is_missing_system_stats = df[cpu_system_stats_columns].isna().any(axis="columns")

    sparsity_info_across_clusters = {
        "has_gpu": has_gpu.mean(),
        "is_missing_gpu_stats": is_missing_gpu_stats.mean(),
        "is_missing_system_stats": is_missing_system_stats.mean(),
    }
    logger.info({k: f"{v:.2%}" for k, v in sparsity_info_across_clusters.items()})

    for cluster in clusters:
        is_in_cluster = df["cluster_name"] == cluster
        cluster_mean_stats = {
            col: df[is_in_cluster][col].dropna().mean() for col in stat_columns
        }
        # Use the cluster average if possible, otherwise use the average across all clusters.
        missing_stats = [k for k, v in cluster_mean_stats.items() if np.isnan(v)]
        if missing_stats:
            logger.warning(
                f"Missing stats for {cluster=}: {missing_stats}.\n"
                f"The average of available stats across other clusters will be used."
            )
        stats_to_use = {
            col: (
                cluster_mean
                if not np.isnan(cluster_mean)
                else across_cluster_means[col]
                # todo: else use across-users mean.
            )
            for col, cluster_mean in cluster_mean_stats.items()
        }
        missing_stats_str = _get_stats_str(
            {k: v for k, v in stats_to_use.items() if k in missing_stats}
        )
        if missing_stats:
            logger.info(
                f"Stats to be used when infilling missing values for {cluster}: {missing_stats_str}"
            )
        df.loc[is_in_cluster & is_missing_gpu_stats, gpu_columns] = [
            stats_to_use[col] for col in gpu_columns
        ]
        df.loc[is_in_cluster & is_missing_system_stats, cpu_system_stats_columns] = [
            stats_to_use[col] for col in cpu_system_stats_columns
        ]
    return df


def _get_stats_str(stats_to_use: Mapping[str, np.ndarray | float]):
    return {
        k: (f"{v:.1f}" if k == "gpu_power" else f"{v:.2%}")
        for k, v in stats_to_use.items()
    }


def _fix_lost_jobs(df: pd.DataFrame):
    _28_days = timedelta(days=28)
    lost_jobs = df["elapsed_time"] > _28_days.total_seconds()
    df.loc[lost_jobs, "elapsed_time"] = _28_days.total_seconds()
    df.loc[lost_jobs, "end_time"] = df.loc[lost_jobs, "start_time"] + _28_days
    return df


def _fix_unaligned_cache(df: pd.DataFrame, start: datetime, end: datetime):
    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    df = df[df["end_time"].isnull() | (df["end_time"] > start)]
    df = df[df["start_time"].notnull() & (df["start_time"] < end)]

    # print("max start", df["start_time"].max())
    # print("min end", df["end_time"].min())

    return df


def _filter_users(df: pd.DataFrame, users_file: Path):
    with users_file.open("r") as file:
        users = set(file.read().splitlines())
        df = df[df["user"].isin(users)]

    return df


def _remove_old_nodes(df: pd.DataFrame):
    # Filter old nodes and unallocated jobs
    nodes = df["nodes"].str[0]
    old_nodes = [
        "kepler3",
        "kepler4",
        "kepler5",
        "mila01",
        "mila02",
        "mila03",
        "rtx1",
        "rtx3",
        "rtx4",
        "rtx5",
        "rtx7",
    ]
    df = df[~(nodes.isnull() | (nodes.isin(old_nodes)))]
    return df


def _validate_gpu_ram():
    missing_ram = set(_gpu_name_mapping.values()) - set(_gpu_ram.keys())
    if missing_ram:
        raise ValueError(f"Missing ram: {missing_ram}")


# todo: replace with the actual `get_node_to_gpu` function once it works with the client config.
def _get_node_to_gpu(cluster_name: str):
    # node_to_gpu = get_node_to_gpu(cluster_name=cluster_name)
    # return node_to_gpu
    with open(Path(__file__).parent.parent / "config/node_to_gpu.json") as f:
        cluster_configs: dict[str, dict[str, str]] = json.load(f)
    return cluster_configs[cluster_name]


def _get_cluster_configs() -> dict[str, ClusterConfig]:
    with open(Path(__file__).parent.parent / "config/sarc-dev.json") as f:
        cluster_configs = {
            k: ClusterConfig(**v) for k, v in json.load(f)["clusters"].items()
        }
    return cluster_configs

    with open(Path(__file__).parent.parent / "config/sarc-dev.yaml") as f:
        cluster_configs = {
            k: ClusterConfig(**v)
            for k, v in yaml.safe_load(f)["sarc"]["clusters"].items()
        }
    return cluster_configs


def _fix_missing_gpu_type(df: pd.DataFrame, clusters: list[str] | None = None):
    # Fix missing gpu_type
    if not clusters:
        clusters = df["cluster_name"].unique()  # type: ignore
    assert clusters is not None and len(clusters)

    for cluster_name in clusters:
        node_to_gpu = _get_node_to_gpu(cluster_name=cluster_name)
        # node_to_gpu = get_node_to_gpu(cluster_name=cluster_name)
        assert node_to_gpu is not None
        non_mapped_gpu_types_mask = (
            (df["cluster_name"] == cluster_name)
            & (df["elapsed_time"] > 0)
            & df["allocated.gpu_type"].isnull()
        )
        # NOTE: We assume uniformity of gpu types on all nodes
        nodes = df[non_mapped_gpu_types_mask]["nodes"].str[0]
        # NOTE: some nodes don't have GPUs, so we have 'allocated.gpu_type' set to `None` in that case.
        mapping = {node: node_to_gpu.get(node) for node in nodes.unique()}

        df.loc[non_mapped_gpu_types_mask, "allocated.gpu_type"] = nodes.map(mapping)

    missing_gpu_types_mask = (
        (df["requested.gres_gpu"] > 0)
        * (df["elapsed_time"] > 0)
        * (df["allocated.gpu_type"].isnull())
    )
    missing_gpu_types = df[missing_gpu_types_mask]

    if missing_gpu_types.shape[0] > 0:
        print(
            "GPU types not mapped",
            missing_gpu_types.groupby(["cluster_name"]).count()["id"],
        )
        print(missing_gpu_types["nodes"].str[0].unique())
        breakpoint()

    missing_mappings = set(
        df[~df["allocated.gpu_type"].isnull()]["allocated.gpu_type"].unique()
    ) - set(_gpu_name_mapping.keys())
    if missing_mappings:
        print("Missing mappings:", missing_mappings)
        print(
            df[df["allocated.gpu_type"].isin(missing_mappings)]
            .groupby(["allocated.gpu_type", "cluster_name"])
            .count()["id"]
        )
        breakpoint()
        # How can this produce NaNs if we made sure no GPU types were missing?!

    def _fn(x):
        if x in _gpu_name_mapping:
            return _gpu_name_mapping[x]
        elif x is None:
            return x
        else:
            logger.warning(f"Missing GPU name mapping: {x}")
            return x

    # df["allocated.gpu_type"] = df["allocated.gpu_type"].map(_gpu_name_mapping)
    df["allocated.gpu_type"] = df["allocated.gpu_type"].map(_fn)
    df.fillna({"allocated.gpu_type": "unknown"}, inplace=True)

    # ugly patch:
    unknown_gpu = df["allocated.gpu_type"] == "unknown"
    for cluster_name in clusters:
        mask = (df["cluster_name"] == cluster_name) & unknown_gpu
        df.loc[mask, "allocated.gpu_type"] = (
            df[mask]["nodes"]
            .str[0]
            .map(_get_node_to_gpu(cluster_name))
            .map(_gpu_name_mapping)
        )

    return df


def _fix_allocated_cpus_drac_inplace(df: pd.DataFrame):
    # TODO we should fix this in SARC.
    is_drac = df["cluster_name"] != "mila"
    slice_during_rgu_time = (
        is_drac
        & (df["start_time"] >= datetime(2024, 4, 1, tzinfo=MTL))
        & (df["elapsed_time"] > 0)
    )
    df.loc[slice_during_rgu_time, "allocated.cpu"] /= 1000.0

    # df.loc[df["job_id"] == 48738025, "allocated.cpu"] /= 1000
    # is_narval = df["cluster_name"] == "narval"

    # Here we do it for all timeframes.
    outrageous_num_of_cpus = df["allocated.cpu"] >= 1000
    df.loc[is_drac & outrageous_num_of_cpus, "allocated.cpu"] /= 1000.0


def _fix_rgu_discrepencies_inplace(df: pd.DataFrame) -> None:
    # NOTE: Fixing switch to RGU billing for a second time on Narval
    # narval_config = config().clusters["narval"]
    cluster_configs = _get_cluster_configs()
    narval_config = cluster_configs["narval"]

    assert df["allocated.gres_gpu"].notnull().all()
    assert df["requested.gres_gpu"].notnull().all()

    slice_during_rgu_time = (
        (df["cluster_name"] == "narval")
        & (df["start_time"] >= datetime(2023, 11, 28, tzinfo=MTL))
        & (
            df["start_time"]
            < datetime.fromisoformat(narval_config.rgu_start_date).astimezone(MTL)
        )
        & (df["elapsed_time"] > 0)
    )
    non_updated_df = df[slice_during_rgu_time]

    # NOTE: Hacky fix, because we don't use the sarc-dev config.
    # df = update_job_series_rgu(df)
    # for cluster_config in config().clusters.values():
    #     update_cluster_job_series_rgu(df, cluster_config)
    # return df
    for cluster_config in cluster_configs.values():
        # Make sure that we are indeed doing this processing for each cluster.
        assert cluster_config.name
        # name = cluster_config.host
        if cluster_config.name == "mila":
            assert (
                cluster_config.rgu_start_date is None
                and cluster_config.gpu_to_rgu_billing is None
            )
        else:
            assert (
                cluster_config.rgu_start_date
                and cluster_config.gpu_to_rgu_billing
                and Path(cluster_config.gpu_to_rgu_billing).is_file()
            )
        # note: This might introduce some NANs in the `allocated.gres_gpu` for some jobs.
        update_cluster_job_series_rgu(df, cluster_config)

    # TODO: isn't this supposed to be fixed in SARC? Why do we need this mapping here?
    gpu_to_rgu_billing = {
        "a100-40gb": 700,
        "a100-40gb-3g.20gb": 1714.29 / 4000 * 700,
        "a100-40gb-4g.20gb": 2285.71 / 4000 * 700,
    }
    col_ratio_rgu_by_gpu = df.loc[slice_during_rgu_time, "allocated.gpu_type"].map(
        gpu_to_rgu_billing
    )
    df.loc[slice_during_rgu_time, "allocated.gpu_type_rgu"] = col_ratio_rgu_by_gpu
    # todo: warning about type non-compatible with int64.
    df.loc[slice_during_rgu_time, "allocated.gres_gpu"] = (
        non_updated_df["allocated.gres_gpu"] / col_ratio_rgu_by_gpu
    )

    # TODO: Apply only during this period

    # narval_rgu['mappings'] = {"a100-40gb": 700, "a100-40gb-3g.20gb": 1714.29/4000*700, "a100-40gb-4g.20gb": 2285.71/4000*700}
    # narval_config.rgu_start_date = "2024-04-01"
    # with open(narval_config.gpu_to_rgu_billing, 'w', encoding='utf-8') as file:
    #     json.dump(narval_rgu, file)
    # df = update_cluster_job_series_rgu(df, narval_config)

    # narval_rgu['mappings'] = previous_mappings
    # with open(narval_config.gpu_to_rgu_billing, 'w', encoding='utf-8') as file:
    #     json.dump(narval_rgu, file)
    # End of hacky fix

    # Overwrite all RGU values.
    df["allocated.gpu_type_rgu"] = df["allocated.gpu_type"].map(_RGUS)


def _set_cpu_gpu_billed(stats: pd.DataFrame):
    assert (
        stats["allocated.cpu"].notna().all()
        and (stats["allocated.cpu"] > 0).all()
        # todo: some jobs have 0.001 cpus (because of the /1000).
        and (stats["allocated.cpu"] < 1000).all()
    )
    assert (
        stats["allocated.gres_gpu"].notna().all()
        and (stats["allocated.gres_gpu"] >= 0).all()
    )
    return stats.assign(
        **{
            "cpu_billed": stats["elapsed_time"] * stats["allocated.cpu"],
            "gpu_billed": stats["elapsed_time"] * stats["allocated.gres_gpu"],
        }
    )


def compute_time_frames(
    jobs: pd.DataFrame,
    columns: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    start_column: str = "start_time",
    end_column: str = "end_time",
    frame_size: str | timedelta = "MS",
    # frame_size: timedelta = timedelta(days=7),
    callback: None | Callable = None,
):
    """Slice jobs into time frames and adjust columns to fit the time frames.

    Jobs that start before `start` or ends after `end` will have their running
    time clipped to fitting within the interval (`start`, `end`).

    Jobs spanning multiple time frames will have their running time sliced
    according to the time frames.

    The resulting DataFrame will have the additional columns 'elapsed_time' and 'timestamp'
    which represent the elapsed_time of a job within a time frame and the start of the time frame.

    Parameters
    ----------
    jobs: pandas.DataFrame
        DataFrame containing jobs data. Typically generated with `load_job_series`.
        Must contain columns `start` and `end`.
    columns: list of str
        Columns to adjust based on time frames.
    start: datetime, optional
        Start of the time frame. If None, use the first job start time.
    end: datetime, optional
        End of the time frame. If None, use the last job end time.
    frame_size: timedelta, optional
        Size of the time frames used to compute histograms. Default to 7 days.

    Examples
    --------
    >>> data = pd.DataFrame(
        [
            [datetime(2023, 3, 5), datetime(2023, 3, 6), "a", "A", 10],
            [datetime(2023, 3, 6), datetime(2023, 3, 9), "a", "B", 10],
            [datetime(2023, 3, 6), datetime(2023, 3, 7), "b", "B", 20],
            [datetime(2023, 3, 6), datetime(2023, 3, 8), "b", "B", 20],
        ],
        columns=["start_time", "end_time", "user", "cluster", 'cost'],
    )
    >>> compute_time_frames(data, columns=['cost'], frame_size=timedelta(days=2))
           start        end user cluster       cost  elapsed_time  timestamp
    0 2023-03-05 2023-03-06    a       A  10.000000   86400.0 2023-03-05
    1 2023-03-06 2023-03-09    a       B   3.333333   86400.0 2023-03-05
    2 2023-03-06 2023-03-07    b       B  20.000000   86400.0 2023-03-05
    3 2023-03-06 2023-03-08    b       B  10.000000   86400.0 2023-03-05
    1 2023-03-06 2023-03-09    a       B   6.666667  172800.0 2023-03-07
    3 2023-03-06 2023-03-08    b       B  10.000000   86400.0 2023-03-07
    """
    if columns is None:
        columns = []

    if start is None:
        start = jobs[start_column].min()

    if end is None:
        end = jobs[end_column].max()

    data_frames = []

    total_elapsed_times = (jobs[end_column] - jobs[start_column]).dt.total_seconds()

    jobs = jobs.copy()
    for time_column in [start_column, end_column]:
        jobs[time_column] = (
            jobs[time_column].dt.tz_localize(None).astype("datetime64[ns]")
        )

    timestamps = pd.date_range(
        start, end, freq=frame_size, inclusive="both"
    ).tz_localize(None)
    # for frame_start in pd.date_range(start, end, freq=f"MS"):
    for frame_start, frame_end in zip(timestamps, timestamps[1:]):
        mask = (jobs[start_column] < frame_end) & (jobs[end_column] > frame_start)
        frame = jobs[mask].copy()
        total_elapsed_times_in_frame = total_elapsed_times[mask]
        frame["elapsed_time"] = (
            frame[end_column].clip(frame_start, frame_end)
            - frame[start_column].clip(frame_start, frame_end)
        ).dt.total_seconds()

        # Adjust columns to fit the time frame.
        for column in columns:
            frame[column] *= frame["elapsed_time"] / total_elapsed_times_in_frame

        frame["timestamp"] = frame_start

        if callback:
            callback(frame, frame_start, frame_end)

        data_frames.append(frame)

    return pd.concat(data_frames, axis=0)


if __name__ == "__main__":
    main()
