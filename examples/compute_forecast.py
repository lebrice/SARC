import argparse
import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, ParamSpec, TypeVar

import numpy as np
import pandas as pd
import rich
import rich.logging
import simple_parsing
import yaml
from matplotlib import pyplot as plt
from typing_extensions import Self

os.environ.setdefault("SARC_CONFIG", "config/sarc-client.yaml")

from sarc.client.job import JobStatistics
from sarc.client.series import compute_cost_and_waste, load_job_series
from sarc.client.users.api import User, get_users
from sarc.config import MTL, ClusterConfig
from sarc.jobs.series import update_cluster_job_series_rgu

if (_sarc_config := os.environ.get("SARC_CONFIG")) and Path(_sarc_config).exists():
    CONFIG_FOLDER = Path(_sarc_config).parent
else:
    CONFIG_FOLDER = Path(__file__).parent / "config"

logger = logging.getLogger(__name__)

T = TypeVar("T", float, timedelta)
P = ParamSpec("P")
OutT = TypeVar("OutT")

pd.options.display.max_colwidth = 300
pd.options.display.max_rows = 1000
pd.options.display.float_format = lambda x: f"{x:.3f}"

ALL_CLUSTERS = ["mila", "narval", "beluga", "cedar", "graham"]
CACHE_DIR: Path | None = (
    Path(os.environ["CF_DATA"]) if "CF_DATA" in os.environ else None
)
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

_PROFS = [
    "aishwarya.agrawal@mila.quebec",
    "blake.richards@mila.quebec",
    "christopher.pal@mila.quebec",
    "gidelgau@mila.quebec",
    "glen.berseth@mila.quebec",
    "pierre-luc.bacon@mila.quebec",
    # Big drop in 2024 compared to 2022 and 2023.
    # Observations:
    # - Pierluca,Tianwei,Evgenii were biggest compute users of that group in 2022 (73,58,51 rgu*years)
    # - Tianwei, Simon, Sobhanless top users in 2023 (90,31,23) rgu*years.
    # Possible explanations:
    # - Some students transitioned away from Mila/DRAC clusters and towards using corporate clusters?
    "rabussgu@mila.quebec",
    "siva.reddy@mila.quebec",
    "hernanga@mila.quebec",
    "alex.hernandez-garcia@mila.quebec",  # Missing student mapping in users db
    "tegan.maharaj@mila.quebec",  # Missing student mapping in users db
    "arbeltal@mila.quebec",
    "cheungja@mila.quebec",
    "drolnick@mila.quebec",
    "guillaume.lajoie@mila.quebec",
    "lcharlin@mila.quebec",
    "moonajung@mila.quebec",  # no data in SARC!
    "prakash.panangaden@mila.quebec",
    "reihaneh.rabbany@mila.quebec",
    "david.adelani@mila.quebec",
    "kruegerd@mila.quebec",
    "bzdokdan@mila.quebec",
    "courvila@mila.quebec",
    "dhanya.sridhar@mila.quebec",
    "farnadig@mila.quebec",
    "odonnelt@mila.quebec",
    "paulll@mila.quebec",
    "siamak.ravanbakhsh@mila.quebec",
    "slacoste@mila.quebec",
    "farahmand@mila.quebec",
    "matt.kusner@gmail.com",
    "derek@mila.quebec",
    "ioannis@mila.quebec",
    "irina.rish@mila.quebec",
    "jpineau@mila.quebec",
    "precupdo@mila.quebec",
    "sarath.chandar@mila.quebec",
    "tangjian@mila.quebec",
    "wolfguy@mila.quebec",
    "yoshua.bengio@mila.quebec",
    "kirill.neklyudov@mila.quebec",
]
_PROFS = sorted(_PROFS)


def _midnight(dt: datetime) -> datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


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
        default=(
            Path(os.environ["CF_DATA"])
            if "CF_DATA" in os.environ
            else Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
        ),
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
        return sorted(user_emails)

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
        Options,
        default=Options(
            user=_PROFS,  # query for all profs by default.
            start=datetime(2022, 1, 1),
            end=datetime(2025, 1, 1),
            verbose=0,
        ),
    )
    _setup_logging(verbose=options.verbose)

    global CACHE_DIR
    CACHE_DIR = options.cache_dir

    profs = options.get_users()  # also supports using a file for the prof emails.
    start = options.start  # datetime(2022, 1, 1)
    end = options.end  # datetime(2025, 1, 1)

    # Uncomment to download all SARC data for that period only once, and filter it after.
    if set(profs) == set(_PROFS):
        _all_users_option = dataclasses.replace(options, user=[])
        if not (
            CACHE_DIR / _get_cache_file_name(_get_cleaned_df, options=_all_users_option)
        ).exists():
            # We did not previously load all data from SARC. Do it now to make the rest of the code faster.
            cached(_get_cleaned_df)(options=_all_users_option)

    all_profs_dataframes: dict[str, pd.DataFrame] = {}
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    redirect_output = options.verbose == 0  # when -v is passed, display output instead.
    for prof in profs:
        output_file = output_dir / f"{prof}.txt"
        with (
            open(output_file, "w") as f,
            (
                contextlib.redirect_stdout(f)
                if redirect_output
                else contextlib.nullcontext()
            ),
        ):
            # Always cached. No need to wrap.
            students = get_group_students(prof_email=prof, start=start, end=end)
            _student_emails = get_group_students_emails(
                prof_email=prof, start=start, end=end
            )
            print(f"Students supervised by {prof}: {[s.name for s in students]}")
            if not students:
                logger.error(f"Prof {prof} has no students in SARC! Skipping.")
                continue
            # Always cached. No need to wrap.
            group_usage_per_student = get_group_usage_by_student(
                prof_email=prof, start=start, end=end
            )
            if group_usage_per_student.empty:
                logger.error(
                    f"There is no data in SARC for job from any of Prof {prof}'s students! Skipping."
                )
                continue
            print(f"Compute usage in {prof}'s group:")
            k = 5
            for year in sorted(group_usage_per_student["year"].unique()):
                mask = group_usage_per_student["year"] == year
                print(
                    f"{k} students that used the most compute in {prof}'s group in {year}:"
                )
                print(group_usage_per_student[mask].nlargest(5, "gpu_years"))
            # Also always cached.
            group_usage = get_group_usage(prof_email=prof, start=start, end=end)
            all_profs_dataframes[prof] = group_usage
            # NOTE: Dataframe arguments are ignored by the `cached` wrapper.
            # Here this function is not cached by default, and the `cached` wrapper is only added
            # here instead, because we use this function below with only the `group_usage` argument
            # (not passing `prof_email`).
            usage_projections = get_group_usage_projections(
                prof_email=prof,
                usage_start=datetime(2022, 1, 1),
                usage_end=datetime(2025, 1, 1),
                projection_end=datetime(2027, 1, 1),
            )
            _print_like_form_shows(pd.concat([group_usage, usage_projections]))

    if len(profs) == 1:
        return
    all_profs_data = pd.concat(
        {k: v.set_index("year") for k, v in all_profs_dataframes.items()},
        names=["prof", "year"],
    )

    # todo: make some nice plots!
    # all_profs_data[["gpu_years", "cpu_years"]].plot(
    #     x="year", kind="bar", figsize=(12, 6)
    # )
    # plt.show()

    total_profs_data = all_profs_data.groupby(level="year").sum().reset_index()
    # Uncached, because we pass the dataframe as the argument.
    usage_projections = _get_group_usage_projections(group_usage=total_profs_data)
    print(f"Total for {len(profs)} profs:")
    _print_like_form_shows(pd.concat([total_profs_data, usage_projections]))

    plot_usage_projections(total_profs_data, usage_projections)


def plot_usage_projections(
    total_profs_data: pd.DataFrame, usage_projections: pd.DataFrame
):
    df = pd.concat([total_profs_data, usage_projections])
    new_rows = df.iloc[:2] * np.nan
    new_rows["year"] = [2020, 2021]
    df = pd.concat([new_rows, df], ignore_index=True)
    df["available"] = [
        np.mean(y)
        for y in [
            [709, 1352, 1155],
            [1487, 1651, 2000],
            [2300, 2702, 3201],
            [3235, 3199, 3199],
            [3263, 3113, 6884],
            [7740, 10444, 11223],
            [16585, 16585, 16858],
        ]
    ]

    df[["year", "gpu_years", "available"]].plot(x="year", kind="bar", figsize=(12, 6))
    plt.savefig("outputs/usage_projections.png")


def cached(fn: Callable[P, OutT]) -> Callable[P, OutT]:
    """Caches a function in a given cache dir."""
    if CACHE_DIR is not None:
        cache_dir = CACHE_DIR
    else:
        parser = argparse.ArgumentParser(add_help=False)
        default_cache_dir = Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
        parser.add_argument("--cache_dir", type=Path, default=default_cache_dir)
        cache_dir: Path = parser.parse_known_args()[0].cache_dir

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> OutT:
        """Decorator to cache the results of a function."""
        # TODO: in _get_cache_file_name, use a .txt extension if the return annotation is `str`
        cache_file = cache_dir / _get_cache_file_name(fn, *args, **kwargs)
        if cache_file.exists():
            logger.info(f"Loading result of {fn.__name__} from {cache_file}")
            return pickle.loads(cache_file.read_bytes())
        else:
            logger.debug(f"Cache miss for {fn.__name__} at {cache_file}")
            result = fn(*args, **kwargs)
            # TODO: Save to a text file if the result is a string.
            cache_file.write_bytes(pickle.dumps(result))
            logger.info(f"Saved result of computing {fn.__name__} to {cache_file}")
            return result

    return wrapper


def _get_cache_file_name(
    fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
) -> str:
    # More interpretable than using this:
    # return hashlib.md5(
    #     json.dumps((fn.__name__, args, kwargs), sort_keys=True, default=str).encode()
    # ).hexdigest()

    def _hash(v) -> str:
        if isinstance(v, pd.DataFrame):
            # Important: Assuming that the other function arguments will be used
            # to recover the same dataframe, so not including it in the hash.
            return ""
        if v is None:
            return str(v)
        if isinstance(v, str):
            return v.removesuffix("@mila.quebec")  # no quotes around strings.
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, datetime):
            if v.hour == 0 and v.minute == 0 and v.second == 0:
                return v.strftime("%Y-%m-%d-%z")
            return v.strftime("%Y-%m-%dT%H:%M:%S%z")
        if isinstance(v, list):
            # Some profs have so many students that we can't concat them.
            if v and isinstance(v[0], User):
                return hashlib.md5(
                    "+".join(sorted(student.mila.username for student in v)).encode()
                ).hexdigest()[:12]
            return "+".join(sorted(map(_hash, v)))
        if isinstance(v, User):
            return v.mila.username
        if isinstance(v, Options):
            return (
                v.unique_path()
                .relative_to(v.cache_dir)
                .stem.removeprefix("compute_profile-")
            )
        raise NotImplementedError(f"Unsupported arg type: {v} of type {type(v)}")

    hashed_args = "-".join(map(_hash, args)) + "-".join(
        f"{k}-{_hash(v)}" for k, v in kwargs.items()
    )
    return f"{fn.__name__}-{hashed_args}.pkl"


_hard_coded_values: dict[str, list[str]] = {
    "alex.hernandez-garcia@mila.quebec": [
        "celine.roget@mila.quebec",
        "dounia.shaaban-kabakibo@mila.quebec",
        "hyeonah.kim@mila.quebec",
        "om.patel@mila.quebec",
    ]
}


@cached
def get_group_students_emails(
    prof_email: str,
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2025, 1, 1),
) -> list[str]:
    """Get list of student emails supervised by a professor."""
    students = get_group_students(prof_email, start=start, end=end)
    return [student.mila.email for student in students]


@cached
def get_group_students(
    prof_email: str,
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2025, 1, 1),
) -> list[User]:
    """Get list of student emails supervised by a professor."""
    all_users = get_users()
    students = [
        user
        for user in all_users
        if user.mila_ldap.get("supervisor") == prof_email
        or user.mila_ldap.get("co_supervisor") == prof_email
    ]
    students = sorted(students, key=lambda v: v.name)
    # NOTE: In case `students` is empty (no mapping in SARC), we couldn't use the jobs data
    # from SARC to find the supervisor, since the supervisor and co-supervisor
    # fields in the jobs are set using the same data source!
    if not students and prof_email in _hard_coded_values:
        logger.warning(
            f"Unable to get the students supervised by {prof_email} from SARC, using hard-coded values instead. "
        )
        student_emails = _hard_coded_values[prof_email]
        students = [
            student for student in all_users if student.mila.email in student_emails
        ]
    return students


@cached
def get_group_usage(
    prof_email: str,
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2025, 1, 1),
) -> pd.DataFrame:
    """Returns the total compute usage for a prof's group in the given period."""
    students = get_group_students(prof_email, start=start, end=end)
    logger.info(f"{prof_email} has apparently {len(students)} students.")
    if not students:
        logger.warning(f"No students found for {prof_email}. Returning zeros.")
        years = list(range(start.year, end.year))
        return pd.DataFrame(
            {
                "year": years,
                "students": np.zeros(len(years)),
                "gpu_years": np.zeros(len(years)),
                "gpu_mem_mean": np.zeros(len(years)),
                "gpu_mem_max": np.zeros(len(years)),
                "gpu_util_mean": np.zeros(len(years)),
                "gpu_cpu_years": np.zeros(len(years)),
                "gpu_cpu_mem_mean": np.zeros(len(years)),
                "gpu_cpu_mem_max": np.zeros(len(years)),
                "cpu_years": np.zeros(len(years)),
                "cpu_mem_mean": np.zeros(len(years)),
                "cpu_mem_max": np.zeros(len(years)),
            }
        )

    options = Options(
        start=start.astimezone(MTL),
        end=end.astimezone(MTL),
        user=[s.mila.email for s in students],
    )
    sarc_data = cached(_get_cleaned_df)(options)
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
    assert (gpu_job_stats["allocated.gres_gpu"] >= 0).all()
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
            gpu_job_stats["system_memory"]
            * (gpu_job_stats["allocated.mem"] // 1024)
        ),
    )
    gpu_job_stats = gpu_job_stats.assign(
        cpu_cost_per_gpu=(
            gpu_job_stats["cpu_equivalent_cost"] / gpu_job_stats["allocated.gres_gpu"]
        ),
        cpu_mem_gb_per_gpu=(
            gpu_job_stats["cpu_mem_gb"] / gpu_job_stats["allocated.gres_gpu"]
        ),
    )
    cpu_job_stats = cpu_job_stats.assign(
        cpu_mem_gb=(
            cpu_job_stats["system_memory"] * (cpu_job_stats["allocated.mem"] // 1024)
        ),
    )
    grouped_gpu_stats = gpu_job_stats.groupby(["timestamp"])
    gpu_sum_metrics_years = (
        grouped_gpu_stats[["rgu_equivalent_cost", "cpu_cost_per_gpu"]].sum()
        / seconds_in_a_year
    )
    gpu_mean_stats = grouped_gpu_stats[
        ["gpu_utilization", "gpu_mem_gb", "cpu_mem_gb_per_gpu"]
    ].mean()
    gpu_max_stats = grouped_gpu_stats[["gpu_mem_gb", "cpu_mem_gb_per_gpu"]].max()

    grouped_cpu_stats = cpu_job_stats.groupby(["timestamp"])
    cpu_sum_metrics_years = (
        grouped_cpu_stats[["cpu_equivalent_cost"]].sum() / seconds_in_a_year
    )
    cpu_mean_stats = grouped_cpu_stats[["cpu_mem_gb"]].mean()
    cpu_max_stats = grouped_cpu_stats[["cpu_mem_gb"]].max()

    n_students_per_year = usage_stats.groupby(["timestamp"])[
        "user.primary_email"
    ].nunique()
    logger.info(f"Number of students with slurm jobs per year: {n_students_per_year}")

    years = sorted(usage_stats["timestamp"].dt.year.unique().astype(int))

    data = {
        "year": years,
        "students": n_students_per_year,
        "gpu_years": gpu_sum_metrics_years["rgu_equivalent_cost"],
        "gpu_mem_mean": gpu_mean_stats["gpu_mem_gb"],
        "gpu_mem_max": gpu_max_stats["gpu_mem_gb"],
        "gpu_util_mean": gpu_mean_stats["gpu_utilization"],
        "gpu_cpu_years": gpu_sum_metrics_years["cpu_cost_per_gpu"],
        "gpu_cpu_mem_mean": gpu_mean_stats["cpu_mem_gb_per_gpu"],
        "gpu_cpu_mem_max": gpu_max_stats["cpu_mem_gb_per_gpu"],
        "cpu_years": cpu_sum_metrics_years["cpu_equivalent_cost"],
        "cpu_mem_mean": cpu_mean_stats["cpu_mem_gb"],
        "cpu_mem_max": cpu_max_stats["cpu_mem_gb"],
    }
    data = pd.DataFrame(data)
    data = data.astype({"year": int})
    data = data.set_index("year").sort_index()
    data = data.reindex(range(start.year, end.year), fill_value=np.nan)
    data = (
        data.reset_index()
    )  # don't actually want `year` as the index (sticking to Xavier's interface)
    # Change the `year` column to have int dtype:
    return data


@cached
def get_group_usage_by_student(
    prof_email: str,
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2025, 1, 1),
) -> pd.DataFrame:
    """Returns the total compute usage for a prof's group in the given period."""
    students_emails = get_group_students_emails(prof_email, start=start, end=end)
    logger.info(f"{prof_email} has apparently {len(students_emails)} students.")
    start = start.astimezone(MTL)
    end = end.astimezone(MTL)
    options = Options(start=start, end=end, user=students_emails)
    sarc_data = cached(_get_cleaned_df)(options)
    if sarc_data.empty:
        logger.warning(
            f"No data found in SARC for {prof_email} from {start} to {end}. Returning empty dataframe."
        )
        return pd.DataFrame(
            columns=[
                "user",
                "year",
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
        )
    usage_stats = _get_stats(sarc_data, options, frame_size="YS")
    gpu_job_stats = usage_stats[usage_stats["requested.gres_gpu"] > 0]
    cpu_job_stats = usage_stats[usage_stats["requested.gres_gpu"] == 0]

    assert all(gpu_job_stats["allocated.gres_gpu"] > 0)

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
            gpu_job_stats["system_memory"]
            * (gpu_job_stats["allocated.mem"] // 1024)
        ),
    )
    gpu_job_stats = gpu_job_stats.assign(
        cpu_cost_per_gpu=(
            gpu_job_stats["cpu_equivalent_cost"] / gpu_job_stats["allocated.gres_gpu"]
        ),
        cpu_mem_gb_per_gpu=(
            gpu_job_stats["cpu_mem_gb"] / gpu_job_stats["allocated.gres_gpu"]
        ),
    )
    cpu_job_stats = cpu_job_stats.assign(
        cpu_mem_gb=(
            cpu_job_stats["system_memory"] * (cpu_job_stats["allocated.mem"] // 1024)
        ),
    )

    grouped_gpu_stats = gpu_job_stats.groupby(["timestamp", "user.primary_email"])
    gpu_sum_metrics_years = (
        grouped_gpu_stats[["rgu_equivalent_cost", "cpu_cost_per_gpu"]].sum()
        / seconds_in_a_year
    )
    gpu_mean_stats = grouped_gpu_stats[
        ["gpu_utilization", "gpu_mem_gb", "cpu_mem_gb_per_gpu"]
    ].mean()
    gpu_max_stats = grouped_gpu_stats[["gpu_mem_gb", "cpu_mem_gb_per_gpu"]].max()

    grouped_cpu_stats = cpu_job_stats.groupby(["timestamp", "user.primary_email"])
    cpu_sum_metrics_years = (
        grouped_cpu_stats[["cpu_equivalent_cost"]].sum() / seconds_in_a_year
    )
    cpu_mean_stats = grouped_cpu_stats[["cpu_mem_gb"]].mean()
    cpu_max_stats = grouped_cpu_stats[["cpu_mem_gb"]].max()

    values: list[dict] = []
    index: list[tuple[int, str]] = []

    timestamps = sorted(usage_stats["timestamp"].unique())
    years = sorted(usage_stats["timestamp"].dt.year.astype(int).unique())
    assert len(years) == len(timestamps)
    for year, timestamp in zip(years, timestamps):
        for student_email in students_emails:
            index.append((year, student_email))

            def _get(df: pd.DataFrame, column: str, default=np.nan) -> float:
                return df[column].get((timestamp, student_email), default)

            user_year_values = {
                "user": student_email,
                "year": year,
                "gpu_years": _get(gpu_sum_metrics_years, "rgu_equivalent_cost", 0.0),
                "gpu_mem_mean": _get(gpu_mean_stats, "gpu_mem_gb"),
                "gpu_mem_max": _get(gpu_max_stats, "gpu_mem_gb"),
                "gpu_util_mean": _get(gpu_mean_stats, "gpu_utilization"),
                "gpu_cpu_years": _get(gpu_sum_metrics_years, "cpu_cost_per_gpu", 0.0),
                "gpu_cpu_mem_mean": _get(gpu_mean_stats, "cpu_mem_gb_per_gpu"),
                "gpu_cpu_mem_max": _get(gpu_max_stats, "cpu_mem_gb_per_gpu"),
                "cpu_years": _get(cpu_sum_metrics_years, "cpu_equivalent_cost", 0.0),
                "cpu_mem_mean": _get(cpu_mean_stats, "cpu_mem_gb"),
                "cpu_mem_max": _get(cpu_max_stats, "cpu_mem_gb"),
            }
            values.append(user_year_values)
    # Could also make a multiindex, but makes it a bit harder to work with.
    # return pd.DataFrame.from_records(
    #     values, index=pd.MultiIndex.from_tuples(index, names=["year", "user"])
    # )
    df = pd.DataFrame.from_records(values)
    return df


@cached
def get_group_usage_projections(
    prof_email: str,
    usage_start: datetime = datetime(2022, 1, 1),
    usage_end: datetime = datetime(2025, 1, 1),
    projection_end: datetime = datetime(2027, 1, 1),
    use_exponential_trend: bool = True,
) -> pd.DataFrame:
    """Extrapolates the group compute usage and returns projection data for years from `start` to `end` (inclusive).

    By default assumes that the data is for years leading to 2025 and makes predictions for 2025 and 2026.
    """
    group_usage = get_group_usage(
        prof_email=prof_email, start=usage_start, end=usage_end
    )

    return _get_group_usage_projections(
        group_usage=group_usage,
        projection_end=projection_end,
        use_exponential_trend=use_exponential_trend,
    )


def _get_group_usage_projections(
    group_usage: pd.DataFrame,
    projection_end: datetime = datetime(2027, 1, 1),
    use_exponential_trend: bool = True,
) -> pd.DataFrame:

    projection_start_year = group_usage["year"].max() + 1
    assert projection_start_year < projection_end.year
    new_x = list(range(projection_start_year, projection_end.year))

    extrapolations = extrapolate(
        group_usage, new_x, use_exponential_trend=use_exponential_trend
    ).clip(lower=0)
    # Note: round students to the nearest integer? (small detail perhaps)
    extrapolations = extrapolations.astype({"year": int}).assign(
        students=extrapolations["students"].round(),
        gpu_util_mean=extrapolations["gpu_util_mean"].clip(upper=1.0),
    )
    return extrapolations


def extrapolate(
    group_usage: pd.DataFrame, new_x: list[int], use_exponential_trend: bool = True
) -> pd.DataFrame:
    # Linear extrapolation function
    x = group_usage["year"].to_numpy().astype(int)
    result: list[np.ndarray] = []
    for col in group_usage.columns:
        if col == "year":
            result.append(np.asarray(new_x))
            continue

        y = group_usage[col].to_numpy()
        if col == "students" or not use_exponential_trend:
            coeffs = np.polyfit(x, y, 1)  # Linear fit
            extrapolated_vals = np.poly1d(coeffs)(new_x)
        else:
            # Use a linear fit in a log space then exponentiate the result,
            # so that projections follow an exponential trend.
            linear_coeffs = np.polyfit(x, y, 1)  # Linear fit in log space
            linear_extrapolated_vals = np.exp(np.poly1d(linear_coeffs)(new_x))

            log_y = np.log(y)
            coeffs = np.polyfit(x, log_y, 1)  # Linear fit in log space
            exp_extrapolated_vals = np.exp(np.poly1d(coeffs)(new_x))
            if any(np.isinf(exp_extrapolated_vals) | np.isnan(exp_extrapolated_vals)):
                logger.warning(
                    f"Extrapolated values for {col} contain inf or nan values. "
                    "Using linear extrapolation instead."
                )
                extrapolated_vals = linear_extrapolated_vals
            else:
                extrapolated_vals = exp_extrapolated_vals

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


def _filter_sarc_data(
    all_sarc_data_cleaned: pd.DataFrame, filtering_options: Options
) -> pd.DataFrame:
    users = filtering_options.get_users()
    users = list(map(_check_is_email_and_lower, users))
    df = all_sarc_data_cleaned
    if users:
        df = df[df["user.mila.email"].isin(users)]
    if filtering_options.clusters:
        df = df[df["cluster_name"].isin(filtering_options.clusters)]
    df = df[df["start_time"].between(filtering_options.start, filtering_options.end)]
    return df


def _check_is_email_and_lower(v: str):
    if not v:
        return v
    if "@" not in v or v.count("@") != 1:
        raise ValueError(f"'{v}' is not a valid email address.")

    return v.lower()


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
            else (
                "MS"
                if (_period := (options.end - options.start)) > timedelta(days=90)
                else (
                    timedelta(days=7)
                    if _period > timedelta(days=30)
                    else timedelta(days=1)
                )
            )
        ),
    )
    stats = stats.assign(
        rgu_equivalent_cost=(
            stats["gpu_equivalent_cost"] * stats["allocated.gpu_type_rgu"]
        )
    )

    return stats


def _get_cleaned_df(options: Options) -> pd.DataFrame:
    """Gets "cleaned" SARC data for a given period, including *lots* of patches."""
    options = dataclasses.replace(
        options,
        start=options.start.astimezone(MTL),
        end=options.end.astimezone(MTL),
    )
    cache_file = options.unique_path()
    _user_emails = options.get_users(assume_mila_email=True)
    assert all(map(_check_is_email_and_lower, _user_emails))

    email_to_user: dict[str, User] = {
        # NOTE: assuming that all students have a mila email would be ok for now,
        # but perhaps this will be a bit more resilient.
        (user.mila.email or (user.drac.email if user.drac else "")): user
        for user in get_users()
    }

    def get_usernames(email: str) -> list[str]:
        if email not in email_to_user:
            # Note: Would be weird to get here atm, since we get the emails from the user database.
            # But if we made a query with a particular email of a researcher for example, we might
            # get here, in which case perhaps we can use the first part of the email as username?
            raise RuntimeError(f"Email '{email}' is not found in the user database!")
            username = email.partition("@")[0]
            return [username]
        user = email_to_user[email]
        if user.drac is not None:
            return [user.mila.username, user.drac.username]
        return [user.mila.username]

    logger.debug(
        f"Looking up for data between {options.start} and {options.end} for users: {_user_emails or 'all'} and clusters {options.clusters or 'all'}"
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
        if _user_emails:
            df = df[df["user.primary_email"].isin(_user_emails)]
    else:
        logger.info(
            f"Did not find previous results at {cache_file}. Fetching job data."
        )
        all_usernames_of_students = sum(map(get_usernames, _user_emails), [])
        logger.debug(f"Usernames used when querying SARC: {all_usernames_of_students}")
        # In SARC we currently can't query by user.mila.email, so we query with all
        # usernames and filter by user.mila.email after.
        df = load_job_series(
            start=options.start,
            end=options.end,
            user=(
                {"$in": all_usernames_of_students}
                if all_usernames_of_students
                else None
            ),  # support querying for multiple users.
            clip_time=False,  # True,
        )
        if _user_emails and "user.primary_email" in df.columns:
            df = df[df["user.primary_email"].isin(_user_emails)]
        logger.info(f"Saving data to {cache_file}")
        df.to_pickle(cache_file)

    if df.empty:
        return df

    for time_column in ["submit_time", "start_time", "end_time"]:
        # df[time_column] = df[time_column].dt.tz_localize("UTC").dt.tz_convert(MTL)
        df[time_column] = df[time_column].dt.tz_convert(MTL)

    if df.shape[0] == 0:
        # NO data in SARC!
        logger.warning(f"No data found in SARC for {options}.")

    _validate_gpu_ram()

    # Clusters we want to compare
    if options.clusters:
        # Filter clusters
        df = df[df["cluster_name"].isin(options.clusters)]

    df.fillna({"requested.gres_gpu": 0.0, "allocated.gres_gpu": 0.0}, inplace=True)
    df = _fix_lost_jobs(df)
    df = _fix_unaligned_cache(df, options.start, options.end)
    df = _remove_old_nodes(df)
    df = _replace_outlier_stats_with_na(df)
    df = _fix_missing_gpu_type(df)

    _fix_rgu_discrepencies_inplace(df)

    df = _fix_requested_allocated_gres_gpu(df)

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
        logger.info(f"Missing the mila email for these users: {sorted(missing_users)}")

    return df


def _fix_requested_allocated_gres_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Fix: Some jobs on Narval have requested.gres_gpu>0 but have allocated.gres_gpu=0!

    Job ids of examples: 5083814, 5083815, 5113377
    """
    # Note: This is a workaround for a bug in SARC, where some jobs have requested.gres_gpu > 0
    # but allocated.gres_gpu = 0. This is not correct, since it means that the job was not actually
    # allocated any GPUs, but it was requested.
    # So we set allocated.gres_gpu = requested.gres_gpu for those jobs.
    # mask = (df["requested.gres_gpu"] > 0) & (df["allocated.gres_gpu"] == 0)
    # Some jobs on Narval have requested.gres_gpu>0 but have allocated.gres_gpu=0!

    requested_gres_gpu = df["requested.gres_gpu"]
    allocated_gres_gpu = df["allocated.gres_gpu"]
    # If a job requested a GPU, it has to be allocated at least one GPU.
    # In general, we assume that 1 <= requested.gres_gpu <= allocated.gres_gpu
    # If 0 < requested.gres_gpu < 1, then set it to 1.0.
    # NOTE: Actually, because of MIG GPUs, we can have requested.gres_gpu < 1.0
    # so we leave it as-is.
    # requested_gres_gpu = requested_gres_gpu.where(
    #     requested_gres_gpu > 0, np.maximum(requested_gres_gpu, 1.0)
    # )

    # IDEA:
    # requested_gres_gpu = requested_gres_gpu.where(
    #     requested_gres_gpu > 0, np.maximum(requested_gres_gpu, 1.0)
    # )

    # If allocated.gres_gpu == 0 but requested.gres_gpu > 0, set it to requested.gres_gpu.
    allocated_gres_gpu = allocated_gres_gpu.where(
        (requested_gres_gpu > 0) & (allocated_gres_gpu == 0), requested_gres_gpu
    )
    return df.assign(
        **{
            # "requested.gres_gpu": requested_gres_gpu,
            "allocated.gres_gpu": allocated_gres_gpu,
        }
    )


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
            logger.debug(
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
            logger.debug(
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
    with open(f"{CONFIG_FOLDER}/node_to_gpu.json") as f:
        cluster_configs: dict[str, dict[str, str]] = json.load(f)
    return cluster_configs[cluster_name]


def _get_cluster_configs() -> dict[str, ClusterConfig]:
    with open(f"{CONFIG_FOLDER}/sarc-dev.json") as f:
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
        clusters = df["cluster_name"].unique().tolist()
    if not clusters:
        assert df.shape[0] == 0
        clusters = ALL_CLUSTERS
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
