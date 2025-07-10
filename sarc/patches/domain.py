import dataclasses
import functools
from datetime import datetime
from typing import Self

import pandas as pd


@dataclasses.dataclass(frozen=True, unsafe_hash=True, kw_only=True)
class Domain:  # noqa: PLW1641
    """Configuration options for this script."""

    job_ids: set[int]
    """ Which job ids to apply this patch to. Leave blank to apply to all jobs."""

    start_time: datetime | None
    """ Apply this patch to jobs that started after this date. Leave blank to ignore the start date."""

    end_time: datetime | None
    """ Apply this patch to jobs that ended before this date. Leave blank to ignore the end date."""

    users: set[str]
    """ Which users's jobs to apply this patch to. Leave blank to apply to all users."""
    clusters: set[str]
    """ Which clusters's jobs to apply this patch to. Leave blank to apply to all clusters."""


@dataclasses.dataclass(frozen=True, unsafe_hash=True, kw_only=True)
class PartialDomain(Domain):  # noqa: PLW1641
    """Configuration options for this script."""

    job_ids: set[int] = set()
    start_time: datetime | None = None
    end_time: datetime | None = None
    users: set[str] = set()
    clusters: set[str] = set()

    def replace_all_with_values(self, other: "ConcreteDomain"):
        """When a value is left blank in a domain, here we replace it with the actual value from the data."""
        domain_of_df = other
        return dataclasses.replace(
            domain_of_df,
            **{
                f.name: self_value
                for f in dataclasses.fields(self)
                if bool(self_value := getattr(self, f.name))
            },
        )


@functools.total_ordering
@dataclasses.dataclass(frozen=True, unsafe_hash=True, kw_only=True)
class ConcreteDomain(Domain):  # noqa: PLW1641
    """A concrete domain where all fields are set and non-empty.

    This can be compared against another domain to see if it contains it or not.
    """

    job_ids: set[int]
    start_time: datetime
    end_time: datetime
    users: set[str]
    clusters: set[str]

    def __post_init__(self):
        if not (
            self.users
            and self.clusters
            and self.job_ids
            and self.start_time
            and self.end_time
        ):
            raise ValueError("Domain should be fully defined!")

    def __eq__(self, other: object) -> bool:
        """Returns whether this filter is equal to the other."""
        if not isinstance(other, ConcreteDomain):
            return NotImplemented
        return (
            self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.job_ids == other.job_ids
            and self.users == other.users
            and self.clusters == other.clusters
        )

    def __gt__(self, other: Self) -> bool:
        """Returns whether this domain contains another."""
        assert (
            self.users
            and self.clusters
            and self.job_ids
            and self.start_time
            and self.end_time
        )
        if isinstance(other, PartialDomain):
            raise NotImplementedError(
                "Cannot compare domains that are not fully defined."
            )
        if not isinstance(other, ConcreteDomain):
            return NotImplemented

        return (
            self.job_ids.issuperset(other.job_ids)
            and self.users.issuperset(other.users)
            and self.clusters.issuperset(other.clusters)
            and self.start_time <= other.start_time
            and other.end_time <= self.end_time
        )

    @classmethod
    def from_data(cls, data: pd.DataFrame):
        """Get the domain from the data."""
        return cls(
            start_time=data["start_time"].min(),
            end_time=data["end_time"].max(),
            users=set(data["user"].unique()),
            clusters=set(data["cluster_name"].unique()),
            job_ids=set(data["job_id"].unique()),
        )
