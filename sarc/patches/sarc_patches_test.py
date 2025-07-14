import datetime
from typing import Sequence

import pandas as pd
import pytest
import rich
import rich.pretty

from sarc.client.series import load_job_series, update_job_series_rgu
from sarc.config import MTL

from .domain import Domain, PartialDomain
from .sarc_patches import (
    clean_sarc_data,
    get_raw_sarc_data,
)


def midnight(dt: datetime.datetime) -> datetime.datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


@pytest.fixture(scope="module")
def cluster(request: pytest.FixtureRequest) -> str | Sequence[str] | None:
    return getattr(request, "param", None)


@pytest.fixture(scope="module")
def period(cluster: str | Sequence[str] | None):
    today_eod = midnight(datetime.datetime.now()) + datetime.timedelta(days=1)
    return PartialDomain(
        start_time=today_eod - datetime.timedelta(days=7),
        end_time=today_eod,
        clusters=(
            {cluster}
            if isinstance(cluster, str)
            else set(cluster)
            if cluster
            else set()
        ),
    )


@pytest.fixture(scope="module")
def load_job_series_data(period: FilteringOptions):
    return cache_results_to_file(get_raw_sarc_data)(period)


@pytest.fixture(scope="module")
def sarc_data_with_rgus(load_job_series_data: pd.DataFrame):
    return update_job_series_rgu(load_job_series_data.copy())


@pytest.fixture(scope="module")
def cleaned_sarc_data(load_job_series_data: pd.DataFrame, period: FilteringOptions):
    return clean_sarc_data(load_job_series_data.copy(), period)


def show_first_entry(df: pd.DataFrame):
    return rich.pretty.pretty_repr(df.to_dict(orient="records")[0])


class BaseTests:
    data: ...  # to be overridden by subclasses

    def test_data_fits_period(self, data: pd.DataFrame, period: FilteringOptions):
        """Test that the data fits the period."""
        assert not data.empty, "Data is empty"
        assert data["start_time"].dt.tz_convert(MTL).max() <= period.end.astimezone(
            MTL
        ), "Some returned jobs start after the period ends!"
        assert data["end_time"].dt.tz_convert(MTL).min() >= period.start.astimezone(
            MTL
        ), "Some returned jobs end before the start data in the query!"

    def test_no_lost_jobs(self, data: pd.DataFrame, period: FilteringOptions):
        """Test that there are no weird lost jobs that lasted years."""
        if data.dtypes["elapsed_time"] != "timedelta64[ns]":
            data = data.assign(
                elapsed_time=pd.to_timedelta(data["elapsed_time"], unit="s")
            )
        assert (t := data.query("elapsed_time.dt.days > 28")).empty, show_first_entry(t)

    @pytest.fixture(scope="class")
    def gpu_jobs(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.query("`requested.gres_gpu` > 0")

    def test_allocated_gpu_type_is_set(self, gpu_jobs: pd.DataFrame):
        """Test that the GPU utilization is not NaN for jobs that use a GPU."""
        assert (t := gpu_jobs.query("`allocated.gpu_type`.isna()")).empty, (
            show_first_entry(t)
        )

    def test_allocated_gpu_is_not_zero_when_requested_gpu_is_positive(
        self, gpu_jobs: pd.DataFrame
    ):
        """Test that the allocated GPU is not zero when requested GPU is positive."""
        # This is a common issue where jobs request GPUs but do not get allocated any.
        assert (
            t := gpu_jobs.query("`requested.gres_gpu` > 0 & `allocated.gres_gpu` == 0")
        ).empty, show_first_entry(t)

    @pytest.mark.parametrize(
        "cluster",
        [
            # None,  # all clusters
            "mila",
            "cedar",
        ],
        indirect=True,
    )
    def test_gpu_util_isnt_nan(self, gpu_jobs: pd.DataFrame):
        """Test that the GPU utilization is not NaN for jobs that use a GPU."""
        if gpu_jobs.dtypes["elapsed_time"] != "timedelta64[ns]":
            gpu_jobs = gpu_jobs.assign(
                elapsed_time=pd.to_timedelta(gpu_jobs["elapsed_time"], unit="s")
            )
        assert (
            t := gpu_jobs.query(
                f"(elapsed_time.dt.seconds > 5) & gpu_utilization.isna()"
            )
        ).empty, show_first_entry(t)


class TestLoadJobSeries(BaseTests):
    """Test the loading of job series data."""

    data = staticmethod(load_job_series_data)

    # IDEA: Add a scrict xfail to some tests to ensure they fail if this job is in the data.
    # Also, could run the tests twice, before and after removing these known problematic jobs from the data.
    known_failures = [{"cluster_name": "mila", "job_id": 16}]

    @pytest.mark.xfail(
        reason="This currently fails! For example, job id 16 on the Mila cluster.",
        strict=True,
    )
    def test_allocated_gpu_type_is_set(self, gpu_jobs: pd.DataFrame):
        super().test_allocated_gpu_type_is_set(gpu_jobs)

    @pytest.mark.xfail(
        reason="This currently fails! For example, job id 16 on the Mila cluster.",
        strict=True,
    )
    def test_no_lost_jobs(self, data: pd.DataFrame, period: FilteringOptions):
        """Test that there are no weird lost jobs that lasted years."""
        super().test_no_lost_jobs(data, period)

    @pytest.mark.xfail(
        reason="This currently fails! For example, job id 16 on the Mila cluster.",
        strict=True,
    )
    def test_gpu_util_isnt_nan(self, gpu_jobs: pd.DataFrame):
        super().test_gpu_util_isnt_nan(gpu_jobs)


class TestUpdateJobSeriesRgu(TestLoadJobSeries):
    """Test the loading of job series data."""

    data = staticmethod(sarc_data_with_rgus)

    def test_rgu_type_rgu_is_set(self, gpu_jobs: pd.DataFrame):
        assert (t := gpu_jobs.query("gpu_type_rgu.isna()")).empty, show_first_entry(t)

    def test_allocated_gres_rgu_is_positive(self, gpu_jobs: pd.DataFrame):
        assert (t := gpu_jobs.query("`allocated.gres_rgu` <= 0")).empty, (
            show_first_entry(t)
        )


class TestPatchedSarcData(BaseTests):
    """Test the loading of job series data."""

    data = staticmethod(cleaned_sarc_data)
