import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import rich
import rich.pretty

from examples.waste_monitor.sarc_patches import (
    clean_sarc_data_v1,
    clean_sarc_data_v2,
    get_raw_sarc_data,
)
from sarc.client.series import update_job_series_rgu
from sarc.config import MTL

from .common_utils import FilteringOptions, cache_results_to_file, midnight


@pytest.fixture(scope="module")
def cluster(request: pytest.FixtureRequest) -> str | Sequence[str] | None:
    return getattr(request, "param", None)


@pytest.fixture(scope="module")
def period(cluster: str | Sequence[str] | None):
    today_eod = midnight(datetime.datetime.now(tz=MTL)) + datetime.timedelta(days=1)
    return FilteringOptions(
        start=today_eod - datetime.timedelta(days=7),
        end=today_eod,
        clusters=[cluster] if isinstance(cluster, str) else cluster or [],
    )


@pytest.fixture(scope="module")
def sarc_data(period: FilteringOptions):
    data = cache_results_to_file(get_raw_sarc_data)(period)
    data = update_job_series_rgu(data)
    return data


@pytest.fixture(scope="module")
def sarc_data_with_rgus(load_job_series_data: pd.DataFrame):
    return update_job_series_rgu(load_job_series_data.copy())


@pytest.fixture(scope="module")
def cleaned_sarc_data_v1(period: FilteringOptions):
    data = cache_results_to_file(get_raw_sarc_data)(period)
    return clean_sarc_data_v1(data, period)


@pytest.fixture(scope="module")
def cleaned_sarc_data_v2(period: FilteringOptions):
    data = cache_results_to_file(get_raw_sarc_data)(period)
    return clean_sarc_data_v2(data)


def show_first_entry(df: pd.DataFrame):
    return rich.pretty.pretty_repr(df.to_dict(orient="records")[0])


class BaseTests:
    data = ...

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
        weird_gap = (data["submit_time"] - data["start_time"]).dt.days > 30
        assert (t := data[weird_gap]).empty, show_first_entry(t)
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
                "(elapsed_time.dt.seconds > 5) & gpu_utilization.isna()"
            )
        ).empty, show_first_entry(t)

    def test_each_gpu_has_same_rgu(self, gpu_jobs: pd.DataFrame):
        """Test that each GPU type maps to a single RGU type."""
        gpu_and_rgu = gpu_jobs[
            ["allocated.gpu_type", "allocated.gpu_type_rgu"]
        ].drop_duplicates()
        if gpu_jobs["rgu_equivalent_cost"].dtype != np.dtype("timedelta64[ns]"):
            gpu_jobs = gpu_jobs.assign(
                rgu_equivalent_cost=pd.to_timedelta(
                    gpu_jobs["rgu_equivalent_cost"], unit="s"
                )
            )

        mig_gpu_jobs = gpu_jobs[
            gpu_jobs["allocated.gpu_type"].isin(
                [
                    "A100-SXM4-80GB : 2g.20gb",
                    "A100-SXM4-80GB : 4g.40gb",
                    "A100-SXM4-80GB : 3g.40gb",
                ]
            )
        ]

        fraction_of_compute = (
            mig_gpu_jobs["rgu_equivalent_cost"].dt.days.sum()
            / gpu_jobs["rgu_equivalent_cost"].dt.days.sum()
        )
        assert False, fraction_of_compute
        mapping = (
            gpu_and_rgu.groupby("allocated.gpu_type")["allocated.gpu_type_rgu"]
            .nunique()
            .reset_index()
        )
        assert False, mapping
        assert mapping["allocated.gpu_type_rgu"].max() == 1, (
            "Some GPU types map to multiple RGU values!",
            mapping,
        )

    def test_rgu_type_rgu_is_set(self, gpu_jobs: pd.DataFrame):
        assert (t := gpu_jobs.query("`allocated.gpu_type_rgu`.isna()")).empty, (
            show_first_entry(t)
        )

    def test_allocated_gres_rgu_is_positive(self, gpu_jobs: pd.DataFrame):
        assert (t := gpu_jobs.query("`allocated.gres_rgu` <= 0")).empty, (
            show_first_entry(t)
        )


@pytest.mark.xfail(strict=False, reason="sarc data is dirty")
class TestSarcData(BaseTests):
    """Test the loading of job series data."""

    data = staticmethod(sarc_data)


class TestCleanedSarcDataV1(BaseTests):
    """Test the loading of job series data."""

    data = staticmethod(cleaned_sarc_data_v1)

    # IDEA: Add a scrict xfail to some tests to ensure they fail if this job is in the data.
    # Also, could run the tests twice, before and after removing these known problematic jobs from the data.
    known_failures = [{"cluster_name": "mila", "job_id": 16}]


class TestCleanedSarcDataV2(BaseTests):
    """Test the loading of job series data."""

    data = staticmethod(cleaned_sarc_data_v2)

    # IDEA: Add a scrict xfail to some tests to ensure they fail if this job is in the data.
    # Also, could run the tests twice, before and after removing these known problematic jobs from the data.
    known_failures = [{"cluster_name": "mila", "job_id": 16}]
