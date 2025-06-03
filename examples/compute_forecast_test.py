import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from examples.compute_forecast import (
    _PROFS,
    _get_group_usage_projections,
    _print_like_form_shows,
    get_group_students,
    get_group_usage,
    get_group_usage_projections,
)


@pytest.fixture(scope="module", params=_PROFS)
def prof_email(request: pytest.FixtureRequest) -> str:
    """Fixture to provide a professor's email for testing."""
    return getattr(request, "param", "glen.berseth@mila.quebec")


@pytest.fixture(scope="module")
def fake_get_group_usage(prof_email: str) -> pd.DataFrame:
    """Dummy function that returns random usage data for years 2022-2024."""
    years = range(2022, 2025)
    data = {
        "year": years,
        "students": np.random.randint(5, 20, size=len(years)),
        "gpu_years": np.random.uniform(1, 10, size=len(years)),
        "gpu_mem_mean": np.random.uniform(10, 30, size=len(years)),
        "gpu_mem_max": np.random.uniform(30, 50, size=len(years)),
        "gpu_util_mean": np.random.uniform(0.3, 0.9, size=len(years)),
        "gpu_cpu_years": np.random.uniform(1, 5, size=len(years)),
        "gpu_cpu_mem_mean": np.random.uniform(5, 15, size=len(years)),
        "gpu_cpu_mem_max": np.random.uniform(15, 25, size=len(years)),
        "cpu_years": np.random.uniform(1, 8, size=len(years)),
        "cpu_mem_mean": np.random.uniform(8, 20, size=len(years)),
        "cpu_mem_max": np.random.uniform(20, 40, size=len(years)),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def fake_get_group_usage_projections(prof_email: str) -> pd.DataFrame:
    """Dummy function that returns random projection data for years 2025-2026."""
    years = range(2025, 2027)
    data = {
        "year": years,
        "students": np.random.randint(5, 20, size=len(years)),
        "gpu_years": np.random.uniform(1, 10, size=len(years)),
        "gpu_mem_mean": np.random.uniform(10, 30, size=len(years)),
        "gpu_mem_max": np.random.uniform(30, 50, size=len(years)),
        "gpu_util_mean": np.random.uniform(0.3, 0.9, size=len(years)),
        "gpu_cpu_years": np.random.uniform(1, 5, size=len(years)),
        "gpu_cpu_mem_mean": np.random.uniform(5, 15, size=len(years)),
        "gpu_cpu_mem_max": np.random.uniform(15, 25, size=len(years)),
        "cpu_years": np.random.uniform(1, 8, size=len(years)),
        "cpu_mem_mean": np.random.uniform(8, 20, size=len(years)),
        "cpu_mem_max": np.random.uniform(20, 40, size=len(years)),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def fake_get_group_students(prof_email: str) -> list[str]:
    """Get list of student emails supervised by a professor.
    For now, returns random fake emails for testing.
    """
    # Generate 3-5 random student emails
    num_students = random.randint(3, 5)
    student_emails = []
    for i in range(num_students):
        student_emails.append(f"student{i + 1}@example.com")
    return student_emails


def test_get_group_students(prof_email: str):
    students = get_group_students(prof_email)
    if not students:
        warnings.warn(UserWarning(f"No students found for {prof_email}"))
    for student in students:
        assert (
            student.mila_ldap["supervisor"] == prof_email
            or student.mila_ldap["co_supervisor"] == prof_email
        )


def test_get_group_students_unknown_prof():
    students = get_group_students("foobob_bar@mila.quebec")
    assert not students


def test_get_group_usage(prof_email: str, fake_get_group_usage: pd.DataFrame):
    # Check that the actual `get_group_usage` function gives a
    # dataframe with the same columns, datatypes, etc as the fake one above.
    print("Fake:")
    _print_like_form_shows(fake_get_group_usage)

    actual_df = get_group_usage(prof_email=prof_email)
    print("Actual:")
    _print_like_form_shows(actual_df)
    assert actual_df.shape == fake_get_group_usage.shape
    assert all(actual_df.columns == fake_get_group_usage.columns)
    assert all(
        actual_df.drop(columns=["year", "students"]).dtypes
        == fake_get_group_usage.drop(columns=["year", "students"]).dtypes
    )


def test_get_group_usage_predictions(
    prof_email: str, fake_get_group_usage_projections: pd.DataFrame
):
    # Check that the actual `get_group_usage` function gives a
    # dataframe with the same columns, datatypes, etc as the fake one above.
    fake_df = fake_get_group_usage_projections
    print("Fake:")
    _print_like_form_shows(fake_df)

    actual_df = get_group_usage_projections(prof_email)
    print("Actual:")
    _print_like_form_shows(actual_df)

    assert actual_df.shape == fake_df.shape
    assert all(actual_df.columns == fake_df.columns)
    # not true for `students`, but doesnt really matter.
    assert all(
        actual_df.drop(columns="students").dtypes
        == fake_df.drop(columns="students").dtypes
    )


def test_predictions_for_2025_with_partial_data():
    """Compares the output of `get_group_usage_projections` for 2025 vs scaled up the partial data for that year to date."""

    profs = _PROFS
    # profs = ["aishwarya.agrawal@mila.quebec", "blake.richards@mila.quebec"]
    all_profs_data = pd.concat(
        {
            prof: get_group_usage(
                prof_email=prof, start=datetime(2022, 1, 1), end=datetime(2025, 1, 1)
            ).set_index("year")
            for prof in profs
        },
        names=["prof", "year"],
    )
    total_profs_data = all_profs_data.groupby(level="year").sum().reset_index()
    all_profs_predictions = _get_group_usage_projections(total_profs_data)
    predicted_usage_2025 = all_profs_predictions.query("year == 2025")

    end = datetime(2025, 6, 1)  # first of june as cutoff date ( months)
    usage_first_half_2025 = (
        pd.concat(
            {
                prof: get_group_usage(
                    prof_email=prof, start=datetime(2025, 1, 1), end=end
                ).set_index("year")
                for prof in profs
            },
            names=["prof", "year"],
        )
        .groupby(level="year")
        .sum()
        .reset_index()
    )
    _print_like_form_shows(usage_first_half_2025)
    scaled_usage_prediction_2025 = usage_first_half_2025 * 12 / 5  # naive scaling
    # Display a comparison of the two predictions

    compared = pd.concat(
        {
            "scaled": scaled_usage_prediction_2025.set_index("year"),
            "predicted": predicted_usage_2025.set_index("year"),
        },
        names=["type", "year"],
    )
    print("Comparison of scaled usage prediction vs predicted usage for 2025:")
    print(compared.to_markdown())
    compared.to_csv("scaled_vs_predicted_2025.csv")
    # print("Predicted usage for 2025 with data from 2022-2024:")
    # _print_like_form_shows(predicted_usage_2025)
    # print("Usage in five months of 2025 * 12/5:")
    # _print_like_form_shows(scaled_usage_prediction_2025)
    # _print_like_form_shows(scaled_usage_prediction_2025)
