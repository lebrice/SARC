import random

import numpy as np
import pandas as pd

from examples.compute_forecast import _print_like_form_shows, get_group_usage


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


def test_get_group_usage():
    # Check that the actual `get_group_usage` function gives a
    # dataframe with the same columns, datatypes, etc as the fake one above.
    prof_email = "glen.berseth@mila.quebec"
    actual_df = get_group_usage(prof_email)
    print("Actual:")
    _print_like_form_shows(actual_df)
    fake_df = fake_get_group_usage(prof_email)
    print("Fake:")
    _print_like_form_shows(fake_df)
    assert actual_df.shape == fake_df.shape
    assert all(actual_df.columns == fake_df.columns)
    assert all(actual_df.dtypes == fake_df.dtypes)
