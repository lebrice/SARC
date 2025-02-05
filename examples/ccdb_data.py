from pathlib import Path
from typing import Literal

import yaml

AllocationName = str

ccdb_compute_usage: dict[
    AllocationName, dict[Literal["cpu", "gpu"], dict[str, float]]
] = {
    "narval_compute": {
        "cpu": {
            "April 2024": 20.19,
            "May 2024": 16.59,
            "June 2024": 21.42,
            "July 2024": 1.49,
            "August 2024": 4.25,
            "September 2024": 1.82,
            "October 2024": 14.03,
            "November 2024": 5.34,
            "December 2024": 4.54,
            "January 2025": 13.74,
            "February 2025": 0.00,
        },
    },
    "narval_gpu": {
        "cpu": {
            "April 2024": 31.61,
            "May 2024": 64.51,
            "June 2024": 45.09,
            "July 2024": 60.38,
            "August 2024": 76.23,
            "September 2024": 50.78,
            "October 2024": 51.82,
            "November 2024": 35.54,
            "December 2024": 41.48,
            "January 2025": 30.22,
            "February 2025": 9.96,
        },
        "gpu": {
            "April 2024": 5.29,
            "May 2024": 9.96,
            "June 2024": 7.14,
            "July 2024": 8.12,
            "August 2024": 8.44,
            "September 2024": 8.53,
            "October 2024": 9.03,
            "November 2024": 7.66,
            "December 2024": 9.99,
            "January 2025": 6.43,
            "February 2025": 0.89,
        },
    },
    "beluga_compute": {
        "cpu": {
            "April 2024": 13.78,
            "May 2024": 17.21,
            "June 2024": 0.04,
            "July 2024": 6.59,
            "August 2024": 10.65,
            "September 2024": 3.37,
            "October 2024": 9.07,
            "November 2024": 4.44,
            "December 2024": 2.43,
            "January 2025": 1.11,
        }
    },
    "beluga_gpu": {
        "cpu": {
            "April 2024": 41.33,
            "May 2024": 34.04,
            "June 2024": 26.94,
            "July 2024": 41.45,
            "August 2024": 30.72,
            "September 2024": 29.60,
            "October 2024": 31.96,
            "November 2024": 32.84,
            "December 2024": 14.76,
            "January 2025": 8.22,
        },
        "gpu": {
            "April 2024": 6.57,
            "May 2024": 8.82,
            "June 2024": 6.14,
            "July 2024": 7.45,
            "August 2024": 5.04,
            "September 2024": 8.60,
            "October 2024": 6.98,
            "November 2024": 8.58,
            "December 2024": 6.80,
            "January 2025": 2.75,
        },
    },
    "cedar_compute": {
        "cpu": {
            "April 2024": 7.43,
            "May 2024": 9.24,
            "June 2024": 0.00,
            "July 2024": 0.00,
            "August 2024": 28.21,
            "September 2024": 0.02,
            "October 2024": 0.05,
            "November 2024": 5.21,
            "December 2024": 0.12,
            "January 2025": 0.65,
        }
    },
    "cedar_gpu": {
        "cpu": {
            "April 2024": 10.70,
            "May 2024": 11.89,
            "June 2024": 0.98,
            "July 2024": 1.38,
            "August 2024": 13.51,
            "September 2024": 2.60,
            "October 2024": 2.13,
            "November 2024": 9.50,
            "December 2024": 11.27,
            "January 2025": 15.27,
            "February 2025": 1.34,
        },
        "gpu": {
            "April 2024": 1.98,
            "May 2024": 2.21,
            "June 2024": 0.16,
            "July 2024": 1.06,
            "August 2024": 2.24,
            "September 2024": 0.39,
            "October 2024": 0.41,
            "November 2024": 1.40,
            "December 2024": 1.01,
            "January 2025": 2.89,
            "February 2025": 0.16,
        },
    },
    "graham_compute": {
        "cpu": {
            "April 2024": 0.79,
            "May 2024": 0.86,
            "June 2024": 0.83,
            "July 2024": 0.90,
            "August 2024": 0.60,
            "September 2024": 0.99,
            "October 2024": 1.29,
            "November 2024": 0.92,
            "December 2024": 0.11,
        }
    },
    "graham_gpu": {
        "cpu": {
            "April 2024": 0.29,
            "May 2024": 0.69,
            "June 2024": 0.01,
            "August 2024": 0.01,
            "September 2024": 0.00,
        },
        "gpu": {
            "April 2024": 0.07,
            "May 2024": 0.14,
            "June 2024": 0.00,
            "August 2024": 0.00,
            "September 2024": 0.00,
        },
    },
}
"""Usage data manually scrapped from the CCDB website."""


with Path(__file__).with_suffix(".yaml").open("w") as f:
    f.write("# Usage data manually scrapped from the CCDB website.\n")
    yaml.dump(ccdb_compute_usage, f, indent=2)
