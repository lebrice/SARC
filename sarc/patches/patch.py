"""A Patch to be applied to the data that is returned from `load_job_series` + `update_job_series_rgu`.

A Patch can specify a domain of the data that it applies to, and a set of
functions that will be applied to the data.
"""

import abc
import dataclasses

import pandas as pd

from sarc.patches.domain import ConcreteDomain, PartialDomain


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class Patch(abc.ABC):
    """A Patch to be applied to the data that is returned from `load_job_series` + `update_job_series_rgu`.

    A Patch can specify a domain of the data that it applies to, and a set of
    functions that will be applied to the data.
    """

    name: str
    """Name of the patch."""

    domain: PartialDomain | ConcreteDomain | None = None
    """Defines on which part of the data this patch applies. If None, the patch applies to all data."""

    def applies_to(self, data_domain: ConcreteDomain) -> bool:
        """Returns whether this patch applies to the given domain."""
        if self.domain is None:
            return True
        patch_domain = self.domain
        if isinstance(self.domain, PartialDomain):
            patch_domain = self.domain.replace_all_with_values(data_domain)
        assert isinstance(patch_domain, ConcreteDomain)
        return data_domain <= patch_domain

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the patch to the data."""
        if not self.applies_to(ConcreteDomain.from_data(data)):
            return data
        return self(data)

    @abc.abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
