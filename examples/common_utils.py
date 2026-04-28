from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import os
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO
from zoneinfo import ZoneInfo

import gifnoc
import paramiko
import paramiko.config
import simple_parsing
import yaml
from simple_parsing.helpers.serialization.serializable import from_dict
from typing_extensions import Self, Sequence

from sarc.config import ClientConfig

MTL = ZoneInfo("America/Montreal")

logger = logging.getLogger(__name__)


# todo: add a proper date here.
CLUSTER_DOWN: dict[str, bool] = {"cedar": datetime.now() < datetime(2025, 7, 1)}
sarc_client_config_file = Path(__file__).parent.parent / "config/sarc-client.yaml"
sarc_dev_config_file = Path(__file__).parent.parent / "config/sarc-dev.yaml"
if sarc_client_config_file.exists():
    # This script is being executed either from the SARC root, or maybe from an editable install
    # of the SARC package.
    assert sarc_dev_config_file.exists()
elif (
    other_possible_config_path := sarc_client_config_file.parent.parent
    / "sarc"
    / "config"
    / sarc_client_config_file.name
).exists():
    # SARC was installed as a package, and the package-data was included a the path `sarc/config`
    # (via `tool.hatch.build.targets.wheel.force-include`), so the sarc configs are actually now
    # inside SARC (instead of being a separate package in the site-packages directory).
    sarc_client_config_file = other_possible_config_path
    sarc_dev_config_file = sarc_client_config_file.parent / sarc_dev_config_file.name

assert sarc_client_config_file.exists(), sarc_client_config_file
assert sarc_dev_config_file.exists(), sarc_dev_config_file

gifnoc.set_sources(sarc_client_config_file)

sarc_client_config = from_dict(
    ClientConfig, yaml.safe_load(sarc_client_config_file.read_text())["sarc"]
)


CACHE_DIR: Path | None = (
    Path(os.environ["CF_DATA"])
    if "CF_DATA" in os.environ
    else Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
)


async def run_subprocess(
    cmd: str,
    input: str | None = None,
    stdout: int | IO[bytes] | None = asyncio.subprocess.PIPE,
    stderr: int | IO[bytes] | None = asyncio.subprocess.PIPE,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    logger.debug(f"Running command: {cmd!r}")
    proc = await asyncio.create_subprocess_shell(cmd, stdout=stdout, stderr=stderr)
    out_stdout, out_stderr = await proc.communicate(
        input.encode() if input is not None else None
    )
    assert proc.returncode is not None
    if check and proc.returncode != 0:
        raise _CalledProcessError(
            cmd=cmd,
            returncode=proc.returncode,
            output=out_stdout.decode() if out_stdout is not None else None,
            stderr=out_stderr.decode() if out_stderr is not None else None,
        )
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=out_stdout.decode() if out_stdout is not None else None,
        stderr=out_stderr.decode() if out_stderr is not None else None,
    )


class _CalledProcessError(subprocess.CalledProcessError):
    """Custom error class to handle subprocess errors with additional context."""

    def __init__(self, returncode, cmd, output=None, stderr=None):
        super().__init__(returncode, cmd, output=output, stderr=stderr)

    def __str__(self):
        if self.returncode and self.returncode < 0:
            try:
                return "Command '%s' died with %r." % (
                    self.cmd,
                    signal.Signals(-self.returncode),
                )
            except ValueError:
                return "Command '%s' died with unknown signal %d." % (
                    self.cmd,
                    -self.returncode,
                )
        else:
            print(self.stderr, file=sys.stderr)
            return "Command '%s' returned non-zero exit status %d." % (
                self.cmd,
                self.returncode,
            )

    @classmethod
    def from_completed(cls, completed: subprocess.CompletedProcess[str]) -> Self:
        raise cls(
            cmd=completed.args,
            returncode=completed.returncode,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def midnight(dt: datetime) -> datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


# sarc_dev_config = from_dict(
#     Config, yaml.safe_load(sarc_dev_config_file.read_text())["sarc"]
# )
# assert False, sarc_client_config


@functools.total_ordering
@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class FilteringOptions:
    """Configuration options for this script."""

    start: datetime = simple_parsing.field(
        default=(midnight(datetime.now(tz=MTL)) - timedelta(days=30)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ Start date. """

    end: datetime = simple_parsing.field(
        default=midnight(datetime.now(tz=MTL)),
        type=lambda d: datetime.fromisoformat(d).astimezone(MTL),
    )
    """ End date. """

    user: Sequence[str] = dataclasses.field(default_factory=tuple)
    """ Which user(s) to query information for. Leave blank to get a global compute profile."""

    clusters: Sequence[str] = dataclasses.field(default_factory=tuple)
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

    def __eq__(self, other: object) -> bool:
        """Returns whether this filter is equal to the other."""
        if not isinstance(other, FilteringOptions):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and set(self.user) == set(other.user)
            and set(self.clusters) == set(other.clusters)
        )

    def __lt__(self, other: Self) -> bool:
        """Returns whether this filter is strictly more restrictive than the other."""
        if not isinstance(other, FilteringOptions):
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


@functools.cache
def get_available_clusters():
    from sarc.client.job import get_available_clusters

    return tuple(get_available_clusters())


async def setup_sarc_connection():
    ssh_config = paramiko.config.SSHConfig.from_path(Path.home() / ".ssh" / "config")
    control_socket_path = Path(
        ssh_config.lookup("sarc").get(
            "controlpath", Path.home() / ".cache" / "ssh" / "%r@%h:%p"
        )
    ).expanduser()

    # Try sarc01-dev, then sarc, then mila.
    user = ssh_config.lookup("sarc01-dev").get(
        "user",
        ssh_config.lookup("sarc").get("user", ssh_config.lookup("mila").get("user")),
    )
    if not user:
        raise ValueError(
            "Don't know which user to use when connecting to sarc! "
            "Make sure you have either a 'mila' or 'sarc' entry in your SSH config file."
        )

    control_socket_path.parent.mkdir(parents=True, exist_ok=True)
    multiplexing_args = (
        f"-o ControlMaster=auto "
        f"-o 'ControlPath={control_socket_path}' "
        f"-o ControlPersist=yes"
    )

    sarc_client_connection_string = yaml.safe_load(sarc_client_config_file.read_text())[
        "sarc"
    ]["mongo"]["connection_string"]
    assert isinstance(sarc_client_connection_string, str)
    # "mongodb://readuser:readpwd@localhost:8123/sarc" --> "8123"
    sarc_client_local_port = (
        sarc_client_connection_string.rpartition("@")[2]
        .partition(":")[2]
        .partition("/")[0]
    )

    sarc_dev_connection_string = yaml.safe_load(sarc_dev_config_file.read_text())[
        "sarc"
    ]["mongo"]["connection_string"]
    assert isinstance(sarc_dev_connection_string, str)
    # "mongodb://localhost:27017/sarc-dev" --> "27017"
    sarc_dev_remote_port = (
        sarc_dev_connection_string.rpartition("@")[
            2
        ]  # might return the whole string if there is no user:pwd@...
        .partition("://")[2]  # "localhost:27017/sarc-dev"
        .partition("/")[0]  # "localhost:27017"
        .partition(":")[2]  # "27017"
    )
    assert sarc_client_local_port
    assert sarc_dev_remote_port

    port_forwarding_args = (
        f"-o 'LocalForward={sarc_client_local_port} 127.0.0.1:{sarc_dev_remote_port}'"
    )

    await run_subprocess(
        f"ssh -o ProxyJump=mila {port_forwarding_args} {multiplexing_args} {user}@sarc01-dev echo OK",
        stdout=subprocess.PIPE,
    )
    logger.info(
        "Successfully set up SSH connection to sarc01-dev with port forwarding."
    )
