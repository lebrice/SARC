import dataclasses
import functools
import hashlib
import inspect
import logging
import os
import pickle
import tempfile
import typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

import simple_parsing

from sarc.core.models.users import UserData

logger = logging.getLogger(__name__)

MTL = ZoneInfo("America/Montreal")
CACHE_DIR: Path | None = (
    Path(os.environ["CF_DATA"])
    if "CF_DATA" in os.environ
    else Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
)


def _midnight(dt: datetime) -> datetime:
    """Returns the start of the given day (hour 00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class FilteringOptions:
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


def cache_results_to_file[**P, OutT](fn: Callable[P, OutT]) -> Callable[P, OutT]:
    """Caches a function in a given cache dir."""
    assert CACHE_DIR and CACHE_DIR.exists() and CACHE_DIR.is_dir()

    if inspect.iscoroutinefunction(fn):
        raise NotImplementedError("Can't cache result of coroutines just yet.")

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> OutT:
        """Decorator to cache the results of a function."""
        assert CACHE_DIR and CACHE_DIR.exists() and CACHE_DIR.is_dir()
        cache_file = CACHE_DIR / _get_cache_file_name(fn, *args, **kwargs)
        if cache_file.exists():
            logger.info(f"Loading result of {fn.__name__} from {cache_file}")
            if cache_file.suffix == ".txt":
                result = cache_file.read_text()
                # the function returns a string (`OutT` is `str`)
                return typing.cast(OutT, result)
            else:
                result = pickle.loads(cache_file.read_bytes())
            # if inspect.iscoroutinefunction(fn):
            #     # If the function is a coroutine, we need to return an awaitable.
            #     return AwaitableWrapper(result)
            # return AwaitableWrapper(result)
            return result
        else:
            logger.debug(f"Cache miss for {fn.__name__} at {cache_file}")
            result = fn(*args, **kwargs)
            # if inspect.iscoroutinefunction(fn):
            #     result = result.__await__()
            if cache_file.suffix == ".txt":
                if not isinstance(result, str):
                    raise RuntimeError(
                        f"Result should be str (annotation says so!), but got {result}"
                    )
                cache_file.write_text(result)
            else:
                cache_file.write_bytes(pickle.dumps(result))
            logger.info(f"Saved result of computing {fn.__name__} to {cache_file}")
            return result

    return wrapper


def _get_cache_file_name[**P](
    fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
) -> str:
    # More interpretable than using this:
    # return hashlib.md5(
    #     json.dumps((fn.__name__, args, kwargs), sort_keys=True, default=str).encode()
    # ).hexdigest()

    def _hash(v) -> str:
        match v:
            case FilteringOptions():
                return "-".join(
                    [
                        _hash(v.get_users()),
                        _hash(v.start),
                        _hash(v.end),
                        _hash(v.clusters),
                    ]
                )

            case None | str():
                return str(v).removesuffix("@mila.quebec")  # no quotes around strings.
            case int() | float():
                return repr(v)
            case datetime(hour=0, minute=0, second=0, tzinfo=MTL) as d:
                return d.strftime("%Y-%m-%d")
            case datetime() as v:
                return v.strftime("%Y-%m-%dT%H:%M:%S%z")
            case [UserData(), *_]:
                return _hash(sorted([student.username for student in v]))
            case [str(), *_] if len(v) > 2:
                # If there are more than 3 strings, hash them together.
                return hashlib.md5("+".join(sorted(v)).encode()).hexdigest()[:12]
            case list() | tuple():
                return "+".join(sorted(map(_hash, v)))
            case {"$in": list(values)}:
                # Special case for MongoDB-like queries.
                return _hash(values)
            case _:
                raise NotImplementedError(
                    f"Unsupported arg type: {v} of type {type(v)}"
                )

    hashed_args = "-".join(map(_hash, args)) + "-".join(
        f"{k}-{_hash(v)}" for k, v in kwargs.items()
    )
    extension = ".pkl"
    try:
        if typing.get_type_hints(fn).get("return") is str:
            extension = ".txt"
    except TypeError:
        pass
    return f"{fn.__name__}-{hashed_args}{extension}"
