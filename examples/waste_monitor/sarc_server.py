# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "rich",
#     "sarc",
#     "textual",
#     "fastapi",
# ]
#
# [tool.uv.sources]
# sarc = { path = "../../" }
# ///
"""TODO: A simple fastapi server that serves the data from SARC until there is an actual proper SARC server.

This is to be run in another process.
"""

import functools
from datetime import datetime
from typing import Annotated

import fastapi.responses
from fastapi import FastAPI, Query

from .common_utils import FilteringOptions
from .sarc_patches import get_clean_sarc_data

app = FastAPI(default_response_class=fastapi.responses.ORJSONResponse)

# _get_data = cached(get_clean_sarc_data)
_get_data = functools.lru_cache(maxsize=4)(get_clean_sarc_data)


@app.get("/jobs/")
async def get_job_ids(
    start: datetime, end: datetime | None, user: str | None, cluster: str | None
) -> list[dict]:
    if user is None:
        users = ()
    elif "," in user:
        users = tuple(sorted(user.split(",")))
    else:
        users = (user,)
    if cluster is None:
        clusters = ()
    elif "," in cluster:
        clusters = tuple(sorted(cluster.split(",")))
    else:
        clusters = (cluster,)
    end = end or datetime.now()

    options = FilteringOptions(start=start, end=end, user=users, clusters=clusters)
    data = _get_data(options)
    # IDEA: only return the job ids, so that the client can fetch individual job data later.
    # return data["job_id"].tolist() if not data.empty else []
    dicts = data.to_dict(orient="records") if not data.empty else []
    for d in dicts:
        d.pop("id")
    return dicts
    # return  if not data.empty else []

@app.get("/job/")
async def _get_jobs(job_ids: Annotated[list[int], Query()]) -> list[dict]:
    # Idea: Get info for a particular job with the given job id.
    raise NotImplementedError

@app.get("/")
async def root():
    return {"message": "Hello World"}
    return {"message": "Hello World"}
