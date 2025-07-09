""" TODO: A Python client to communicate with the SARC server to get sarc data.
"""

import datetime
import functools

import httpx

sarc_server_address = "http://localhost:8000"

@functools.cache
def get_available_clusters():
    from sarc.client.job import get_available_clusters

    return tuple(get_available_clusters())

async def get_jobs(
    start: datetime.datetime, end: datetime.datetime | None, user: str | list[str] | None, cluster: str | list[str] | None
):
    
    url = f"{sarc_server_address}/jobs"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params={
            "start": start.isoformat(),
            "end": end.isoformat() if end else None,
            "user": user if isinstance(user, str) else ",".join(sorted(set(user))) if user else None,
            "cluster": cluster if isinstance(cluster, str) else ",".join(sorted(set(cluster))) if cluster else None,
        })
        assert False, response.text


if __name__ == "__main__":
    import asyncio

    start = datetime.datetime(2023, 10, 1)
    end = datetime.datetime(2023, 10, 31)
    user = "normandf"
    cluster = "mila"
    asyncio.run(get_jobs(start, end, user, cluster))