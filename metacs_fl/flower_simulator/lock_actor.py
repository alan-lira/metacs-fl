from asyncio import Lock, TimeoutError, wait_for
from ray import remote


@remote
class LockActor:

    def __init__(self):
        self._lock = Lock()

    async def acquire(self,
                      timeout: float | None = 10.0) -> bool:
        try:
            if timeout is None:
                await self._lock.acquire()
                return True
            else:
                return await wait_for(self._lock.acquire(), timeout)
        except TimeoutError:
            return False

    async def release(self) -> None:
        if self._lock.locked():
            self._lock.release()
