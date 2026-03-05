import pytest
import gc
import asyncio


@pytest.fixture(autouse=True)
async def _cleanup_async_between_tests():
    # Run the test
    yield
    # Force cleanup of async generators/tasks before loop teardown
    await asyncio.sleep(0)
    gc.collect()
