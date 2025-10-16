import asyncio
import time
import threading
import pytest

from app.models.infer_queue import InferQueue


# 测试单个任务提交和结果返回
@pytest.mark.asyncio
async def test_single_submission_result():
    q = InferQueue()

    async def run():
        return await q.submit(lambda: 42)

    result = await run()
    assert result == 42


# 测试循环运行状态切换
@pytest.mark.asyncio
async def test_running_state():
    q = InferQueue()

    async def run():
        return await q.submit(lambda: 42)

    await run()
    assert q._is_running is True
    await asyncio.sleep(0)
    assert q._is_running is False


# 多个任务提交，确保按顺序执行且没有并发。
@pytest.mark.asyncio
async def test_multiple_submissions_sequential_non_overlapping():
    q = InferQueue()
    active_lock = threading.Lock()
    running = False
    order = []

    def make_task(i: int, sleep: float):
        def _task():
            nonlocal running
            with active_lock:
                assert not running, "任务并发执行了，违反串行预期"
                running = True
            try:
                order.append(f"start-{i}")
                time.sleep(sleep)
                order.append(f"end-{i}")
                return i
            finally:
                with active_lock:
                    running = False

        return _task

    results = await asyncio.gather(
        q.submit(make_task(0, 0.02)),
        q.submit(make_task(1, 0.01)),
        q.submit(make_task(2, 0.015)),
    )
    assert results == [0, 1, 2]
    for i in range(3):
        start_idx = order.index(f"start-{i}")
        end_idx = order.index(f"end-{i}")
        assert end_idx == start_idx + 1, (
            f"任务 {i} 的执行被其它任务打断，顺序异常: {order}"
        )

    await asyncio.sleep(0.05)
    assert q._is_running is False


# 测试任务中抛异常，确保异常能传递回调用者
@pytest.mark.asyncio
async def test_exception_propagation():
    q = InferQueue()

    class MyErr(Exception):
        pass

    def bad_task():
        raise MyErr("boom")

    with pytest.raises(MyErr, match="boom"):
        await q.submit(bad_task)

    await asyncio.sleep(0.05)
    assert q._is_running is False


# 测试一个任务抛异常不影响后续任务执行
@pytest.mark.asyncio
async def test_exception_does_not_block_subsequent_tasks():
    q = InferQueue()

    def good():
        return "GOOD"

    def bad():
        raise ValueError("BAD")

    # 顺序：好 -> 坏 -> 好
    res1 = await q.submit(good)
    assert res1 == "GOOD"

    with pytest.raises(ValueError):
        await q.submit(bad)

    res2 = await q.submit(good)
    assert res2 == "GOOD"

    await asyncio.sleep(0.05)
    assert q._is_running is False
