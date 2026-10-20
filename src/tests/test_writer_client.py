from queue import Queue
from types import SimpleNamespace

from app import _get_sqlite_busy_timeout_ms
from app.writer.client import WriterClient
from app.writer.protocol import WriteCommand, WriteCommandType, WriteResult


class _ReplyQueue:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def get(self, timeout: float) -> WriteResult:
        self.timeout = timeout
        return WriteResult("cmd", True)


def test_submit_waits_longer_than_the_sqlite_busy_timeout() -> None:
    # A write can wait for the SQLite lock up to busy_timeout before it runs,
    # so a shorter reply wait fails writes that later commit (#267).
    reply_q = _ReplyQueue()
    client = WriterClient()
    client.queue = Queue()
    client.manager = SimpleNamespace(Queue=lambda: reply_q)

    cmd = WriteCommand(id="cmd", type=WriteCommandType.ACTION, model=None, data={})
    result = client.submit(cmd, wait=True)

    assert result is not None and result.success
    assert reply_q.timeout is not None
    assert reply_q.timeout * 1000 > _get_sqlite_busy_timeout_ms()
