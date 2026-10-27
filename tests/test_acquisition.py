"""Malformed range responses must never turn into archive downloads."""

from unittest.mock import patch

import pytest

from video_compute_value.acquire import RangeReader
from video_compute_value.common import Blocked


class FakeLedger:
    def __init__(self):
        self.transferred = 0
    def allow_video_transfer(self, size):
        pass
    def check(self, **kwargs):
        pass
    def charge_transfer(self, size, video=False):
        self.transferred += size


class Response:
    def __init__(self, status, content_range, data):
        self.status, self.headers, self.data = status, {"Content-Range": content_range}, data
        self.read_called = False
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, size):
        self.read_called = True
        return self.data[:size]


@pytest.mark.parametrize("status,content_range", [(200, None), (206, "bytes 0-3/99"), (206, "bytes 4-7/100")])
def test_full_or_wrong_range_rejected_before_read(status, content_range):
    response = Response(status, content_range, b"data")
    with patch("video_compute_value.acquire.request", return_value=response):
        with pytest.raises(Blocked, match="exact byte range"):
            RangeReader("https://example.invalid/file.zip", 100, FakeLedger()).read(4)
    assert not response.read_called


def test_valid_range_charged_and_truncated_transfer_preserved():
    ledger = FakeLedger()
    with patch("video_compute_value.acquire.request", return_value=Response(206, "bytes 0-3/100", b"data")):
        reader = RangeReader("https://example.invalid/file.zip", 100, ledger)
        assert reader.read(4) == b"data" and reader.tell() == 4 and ledger.transferred == 4
    with patch("video_compute_value.acquire.request", return_value=Response(206, "bytes 0-3/100", b"da")):
        with pytest.raises(Blocked, match="Truncated"):
            RangeReader("https://example.invalid/file.zip", 100, ledger).read(4)
    assert ledger.transferred == 6
