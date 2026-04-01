import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.filler_worker import FillerWorker


class TestFillerWorkerComputeSemantics:
    def test_batch_size_controls_gemm_launch_count(self):
        with patch.dict(
            "os.environ",
            {
                "FILLER_BATCH_SIZE": "5",
                "FILLER_STREAMS": "2",
            },
        ):
            worker = FillerWorker()

        work_items = [{"slot": 0}, {"slot": 1}]
        worker._ensure_compute_resources = Mock(return_value=work_items)
        worker._launch_gemm = Mock()
        worker._synchronize_device = Mock()

        worker._dispatch_compute(1024)

        assert worker._launch_gemm.call_count == 5
        launched = [call.args[0] for call in worker._launch_gemm.call_args_list]
        assert launched == [
            work_items[0],
            work_items[1],
            work_items[0],
            work_items[1],
            work_items[0],
        ]
        worker._synchronize_device.assert_called_once_with()

    def test_streams_create_real_cuda_streams_when_available(self):
        with patch.dict(
            "os.environ",
            {
                "FILLER_BATCH_SIZE": "4",
                "FILLER_STREAMS": "3",
            },
        ):
            worker = FillerWorker()

        allocated = []

        def fake_allocate(size, stream):
            allocated.append(stream)
            return {"stream": stream, "size": size}

        mock_stream_ctor = Mock(side_effect=["s0", "s1", "s2"])
        worker._allocate_work_item = Mock(side_effect=fake_allocate)

        with patch("src.filler_worker.torch.cuda.is_available", return_value=True):
            with patch("src.filler_worker.torch.cuda.Stream", mock_stream_ctor):
                resources = worker._ensure_compute_resources(2048)

        assert len(resources) == 3
        assert allocated == ["s0", "s1", "s2"]
        assert worker._stream_pool == ["s0", "s1", "s2"]

    def test_compute_buffers_are_reused_for_same_matrix_size(self):
        with patch.dict(
            "os.environ",
            {
                "FILLER_BATCH_SIZE": "2",
                "FILLER_STREAMS": "1",
            },
        ):
            worker = FillerWorker()

        worker._allocate_work_item = Mock(return_value={"stream": None, "size": 4096})

        with patch("src.filler_worker.torch.cuda.is_available", return_value=False):
            first = worker._ensure_compute_resources(4096)
            second = worker._ensure_compute_resources(4096)
            third = worker._ensure_compute_resources(8192)

        assert first is second
        assert third is not first
        assert worker._allocate_work_item.call_count == 2

    def test_single_stream_uses_default_stream_path(self):
        with patch.dict(
            "os.environ",
            {
                "FILLER_STREAMS": "1",
            },
        ):
            worker = FillerWorker()

        with patch("src.filler_worker.torch.cuda.is_available", return_value=True):
            resources = worker._ensure_compute_resources = Mock(
                return_value=[{"stream": None}]
            )
            worker._launch_gemm = Mock()
            worker._synchronize_device = Mock()
            worker._dispatch_compute(1024)

        worker._launch_gemm.assert_called()
        worker._synchronize_device.assert_called_once_with()
