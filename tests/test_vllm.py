"""Tests for vLLM server utilities."""

import subprocess
import warnings
from unittest.mock import Mock, patch

import pytest
import requests

from scicode_lint.vllm import (
    GPUInfo,
    ServerInfo,
    VLLMServer,
    _get_default_model,
    get_gpu_info,
    get_server_info,
    is_running,
    start_server,
    stop_server,
    wait_for_ready,
)

# Use the same default model as production code (DRY)
DEFAULT_MODEL = _get_default_model()


class TestIsRunning:
    """Tests for is_running function."""

    def test_is_running_when_server_responds(self) -> None:
        """Should return True when server health endpoint returns 200."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert is_running("http://localhost:5001") is True
            mock_get.assert_called_once_with("http://localhost:5001/health", timeout=2)

    def test_is_running_when_server_down(self) -> None:
        """Should return False when server is not responding."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")

            assert is_running("http://localhost:5001") is False

    def test_is_running_with_non_200_status(self) -> None:
        """Should return False when server returns non-200 status."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            assert is_running("http://localhost:5001") is False


class TestWaitForReady:
    """Tests for wait_for_ready function."""

    def test_wait_for_ready_immediate(self) -> None:
        """Should return True immediately if server is ready."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            assert wait_for_ready(timeout=5) is True

    def test_wait_for_ready_after_delay(self) -> None:
        """Should return True after server becomes ready."""
        call_count = [0]

        def mock_is_running(_: str) -> bool:
            call_count[0] += 1
            return call_count[0] >= 3  # Ready on third call

        with patch("scicode_lint.vllm.is_running", side_effect=mock_is_running):
            with patch("time.sleep"):  # Speed up test
                assert wait_for_ready(timeout=10, check_interval=0.1) is True

    def test_wait_for_ready_timeout(self) -> None:
        """Should return False if timeout reached."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("time.sleep"):  # Speed up test
                assert wait_for_ready(timeout=1, check_interval=0.1) is False


class TestStartServer:
    """Tests for start_server function."""

    def test_start_server_raises_if_already_running(self) -> None:
        """Should raise RuntimeError if server already running on port."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with pytest.raises(RuntimeError, match="vLLM server already running on port 5001"):
                start_server(port=5001)

    def test_start_server_raises_if_vllm_not_found(self) -> None:
        """Should raise FileNotFoundError if vllm command not found."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm._auto_detect_vram_settings", return_value=(20096, 0.9)):
                with patch("subprocess.Popen", side_effect=FileNotFoundError):
                    with pytest.raises(FileNotFoundError, match="vllm command not found"):
                        start_server()

    def test_start_server_success(self) -> None:
        """Should start server and return process."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm._check_vllm_version"):
                with patch(
                    "scicode_lint.vllm._auto_detect_vram_settings", return_value=(20096, 0.9)
                ):
                    mock_proc = Mock(spec=subprocess.Popen)
                    with patch("subprocess.Popen", return_value=mock_proc):
                        proc = start_server(wait=False)

                        assert proc == mock_proc

    def test_start_server_with_wait(self) -> None:
        """Should wait for server to be ready when wait=True."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm._check_vllm_version"):
                with patch(
                    "scicode_lint.vllm._auto_detect_vram_settings", return_value=(20096, 0.9)
                ):
                    mock_proc = Mock(spec=subprocess.Popen)
                    with patch("subprocess.Popen", return_value=mock_proc):
                        with patch("scicode_lint.vllm.wait_for_ready", return_value=True):
                            proc = start_server(wait=True)
                            assert proc == mock_proc

    def test_start_server_wait_timeout(self) -> None:
        """Should raise TimeoutError if server doesn't start in time."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm._check_vllm_version"):
                with patch(
                    "scicode_lint.vllm._auto_detect_vram_settings", return_value=(20096, 0.9)
                ):
                    mock_proc = Mock(spec=subprocess.Popen)
                    with patch("subprocess.Popen", return_value=mock_proc):
                        with patch("scicode_lint.vllm.wait_for_ready", return_value=False):
                            with patch("scicode_lint.vllm.stop_server"):
                                with pytest.raises(TimeoutError, match="Server failed to start"):
                                    start_server(wait=True, wait_timeout=5)


class TestStopServer:
    """Tests for stop_server function."""

    def test_stop_server_graceful(self) -> None:
        """Should terminate process gracefully."""
        mock_proc = Mock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # Still running
        mock_proc.wait.return_value = None

        stop_server(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=10)

    def test_stop_server_force_kill_on_timeout(self) -> None:
        """Should force kill if graceful termination times out."""
        mock_proc = Mock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("vllm", 10),
            None,
        ]

        stop_server(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    def test_stop_server_already_stopped(self) -> None:
        """Should do nothing if process already stopped."""
        mock_proc = Mock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0  # Already stopped

        stop_server(mock_proc)

        mock_proc.terminate.assert_not_called()


class TestVLLMServerContextManager:
    """Tests for VLLMServer context manager."""

    def test_local_server_not_running_starts_and_stops(self) -> None:
        """Should start local server if not running, stop on exit."""
        mock_proc = Mock(spec=subprocess.Popen)

        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm.start_server", return_value=mock_proc):
                with patch("scicode_lint.vllm.stop_server") as mock_stop:
                    with VLLMServer():
                        pass

                    mock_stop.assert_called_once_with(mock_proc)

    def test_local_server_already_running_reuses(self) -> None:
        """Should reuse local server if already running, not stop on exit."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("scicode_lint.vllm.requests.get") as mock_get:
                # Mock models endpoint
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {"data": [{"id": DEFAULT_MODEL}]}

                with patch("scicode_lint.vllm.start_server") as mock_start:
                    with patch("scicode_lint.vllm.stop_server") as mock_stop:
                        with VLLMServer():
                            pass

                        mock_start.assert_not_called()
                        mock_stop.assert_not_called()

    def test_local_server_wrong_model_warns(self) -> None:
        """Should warn if local server running with different model."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("scicode_lint.vllm.requests.get") as mock_get:
                # Mock models endpoint with different model
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {"data": [{"id": "different-model"}]}

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    with VLLMServer(model=DEFAULT_MODEL):
                        pass

                    # Should have warning
                    assert len(w) == 1
                    assert "running with model 'different-model'" in str(w[0].message)
                    assert issubclass(w[0].category, RuntimeWarning)

    def test_remote_server_running_reuses(self) -> None:
        """Should reuse remote server if running."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("scicode_lint.vllm.start_server") as mock_start:
                with patch("scicode_lint.vllm.stop_server") as mock_stop:
                    with VLLMServer(base_url="http://remote:5001"):
                        pass

                    mock_start.assert_not_called()
                    mock_stop.assert_not_called()

    def test_remote_server_not_running_raises(self) -> None:
        """Should raise error if remote server not reachable."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with pytest.raises(RuntimeError, match="Remote vLLM server not reachable"):
                with VLLMServer(base_url="http://remote:5001"):
                    pass

    def test_context_manager_exception_still_stops_server(self) -> None:
        """Should stop server even if exception raised in context."""
        mock_proc = Mock(spec=subprocess.Popen)

        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm.start_server", return_value=mock_proc):
                with patch("scicode_lint.vllm.stop_server") as mock_stop:
                    with pytest.raises(ValueError):
                        with VLLMServer():
                            raise ValueError("Test error")

                    # Should still stop server
                    mock_stop.assert_called_once_with(mock_proc)


class TestGetGPUInfo:
    """Tests for get_gpu_info function."""

    def test_get_gpu_info_success(self) -> None:
        """Should parse nvidia-smi output correctly."""
        mock_result = Mock()
        mock_result.stdout = "NVIDIA RTX 4000 Ada, 20480, 4096, 16384, 15"

        mock_cuda = Mock()
        mock_cuda.stdout = "535.183.01"

        with patch("subprocess.run", side_effect=[mock_result, mock_cuda]):
            info = get_gpu_info()

            assert info is not None
            assert info.name == "NVIDIA RTX 4000 Ada"
            assert info.total_memory_mb == 20480
            assert info.used_memory_mb == 4096
            assert info.free_memory_mb == 16384
            assert info.utilization_percent == 15
            assert info.cuda_version == "535.183.01"

    def test_get_gpu_info_nvidia_smi_not_found(self) -> None:
        """Should return None if nvidia-smi not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            info = get_gpu_info()
            assert info is None

    def test_get_gpu_info_parse_error(self) -> None:
        """Should return None if parsing fails."""
        mock_result = Mock()
        mock_result.stdout = "invalid output"

        with patch("subprocess.run", return_value=mock_result):
            info = get_gpu_info()
            assert info is None


class TestGetServerInfo:
    """Tests for get_server_info function."""

    def test_get_server_info_running(self) -> None:
        """Should return server info when running."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                model_id = "RedHatAI/Qwen3-8B-FP8-dynamic"
                mock_response.json.return_value = {"data": [{"id": model_id}]}
                mock_get.return_value = mock_response

                info = get_server_info()

                assert info.is_running is True
                assert info.model == DEFAULT_MODEL
                assert info.base_url == "http://localhost:5001"

    def test_get_server_info_not_running(self) -> None:
        """Should return not running status."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            info = get_server_info()

            assert info.is_running is False
            assert info.model is None

    def test_get_server_info_model_fetch_fails(self) -> None:
        """Should handle model fetch failure gracefully."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("requests.get", side_effect=requests.RequestException):
                info = get_server_info()

                assert info.is_running is True
                assert info.model is None


class TestDataClasses:
    """Tests for dataclasses."""

    def test_gpu_info_dataclass(self) -> None:
        """Should create GPUInfo correctly."""
        info = GPUInfo(
            name="Test GPU",
            total_memory_mb=16000,
            used_memory_mb=8000,
            free_memory_mb=8000,
            utilization_percent=50,
            cuda_version="12.0",
        )

        assert info.name == "Test GPU"
        assert info.total_memory_mb == 16000
        assert info.used_memory_mb == 8000
        assert info.free_memory_mb == 8000
        assert info.utilization_percent == 50
        assert info.cuda_version == "12.0"

    def test_server_info_dataclass(self) -> None:
        """Should create ServerInfo correctly."""
        info = ServerInfo(model="test-model", is_running=True, base_url="http://localhost:5001")

        assert info.model == "test-model"
        assert info.is_running is True
        assert info.base_url == "http://localhost:5001"


class TestVersionChecking:
    """Tests for vLLM version checking."""

    def test_version_check_passes_for_016(self) -> None:
        """Should pass for vLLM 0.16.0."""
        from scicode_lint.vllm import _check_vllm_version

        mock_result = Mock()
        mock_result.stdout = "vllm 0.16.0"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            _check_vllm_version()  # Should not raise

    def test_version_check_passes_for_017(self) -> None:
        """Should pass for vLLM 0.17.0."""
        from scicode_lint.vllm import _check_vllm_version

        mock_result = Mock()
        mock_result.stdout = "vllm 0.17.0"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            _check_vllm_version()  # Should not raise

    def test_version_check_fails_for_015(self) -> None:
        """Should raise RuntimeError for vLLM 0.15.x."""
        from scicode_lint.vllm import _check_vllm_version

        mock_result = Mock()
        mock_result.stdout = "vllm 0.15.4"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="version 0.15.x is too old"):
                _check_vllm_version()

    def test_version_check_fails_for_014(self) -> None:
        """Should raise RuntimeError for vLLM 0.14.x."""
        from scicode_lint.vllm import _check_vllm_version

        mock_result = Mock()
        mock_result.stdout = "vllm 0.14.0"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="version 0.14.x is too old"):
                _check_vllm_version()

    def test_version_check_handles_vllm_not_found(self) -> None:
        """Should not raise if vllm command not found (will be caught later)."""
        from scicode_lint.vllm import _check_vllm_version

        with patch("subprocess.run", side_effect=FileNotFoundError):
            _check_vllm_version()  # Should not raise


class TestVRAMAutoDetection:
    """Tests for VRAM-based auto-detection."""

    def test_20gb_vram_settings(self) -> None:
        """Should use 20K context and GPU memory from config for 16GB+ VRAM."""
        from scicode_lint.vllm import _auto_detect_vram_settings, _get_gpu_memory_utilization

        # Simulate 20GB VRAM (20475MB like RTX 4000 Ada)
        max_len, gpu_mem = _auto_detect_vram_settings(override_vram_mb=20475)

        assert max_len == 20096
        assert gpu_mem == _get_gpu_memory_utilization()

    def test_16gb_vram_settings(self) -> None:
        """Should use 20K context and GPU memory from config for 16GB VRAM (at minimum)."""
        from scicode_lint.vllm import _auto_detect_vram_settings, _get_gpu_memory_utilization

        # Simulate 16GB VRAM - should succeed (at minimum)
        max_len, gpu_mem = _auto_detect_vram_settings(override_vram_mb=16384)
        assert max_len == 20096
        assert gpu_mem == _get_gpu_memory_utilization()

    def test_12gb_vram_settings(self) -> None:
        """Should raise RuntimeError for 12GB VRAM (below minimum)."""
        import pytest

        from scicode_lint.vllm import _auto_detect_vram_settings

        # Simulate 12GB VRAM - should fail (below any reasonable minimum)
        with pytest.raises(RuntimeError, match="Minimum requirement:"):
            _auto_detect_vram_settings(override_vram_mb=12288)

    def test_8gb_vram_settings(self) -> None:
        """Should raise RuntimeError for 8GB VRAM (below minimum)."""
        import pytest

        from scicode_lint.vllm import _auto_detect_vram_settings

        # Simulate 8GB VRAM - should fail (below any reasonable minimum)
        with pytest.raises(RuntimeError, match="Minimum requirement:"):
            _auto_detect_vram_settings(override_vram_mb=8192)

    def test_vram_boundary_16gb(self) -> None:
        """Should correctly handle VRAM at boundary defined in config."""
        import pytest

        from scicode_lint.vllm import _auto_detect_vram_settings, _get_min_vram_mb

        # Get actual minimum from config
        min_vram = _get_min_vram_mb()

        # Just below threshold - should fail
        with pytest.raises(RuntimeError, match="Minimum requirement:"):
            _auto_detect_vram_settings(override_vram_mb=min_vram - 1)

        # At threshold - should succeed
        max_len, gpu_mem = _auto_detect_vram_settings(override_vram_mb=min_vram)
        assert max_len > 0  # Config-driven
        assert gpu_mem > 0  # Config-driven

    def test_start_server_uses_standard_settings(self) -> None:
        """Should use 20K context and GPU memory from config for 16GB+ VRAM."""
        from scicode_lint.vllm import _auto_detect_vram_settings, _get_gpu_memory_utilization

        expected_gpu_mem = _get_gpu_memory_utilization()

        # 16GB VRAM
        max_len, gpu_mem = _auto_detect_vram_settings(override_vram_mb=16384)
        assert max_len == 20096
        assert gpu_mem == expected_gpu_mem

        # 24GB VRAM
        max_len, gpu_mem = _auto_detect_vram_settings(override_vram_mb=24576)
        assert max_len == 20096
        assert gpu_mem == expected_gpu_mem

    def test_no_gpu_fallback(self) -> None:
        """Should raise RuntimeError when GPU cannot be detected."""
        import pytest

        from scicode_lint.vllm import _auto_detect_vram_settings

        with patch("scicode_lint.vllm.get_gpu_info", return_value=None):
            # Should fail if cannot detect GPU
            with pytest.raises(RuntimeError, match="Cannot detect GPU VRAM"):
                _auto_detect_vram_settings()
