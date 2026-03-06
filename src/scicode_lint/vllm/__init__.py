"""Lightweight utilities for managing vLLM server lifecycle.

Optional utilities for GenAI agents and automated testing.
Not required for normal scicode-lint usage.

Supports both local and remote vLLM servers:
    - **Local servers**: Can start/stop programmatically
    - **Remote servers**: Can verify connectivity only

Features:
    - Start/stop local vLLM servers programmatically
    - Check server status and GPU information
    - Context manager for automatic lifecycle management
    - Support for remote vLLM servers

Example - Local Server Management:
    >>> from scicode_lint.vllm import start_server, stop_server, wait_for_ready
    >>> proc = start_server()
    >>> if wait_for_ready():
    ...     linter = SciCodeLinter()
    ...     result = linter.check_file(Path("myfile.py"))
    >>> stop_server(proc)

Example - Context Manager (Local):
    >>> from scicode_lint.vllm import VLLMServer
    >>> with VLLMServer():  # Auto-starts if needed
    ...     linter = SciCodeLinter()
    ...     result = linter.check_file(Path("myfile.py"))

Example - Remote Server:
    >>> with VLLMServer(base_url="http://gpu-cluster:5001"):
    ...     linter = SciCodeLinter()  # Uses remote server
    ...     result = linter.check_file(Path("myfile.py"))

Example - System Info:
    >>> from scicode_lint.vllm import get_gpu_info, print_system_info
    >>> gpu = get_gpu_info()
    >>> print(f"Free VRAM: {gpu.free_memory_mb} MB")
    >>> print_system_info()  # Print complete system status

Also includes start_vllm.sh bash script for manual server startup.
"""

import subprocess
import time
import types
from dataclasses import dataclass
from typing import Optional

import requests


def is_running(base_url: str = "http://localhost:5001") -> bool:
    """Check if vLLM server is running.

    Args:
        base_url: Server URL to check (default: http://localhost:5001)

    Returns:
        True if server is responding, False otherwise

    Example:
        >>> from scicode_lint.vllm import is_running
        >>> if is_running():
        ...     print("Server is ready")
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False


def wait_for_ready(
    base_url: str = "http://localhost:5001",
    timeout: int = 60,
    check_interval: float = 2.0,
) -> bool:
    """Wait for vLLM server to be ready.

    Args:
        base_url: Server URL to check (default: http://localhost:5001)
        timeout: Maximum time to wait in seconds (default: 60)
        check_interval: Time between checks in seconds (default: 2.0)

    Returns:
        True if server became ready, False if timeout reached

    Example:
        >>> from scicode_lint.vllm import wait_for_ready
        >>> if wait_for_ready(timeout=30):
        ...     print("Server is ready")
        ... else:
        ...     print("Timeout waiting for server")
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_running(base_url):
            return True
        time.sleep(check_interval)
    return False


def _check_vllm_version() -> None:
    """Check if vLLM version is 0.16+ and raise if not."""
    try:
        result = subprocess.run(
            ["vllm", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Parse version from output like "vllm 0.16.0"
        import re

        match = re.search(r"vllm (\d+)\.(\d+)", result.stdout + result.stderr)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            if major == 0 and minor < 16:
                raise RuntimeError(
                    f"vLLM version 0.{minor}.x is too old (requires 0.16+). "
                    "Upgrade with: pip install --upgrade vllm"
                )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # vllm command not found or failed - will be caught by start_server
        pass


def _auto_detect_vram_settings(override_vram_mb: Optional[int] = None) -> tuple[int, float]:
    """Verify VRAM requirements and return standard settings.

    Args:
        override_vram_mb: Override VRAM detection for testing (e.g., 20480 for 20GB)

    Returns:
        Tuple of (max_model_len, gpu_memory_utilization)
    """
    if override_vram_mb is not None:
        vram_mb = override_vram_mb
    else:
        gpu_info = get_gpu_info()
        if gpu_info is None:
            # Cannot detect VRAM - require user to specify
            raise RuntimeError(
                "Cannot detect GPU VRAM. Please ensure nvidia-smi is available.\n"
                "Minimum requirement: 20GB VRAM with native FP8 support"
            )
        vram_mb = gpu_info.total_memory_mb

    # Enforce minimum 20GB VRAM
    if vram_mb < 19500:
        vram_gb = vram_mb // 1024
        raise RuntimeError(
            f"Detected {vram_gb}GB VRAM. Minimum requirement: 20GB VRAM.\n"
            "\n"
            "scicode-lint requires 20GB+ VRAM with native FP8 support.\n"
            "Supported GPUs (compute capability >= 8.9):\n"
            "  • Consumer: RTX 4090 (24GB)\n"
            "  • Workstation: RTX 4000 Ada (20GB), RTX 5000 Ada (32GB)\n"
            "  • Cloud/HPC inference: L4 (24GB), L40 (48GB), A10 (24GB)\n"
            "\n"
            "See INSTALLATION.md for deployment options."
        )

    # All configurations use 16K context (standardized)
    # Covers 90-95th percentile based on 10M+ repo analysis
    # Median: 258 lines, Mean: 879 lines, 90th: ~1,500 lines
    # Paged attention means no waste on smaller files
    return 16000, 0.90


def start_server(
    model: str = "RedHatAI/gemma-3-12b-it-FP8-dynamic",
    port: int = 5001,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    wait: bool = False,
    wait_timeout: int = 60,
) -> subprocess.Popen[str]:
    """Start vLLM server as a subprocess.

    Args:
        model: Model name or path (default: RedHatAI/gemma-3-12b-it-FP8-dynamic)
        port: Port to run on (default: 5001)
        max_model_len: Maximum context length (default: 16000 tokens)
        gpu_memory_utilization: GPU memory to use 0.0-1.0 (default: 0.9)
        wait: Wait for server to be ready before returning (default: False)
        wait_timeout: Timeout for waiting in seconds (default: 60)

    Returns:
        subprocess.Popen object representing the server process

    Raises:
        FileNotFoundError: If vllm command not found
        TimeoutError: If wait=True and server doesn't start in time
        RuntimeError: If server is already running on the port

    Example:
        >>> from scicode_lint.vllm import start_server, stop_server
        >>> proc = start_server(wait=True)  # Auto-detects VRAM settings
        >>> try:
        ...     # Use linter
        ...     linter = SciCodeLinter()
        ...     result = linter.check_file(Path("myfile.py"))
        ... finally:
        ...     stop_server(proc)

    Note:
        Model will download automatically on first run (~13GB).
        Server output goes to stdout/stderr unless redirected.
    """
    # Check vLLM version
    _check_vllm_version()

    # Check if already running
    if is_running(f"http://localhost:{port}"):
        raise RuntimeError(
            f"vLLM server already running on port {port}\n\n"
            f"Suggestions:\n"
            f"  • Use existing server: scicode-lint will auto-detect it\n"
            f"  • Stop server: Use stop_server() or kill the vllm process\n"
            f"  • Use different port: start_server(port={port + 1})"
        )

    # Auto-detect settings if not specified
    if max_model_len is None or gpu_memory_utilization is None:
        auto_len, auto_mem = _auto_detect_vram_settings()
        if max_model_len is None:
            max_model_len = auto_len
        if gpu_memory_utilization is None:
            gpu_memory_utilization = auto_mem

    # Build command
    cmd = [
        "vllm",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model",
        model,
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]

    # Start process
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "vllm command not found. Install with: pip install scicode-lint[vllm-server]"
        )

    # Wait for ready if requested
    if wait:
        if not wait_for_ready(f"http://localhost:{port}", timeout=wait_timeout):
            stop_server(proc)
            raise TimeoutError(f"Server failed to start within {wait_timeout} seconds")

    return proc


def stop_server(process: subprocess.Popen[str], timeout: int = 10) -> None:
    """Stop vLLM server process gracefully.

    Args:
        process: Process object returned by start_server()
        timeout: Timeout for graceful shutdown in seconds (default: 10)

    Example:
        >>> from scicode_lint.vllm import start_server, stop_server
        >>> proc = start_server()
        >>> # ... use server ...
        >>> stop_server(proc)

    Note:
        Tries graceful termination first, then kills if necessary.
    """
    if process.poll() is None:  # Process still running
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class VLLMServer:
    """Context manager for vLLM server lifecycle.

    Supports both local and remote vLLM servers:
    - **Local server (default)**: Starts if not running, stops if we started it
    - **Remote server**: Only verifies it's reachable, never starts/stops

    Args:
        model: Model name or path (only used for local servers)
        port: Port to run on (only used for local servers)
        base_url: Full URL for remote server (e.g., "http://10.0.0.5:5001")
                  If provided, port is ignored and no start/stop attempted
        max_model_len: Maximum context length (local servers only, default: 16000)
        gpu_memory_utilization: GPU memory 0.0-1.0 (local servers only, default: 0.9)
        wait_timeout: Timeout for server startup/verification in seconds

    Example - Local server (auto-start):
        >>> from scicode_lint.vllm import VLLMServer
        >>> from scicode_lint import SciCodeLinter
        >>>
        >>> # Starts local server if not running, stops when done
        >>> with VLLMServer():
        ...     linter = SciCodeLinter()
        ...     result = linter.check_file(Path("myfile.py"))

    Example - Remote server:
        >>> # Just verifies remote server is reachable
        >>> with VLLMServer(base_url="http://gpu-cluster.example.com:5001"):
        ...     linter = SciCodeLinter()
        ...     result = linter.check_file(Path("myfile.py"))
    """

    def __init__(
        self,
        model: str = "RedHatAI/gemma-3-12b-it-FP8-dynamic",
        port: int = 5001,
        base_url: Optional[str] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        wait_timeout: int = 60,
    ):
        """Initialize VLLMServer context manager.

        Uses standard settings: 16K context, 0.9 GPU memory utilization.
        """
        self.model = model
        self.port = port
        self.base_url = base_url

        # Verify VRAM requirements and use standard settings (only for local servers)
        if base_url is None and (max_model_len is None or gpu_memory_utilization is None):
            auto_len, auto_mem = _auto_detect_vram_settings()
            self.max_model_len = max_model_len if max_model_len is not None else auto_len
            self.gpu_memory_utilization = (
                gpu_memory_utilization if gpu_memory_utilization is not None else auto_mem
            )
        else:
            # Use provided values or fallback defaults
            self.max_model_len = max_model_len if max_model_len is not None else 16000
            self.gpu_memory_utilization = (
                gpu_memory_utilization if gpu_memory_utilization is not None else 0.9
            )

        self.wait_timeout = wait_timeout
        self.process: Optional[subprocess.Popen[str]] = None
        self.was_already_running = False
        self.is_remote = base_url is not None

    def __enter__(self) -> "VLLMServer":
        """Start vLLM server if local and not already running."""
        server_url = self.base_url or f"http://localhost:{self.port}"

        # Check if server already running
        if is_running(server_url):
            self.was_already_running = True

            # Warn if running model differs from requested (local only)
            if not self.is_remote:
                try:
                    response = requests.get(f"{server_url}/v1/models", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            running_model = data["data"][0].get("id", "unknown")
                            # Check if models differ (ignoring served-model-name aliases)
                            if self.model not in running_model and running_model not in self.model:
                                import warnings

                                msg = (
                                    f"Local vLLM server is running with model "
                                    f"'{running_model}' but you requested '{self.model}'. "
                                    f"Using existing server. To use the requested model, "
                                    f"stop the server first: pkill -f 'vllm serve'"
                                )
                                warnings.warn(msg, RuntimeWarning)
                except (requests.RequestException, KeyError, IndexError):
                    pass  # Can't verify model, proceed anyway

            return self

        # Remote server not reachable
        if self.is_remote:
            raise RuntimeError(
                f"Remote vLLM server not reachable at {server_url}. "
                "Cannot start remote servers - please start it manually."
            )

        # Start local server
        self.process = start_server(
            model=self.model,
            port=self.port,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            wait=True,
            wait_timeout=self.wait_timeout,
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        """Stop vLLM server only if we started it (local only)."""
        # Only stop if we started the server (don't kill existing/remote servers)
        if self.process and not self.was_already_running and not self.is_remote:
            stop_server(self.process)


@dataclass
class GPUInfo:
    """GPU information from nvidia-smi.

    Attributes:
        name: GPU name (e.g., "NVIDIA RTX 4000 Ada")
        total_memory_mb: Total VRAM in MB
        used_memory_mb: Used VRAM in MB
        free_memory_mb: Free VRAM in MB
        utilization_percent: GPU utilization percentage
        cuda_version: CUDA version
    """

    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: int
    cuda_version: str


@dataclass
class ServerInfo:
    """vLLM server information.

    Attributes:
        model: Model name being served
        is_running: Whether server is responding
        base_url: Server URL
    """

    model: Optional[str]
    is_running: bool
    base_url: str


def get_gpu_info() -> Optional[GPUInfo]:
    """Get GPU information using nvidia-smi.

    Returns:
        GPUInfo object with GPU details, or None if nvidia-smi fails

    Example:
        >>> from scicode_lint.vllm import get_gpu_info
        >>> gpu = get_gpu_info()
        >>> if gpu:
        ...     print(f"GPU: {gpu.name}")
        ...     print(f"VRAM: {gpu.used_memory_mb}/{gpu.total_memory_mb} MB")
        ...     print(f"Free: {gpu.free_memory_mb} MB")
        ...     print(f"Utilization: {gpu.utilization_percent}%")
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse first GPU (most systems have one)
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]

        # Get CUDA version
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        cuda_version = cuda_result.stdout.strip()

        return GPUInfo(
            name=parts[0],
            total_memory_mb=int(parts[1]),
            used_memory_mb=int(parts[2]),
            free_memory_mb=int(parts[3]),
            utilization_percent=int(parts[4]),
            cuda_version=cuda_version,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return None


def get_server_info(base_url: str = "http://localhost:5001") -> ServerInfo:
    """Get vLLM server information.

    Args:
        base_url: Server URL (default: http://localhost:5001)

    Returns:
        ServerInfo object with server details

    Example:
        >>> from scicode_lint.vllm import get_server_info
        >>> info = get_server_info()
        >>> if info.is_running:
        ...     print(f"Model: {info.model}")
        >>> else:
        ...     print("Server not running")
    """
    model_name = None
    running = is_running(base_url)

    if running:
        try:
            # Try to get model info from /v1/models endpoint
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_name = data["data"][0].get("id")
        except (requests.RequestException, KeyError, IndexError):
            pass

    return ServerInfo(model=model_name, is_running=running, base_url=base_url)


def print_system_info() -> None:
    """Print system information (GPU, server status).

    Useful for GenAI agents to check system readiness before starting work.

    Example:
        >>> from scicode_lint.vllm import print_system_info
        >>> print_system_info()
        === GPU Information ===
        GPU: NVIDIA RTX 4000 Ada
        VRAM: 4096/20480 MB (20% used, 16384 MB free)
        Utilization: 15%
        CUDA: 535.183.01

        === vLLM Server ===
        Status: Running
        Model: RedHatAI/gemma-3-12b-it-FP8-dynamic
        URL: http://localhost:5001
    """
    print("=== GPU Information ===")
    gpu = get_gpu_info()
    if gpu:
        print(f"GPU: {gpu.name}")
        print(
            f"VRAM: {gpu.used_memory_mb}/{gpu.total_memory_mb} MB "
            f"({gpu.used_memory_mb * 100 // gpu.total_memory_mb}% used, "
            f"{gpu.free_memory_mb} MB free)"
        )
        print(f"Utilization: {gpu.utilization_percent}%")
        print(f"CUDA: {gpu.cuda_version}")
    else:
        print("GPU information not available (nvidia-smi failed)")

    print("\n=== vLLM Server ===")
    server = get_server_info()
    if server.is_running:
        print("Status: Running")
        print(f"Model: {server.model or 'Unknown'}")
        print(f"URL: {server.base_url}")
    else:
        print("Status: Not running")


__all__ = [
    "start_server",
    "stop_server",
    "wait_for_ready",
    "is_running",
    "VLLMServer",
    "get_gpu_info",
    "get_server_info",
    "print_system_info",
    "GPUInfo",
    "ServerInfo",
]
