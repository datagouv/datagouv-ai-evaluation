"""
code capability: Docker-based Python execution + optional datagouv CLI.

One persistent Docker container is created per task (DockerSession), so files written
in one tool call survive for subsequent calls within the same task. This mimics the
behaviour of coding agents (Claude Code, Cursor) and sandboxed agents (E2B, Modal)
where the working directory persists for the duration of a session.

Two Docker images:
  datagouv-agent:base          — Python + requests/httpx/pandas (always available)
  datagouv-agent:datagouv-cli  — base + datagouv-client CLI (requires 'datagouv-cli' capability)

Built from a single Dockerfile with a build arg:
  docker build -t datagouv-agent:base agent_eval/experiment/agent/
  docker build --build-arg DATAGOUV_CLI=true -t datagouv-agent:datagouv-cli agent_eval/experiment/agent/

(ensure_docker_image() does this automatically at experiment startup.)

Container security flags:
  --security-opt no-new-privileges:true
  --cap-drop ALL          no Linux capability escalation
  --memory 512m / --cpus 1
  --network bridge        internet access; host loopback isolated
"""

import ast
import asyncio
import atexit
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Self

_DOCKER_IMAGE_BASE = "datagouv-agent:base"
_DOCKER_IMAGE_CLI = "datagouv-agent:datagouv-cli"
_DOCKERFILE_DIR = Path(__file__).parent

_DOCKER_RUN_FLAGS = [
    "--security-opt",
    "no-new-privileges:true",
    "--cap-drop",
    "ALL",
    "--memory",
    "512m",
    "--cpus",
    "1",
    "--network",
    "bridge",
]

_CLI_PREFIXES_BASE = (
    # Python execution
    "python ",
    "python3 ",
    # File operations
    "ls",
    "mkdir ",
    "rm ",
    "rmdir ",
    "cp ",
    "mv ",
    "cat ",
    "echo ",
    "touch ",
    "find ",
    "head ",
    "tail ",
    "diff ",
    # Text search & processing
    "grep ",
    "sort ",
    "uniq ",
    "wc ",
    "cut ",
    "sed ",
    # Archives
    "tar ",
    "unzip ",
    "gunzip ",
    # HTTP
    "curl ",
    "wget ",
    # JSON
    "jq ",
    # Package management
    "pip ",
    "pip3 ",
)
_CLI_PREFIXES_WITH_DATAGOUV = ("datagouv ",) + _CLI_PREFIXES_BASE

_TIMEOUT_SECONDS = 60


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ensure_print(code: str) -> str:
    """
    If the last non-empty line of *code* is a bare expression (not a statement),
    wrap it in print() so the result reaches stdout.
    Avoids the silent-empty-output trap of `python -c 'expr'`.
    """
    lines = code.rstrip().splitlines()
    if not lines:
        return code
    last = lines[-1]
    _STMT_STARTS = (
        "print",
        "import",
        "from",
        "def ",
        "class ",
        "if ",
        "for ",
        "while ",
        "with ",
        "try:",
        "except",
        "raise",
        "return",
        "yield",
        "#",
        "    ",
        "\t",
    )
    if any(last.lstrip().startswith(s) for s in _STMT_STARTS):
        return code
    try:
        ast.parse(last, mode="eval")
        lines[-1] = f"print({last})"
        return "\n".join(lines)
    except SyntaxError:
        return code


# ── Docker session ────────────────────────────────────────────────────────────


class DockerSession:
    """
    One persistent Docker container for the lifetime of a task.
    Use as a context manager to ensure the container is always cleaned up.
    """

    def __init__(self, container_name: str, image: str = _DOCKER_IMAGE_BASE) -> None:
        self._name = container_name
        self._image = image
        self._started = False

    def start(self) -> None:
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self._name,
                *_DOCKER_RUN_FLAGS,
                self._image,
                "sleep",
                "infinity",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start Docker container '{self._name}' "
                f"(image={self._image}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        self._started = True
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)

                def _handler(signum, frame, _prev=prev, _self=self):
                    _self.stop()
                    if callable(_prev):
                        _prev(signum, frame)

                signal.signal(sig, _handler)
            except (OSError, ValueError):
                pass  # not in main thread

    def stop(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self._name], capture_output=True)
            self._started = False

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    async def exec_async(self, entrypoint: list[str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            self._name,
            *entrypoint,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            proc.kill()
            return f"[error] Docker execution timed out after {_TIMEOUT_SECONDS}s"
        if stdout:
            return stdout.decode(errors="replace")
        if stderr:
            return f"[stderr] {stderr.decode(errors='replace')}"
        return ""


# ── Toolset factory ───────────────────────────────────────────────────────────


def code_toolset(session: DockerSession, has_datagouv_cli: bool = False) -> list:
    """
    Return [execute_python_tool, execute_cli_tool] closed over the given DockerSession.
    has_datagouv_cli controls whether the datagouv CLI is exposed in the whitelist
    and tool description.
    """
    from pydantic_ai.tools import Tool

    allowed_prefixes = (
        _CLI_PREFIXES_WITH_DATAGOUV if has_datagouv_cli else _CLI_PREFIXES_BASE
    )

    async def execute_python(code: str) -> str:
        """Execute Python code in a Docker sandbox (Linux, internet access).
        Pre-installed packages: requests, httpx, pandas.
        Files written to /tmp/ persist for the lifetime of this task (reusable across calls).

        Usage notes:
        - Only stdout is returned — always use print() to output results.
        - The last expression is NOT auto-printed; end scripts with print(result).
        - For complex data work (CSV, JSON, pandas), prefer this over execute_cli.
        - Commands time out after 60 s; split long operations across calls if needed.
        - Use absolute paths (/tmp/file.csv); avoid os.chdir()."""
        return await session.exec_async(["python", "-c", _ensure_print(code)])

    async def execute_cli(command: str) -> str:
        stripped = command.strip()
        if not any(stripped.startswith(p) for p in allowed_prefixes):
            raise ValueError(
                f"Command not in whitelist. Got: {command!r}. "
                f"Allowed prefixes: {', '.join(p.strip() for p in allowed_prefixes)}"
            )
        return await session.exec_async(["sh", "-c", command])

    execute_cli.__doc__ = "\n".join(
        filter(
            None,
            [
                "Run a shell command in the Docker sandbox (Linux, internet access).",
                "",
                "Before creating directories or files, use ls to verify the parent path exists.",
                'Always quote paths that contain spaces: ls "/tmp/my dir" not ls /tmp/my dir.',
                "Chain multiple commands with ; or && — do NOT use newlines as separators.",
                "Use absolute paths (/tmp/…) as the working directory; avoid cd.",
                "Only stdout is returned — stderr appears prefixed with [stderr].",
                "Commands time out after 60 s; break long operations across calls.",
                "For data processing, loops, or JSON parsing prefer execute_python instead.",
                "",
                "Available commands:",
                "  datagouv <subcommands> <options>  — interact with data.gouv.fr catalogue"
                if has_datagouv_cli
                else None,
                "  python/python3 <script>           — run a Python script file",
                "  curl/wget <url>                   — HTTP request",
                "  jq '<filter>' <file>              — JSON processing",
                "  grep <pattern> <file>             — search file content",
                "  find <dir> -name <pattern>        — locate files",
                "  head/tail -n N <file>             — read start/end of large files",
                "  sort / uniq / wc / cut / sed      — text processing",
                "  tar -xf <archive> -C /tmp/        — extract .tar.gz archives",
                "  unzip <file> -d /tmp/             — extract ZIP archives",
                "  gunzip <file>                     — decompress .gz files",
                "  ls / cat / cp / mv / mkdir / rm / rmdir / touch / echo / diff  — file operations",
                "  pip/pip3 install <package>        — install a Python package",
                "Files written within a task persist across calls (use /tmp/ as working dir).",
            ],
        )
    )

    return [Tool(execute_python), Tool(execute_cli)]


# ── Docker image management ───────────────────────────────────────────────────


def _image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        timeout=5,
    )
    return result.returncode == 0


def _check_docker_running() -> None:
    from agent_eval.experiment.agent.builder import CapabilityUnavailableError

    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if result.returncode != 0:
            raise CapabilityUnavailableError("Docker daemon is not running.")
    except FileNotFoundError as exc:
        raise CapabilityUnavailableError("Docker is not installed.") from exc


def _build_image(image: str, build_args: list[str]) -> None:
    import logging

    from agent_eval.experiment.agent.builder import CapabilityUnavailableError

    logger = logging.getLogger(__name__)
    logger.info("Building Docker image '%s' from %s …", image, _DOCKERFILE_DIR)
    result = subprocess.run(
        ["docker", "build", *build_args, "-t", image, str(_DOCKERFILE_DIR)],
        timeout=300,
    )
    if result.returncode != 0:
        raise CapabilityUnavailableError(
            f"Failed to build Docker image '{image}'. "
            f"Check the Dockerfile at {_DOCKERFILE_DIR}."
        )
    logger.info("Docker image '%s' built successfully.", image)


def ensure_docker_image(has_datagouv_cli: bool = False) -> None:
    """
    Ensure the required Docker image(s) exist, building them if not.
    Called once at experiment startup.
    """
    import logging

    logger = logging.getLogger(__name__)
    _check_docker_running()

    if not _image_exists(_DOCKER_IMAGE_BASE):
        _build_image(_DOCKER_IMAGE_BASE, [])
    else:
        logger.info("Docker image '%s' already exists.", _DOCKER_IMAGE_BASE)

    if has_datagouv_cli and not _image_exists(_DOCKER_IMAGE_CLI):
        _build_image(_DOCKER_IMAGE_CLI, ["--build-arg", "DATAGOUV_CLI=true"])
    elif has_datagouv_cli:
        logger.info("Docker image '%s' already exists.", _DOCKER_IMAGE_CLI)


def new_session(has_datagouv_cli: bool = False) -> DockerSession:
    """Create a DockerSession with a unique container name, using the right image."""
    image = _DOCKER_IMAGE_CLI if has_datagouv_cli else _DOCKER_IMAGE_BASE
    return DockerSession(f"datagouv-eval-{uuid.uuid4().hex[:8]}", image=image)


def check_and_create_session(has_datagouv_cli: bool = False) -> DockerSession:
    """Check Docker availability and return a new DockerSession (not yet started)."""
    _check_docker_running()
    image = _DOCKER_IMAGE_CLI if has_datagouv_cli else _DOCKER_IMAGE_BASE
    if not _image_exists(image):
        from agent_eval.experiment.agent.builder import CapabilityUnavailableError

        raise CapabilityUnavailableError(
            f"Docker image '{image}' not found. "
            "Call ensure_docker_image() at startup or build manually."
        )
    return new_session(has_datagouv_cli)
