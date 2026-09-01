from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_PACKAGE = "sparkinfer.comm.pcie"
RUNTIME_CUDA_SOURCES = {
    "pcie_dcp_a2a.cu",
    "pcie_dma.cu",
    "pcie_oneshot.cu",
    "pcie_twoshot.cu",
}


def test_runtime_cuda_sources_are_in_package_data() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"]["package-data"]

    assert package_data[PCIE_PACKAGE] == ["*.cu"]
    assert {
        path.name for path in (ROOT / "sparkinfer" / "comm" / "pcie").glob("*.cu")
    } == RUNTIME_CUDA_SOURCES


def test_release_version_is_supplied_by_the_build() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in config["project"]["dynamic"]
    assert "version" not in config["project"]

    setup_source = (ROOT / "setup.py").read_text()
    assert 'os.getenv("SPARKINFER_VERSION", "1.0.1")' in setup_source


def test_local_extension_builds_prefer_sccache() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    assert 'shutil.which("sccache")' in setup_source
    assert 'os.environ.setdefault("PYTORCH_NVCC"' in setup_source
    assert "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES" in setup_source


def test_isolated_build_guard_does_not_compile_the_extensions_twice() -> None:
    workflow = (ROOT / ".github" / "workflows" / "wheels.yaml").read_text()
    smoke = workflow.split(
        "- name: Smoke test the isolated PEP 517 build", 1
    )[1].split("- name: Assert the wheel is AOT", 1)[0]

    assert "python -m build --sdist" in smoke
    assert "python -m build --wheel" not in smoke
    assert '"extensions_built"' in smoke


def test_container_wheels_use_a_restorable_local_sccache() -> None:
    workflow = (ROOT / ".github" / "workflows" / "wheels.yaml").read_text()

    assert workflow.count("uses: actions/cache@v4") >= 2
    assert workflow.count('-e SCCACHE_DIR=/sccache-cache') >= 2
    assert workflow.count('${RUNNER_TEMP}/sccache:/sccache-cache') >= 2
