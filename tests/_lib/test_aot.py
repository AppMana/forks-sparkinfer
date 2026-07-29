"""The AOT invariants, pinned so a refactor cannot quietly undo them.

Every test here corresponds to a way the shipped cache could stop being used
while everything still looked green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sparkinfer._lib import aot, compiler
from sparkinfer._lib.aot_matrix import REQUIRED_KERNEL_IDS, check_coverage


def test_packaged_cache_is_excluded_from_the_package_fingerprint() -> None:
    """The cache lives under sparkinfer/ and is keyed by a hash of sparkinfer/.

    If the fingerprint included it, adding an object would change the key that
    selects it and nothing shipped could ever be found.
    """
    files = compiler._iter_fingerprint_files(Path(compiler._PACKAGE_ROOT))
    assert not any(aot.AOT_CACHE_DIRNAME in path.parts for path in files)


def test_compile_key_carries_resolved_arch_not_the_raw_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builder sets CUTE_DSL_ARCH, serving node does not; keys must still match.

    The raw variable must not appear, and the resolved value must.
    """
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_121a")
    aot.resolved_gpu_arch.cache_clear()
    compiler._compile_environment_key.cache_clear()
    entries = dict(compiler._compile_environment_key())

    assert "CUTE_DSL_ARCH" not in entries
    assert entries["__resolved_gpu_arch"] == "sm_121a"

    aot.resolved_gpu_arch.cache_clear()
    compiler._compile_environment_key.cache_clear()


def test_readonly_dirs_are_searched_before_the_writable_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged"
    (staged / "ab").mkdir(parents=True)
    monkeypatch.setenv("SPARKINFER_AOT_CACHE_DIR", str(staged))
    monkeypatch.setenv("SPARKINFER_COMPILE_CACHE_DIR", str(tmp_path / "user"))

    key = "ab" + "0" * 62
    paths = [path for path, _packaged in compiler._iter_cache_object_paths(key)]
    assert paths[0] == staged / "ab" / f"{key}.o"
    assert paths[-1] == compiler._cache_object_path(key)


def test_editable_install_with_no_packaged_cache_is_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pip install -e .` must behave exactly as it did before AOT existed.

    No packaged cache, no staged cache, nothing set: the miss guard is off, the
    lookup finds nothing to search, and a compile is free to proceed.
    """
    monkeypatch.delenv("SPARKINFER_REQUIRE_AOT", raising=False)
    monkeypatch.delenv("SPARKINFER_AOT_CACHE_DIR", raising=False)

    assert not aot.have_packaged_cache(), (
        "a source checkout must not carry sparkinfer/_aot_cache; it is a build "
        "output and is gitignored"
    )
    assert aot.readonly_cache_dirs() == ()
    assert aot.require_aot() is False

    # The lookup must still resolve to the writable per-user cache, i.e. the
    # pre-AOT behaviour, rather than erroring on the absent packaged directory.
    key = "cd" + "0" * 62
    paths = [path for path, _packaged in compiler._iter_cache_object_paths(key)]
    assert paths == [compiler._cache_object_path(key)]


def test_jit_miss_is_a_warning_not_an_error_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache is a cache. A miss compiles; it does not fail."""
    monkeypatch.delenv("SPARKINFER_REQUIRE_AOT", raising=False)
    aot.reset_aot_stats()
    with pytest.warns(aot.AotCacheMissWarning):
        aot.note_jit_compile(kernel_id="k", cache_key="0" * 64, detail="target=test")
    assert aot.aot_info()["jit_compiles"] == 1
    aot.reset_aot_stats()


def test_jit_miss_raises_under_require_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in only, for benchmark and CI runs where a compile skews the result."""
    monkeypatch.setenv("SPARKINFER_REQUIRE_AOT", "1")
    aot.reset_aot_stats()
    with pytest.raises(aot.AotCacheMiss) as excinfo:
        aot.note_jit_compile(
            kernel_id="attention.mla.sm120.decode",
            cache_key="0" * 64,
            detail="target=test",
        )
    assert "attention.mla.sm120.decode" in str(excinfo.value)
    assert "SPARKINFER_REQUIRE_AOT" in str(excinfo.value)


def test_jit_miss_warns_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPARKINFER_REQUIRE_AOT", raising=False)
    aot.reset_aot_stats()
    with pytest.warns(aot.AotCacheMissWarning):
        aot.note_jit_compile(kernel_id="k", cache_key="0" * 64, detail="target=test")
    # Once per kernel, not once per launch: a per-launch warning in a decode
    # loop is its own outage.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aot.note_jit_compile(kernel_id="k", cache_key="0" * 64, detail="target=test")
    aot.reset_aot_stats()


def test_operational_env_vars_do_not_change_the_compile_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning the AOT guard on must not invalidate the objects it guards."""
    compiler._compile_environment_key.cache_clear()
    monkeypatch.delenv("SPARKINFER_REQUIRE_AOT", raising=False)
    monkeypatch.delenv("SPARKINFER_AOT_CACHE_DIR", raising=False)
    baseline = compiler._compile_environment_key()

    compiler._compile_environment_key.cache_clear()
    monkeypatch.setenv("SPARKINFER_REQUIRE_AOT", "1")
    monkeypatch.setenv("SPARKINFER_AOT_CACHE_DIR", "/tmp/whatever")
    assert compiler._compile_environment_key() == baseline
    compiler._compile_environment_key.cache_clear()


def test_coverage_floor_rejects_an_empty_capture() -> None:
    assert check_coverage({}) != []
    assert check_coverage({k: 99 for k in REQUIRED_KERNEL_IDS}) == []


def test_target_archs_are_the_two_gated_capabilities() -> None:
    """sm_86 must not be here: gating.py accepts 12.0 and 12.1 only."""
    from sparkinfer._lib.aot_matrix import TARGET_ARCHS

    assert TARGET_ARCHS == ("sm_120a", "sm_121a")


def test_resolved_arch_prefers_the_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    aot.resolved_gpu_arch.cache_clear()
    assert aot.resolved_gpu_arch() == "sm_120a"
    aot.resolved_gpu_arch.cache_clear()


def test_build_info_absent_in_a_source_checkout_is_not_an_error() -> None:
    from sparkinfer._lib import build_info

    if os.path.isfile(build_info.build_info_path()):
        pytest.skip("this tree has a generated _build_info.json")
    assert build_info.build_info() is None
    assert build_info.check_runtime_compatibility() == []
