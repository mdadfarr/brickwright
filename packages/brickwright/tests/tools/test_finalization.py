"""Tests for the finalization tool."""

import sys
import types

import pytest

from brickwright.tools import finalize, rendering
from brickwright.tools.validation import REQUIRED_CHECKS_ENV


@pytest.fixture
def required_check(monkeypatch):
    """Install ``fn`` as the only entry in ``BRICKWRIGHT_REQUIRED_CHECKS``."""

    def _install(fn):
        module = types.ModuleType("fake_required_checks")
        module.check = fn
        monkeypatch.setitem(sys.modules, "fake_required_checks", module)
        monkeypatch.setenv(REQUIRED_CHECKS_ENV, "fake_required_checks:check")

    return _install


class TestRequiredChecks:
    @pytest.fixture(autouse=True)
    def _no_render(self, monkeypatch):
        monkeypatch.setattr(rendering, "render_available", lambda: False)

    def test_failing_required_check_blocks_finalize(
        self, make_record, in_process, valid_build, required_check
    ):
        required_check(lambda ctx: ctx.check("only_owned_parts", False, "1 over"))
        rec = make_record(valid_build)
        report = finalize(rec)
        assert not report.ok
        assert report.validation is not None
        assert report.validation.failed_checks == ["only_owned_parts"]
        assert not rec.read_meta().finalized
        assert not rec.ldr_path.exists()

    def test_passing_required_check_allows_finalize(
        self, make_record, in_process, valid_build, required_check
    ):
        required_check(lambda ctx: ctx.check("only_owned_parts", True))
        rec = make_record(valid_build)
        report = finalize(rec)
        assert report.ok, report.error
        assert "only_owned_parts" in [c.name for c in report.validation.checks]
        assert rec.read_meta().finalized

    def test_build_cannot_replace_a_required_check(
        self, record, in_process, required_check
    ):
        required_check(lambda ctx: ctx.check("only_owned_parts", False, "1 over"))
        record.write_build("""import bricklayer as bl
from brickwright import TestContext
def create_build():
    m = bl.LegoBuild()
    m.add(bl.brick(4, 2, color=bl.Color.RED))
    return m
def run_tests():
    ctx = TestContext(create_build())
    ctx.check("only_owned_parts", True)
    return ctx.report()
""")
        report = finalize(record)
        assert not report.ok
        owned = [c for c in report.validation.checks if c.name == "only_owned_parts"]
        assert len(owned) == 1 and not owned[0].passed

    @pytest.mark.parametrize(
        "target", ["no_such_module_xyz:check", "fake_required_checks:missing"]
    )
    def test_unloadable_required_check_fails_closed(
        self, make_record, in_process, valid_build, required_check, monkeypatch, target
    ):
        required_check(lambda ctx: ctx.check("only_owned_parts", True))
        monkeypatch.setenv(REQUIRED_CHECKS_ENV, target)
        rec = make_record(valid_build)
        report = finalize(rec)
        assert not report.ok
        assert report.validation.failed_checks == [target]

    def test_required_check_that_records_nothing_fails_closed(
        self, make_record, in_process, valid_build, required_check
    ):
        required_check(lambda ctx: None)
        rec = make_record(valid_build)
        report = finalize(rec)
        assert not report.ok
        assert report.validation.failed_checks == ["fake_required_checks:check"]


class TestFinalize:
    def test_rejects_collision_even_when_custom_tests_omit_it(
        self,
        record,
        in_process,
        monkeypatch,
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: False)
        record.write_build("""import bricklayer as bl
def create_build():
    m = bl.LegoBuild()
    m.add(bl.Element("3001", bl.Color.RED))
    m.add(bl.Element("3001", bl.Color.BLUE))
    return m
def run_tests():
    return []
""")
        result = finalize(record)
        assert not result.ok
        assert result.validation is not None
        assert "no_collisions" in result.validation.failed_checks
        assert not record.read_meta().finalized
        assert not record.ldr_path.exists()

    def test_edit_during_export_rejects_the_superseded_result(
        self, record, in_process, floating_build, monkeypatch
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: False)
        original_source = record.read_build()
        export = rendering.build_ldraw

        def concurrent_edit(rec, model=None):
            path = export(rec, model)
            rec.build_path.write_text(floating_build)
            return path

        monkeypatch.setattr(rendering, "build_ldraw", concurrent_edit)
        report = finalize(record)
        assert not report.ok
        assert report.validation is not None
        assert report.validation.error_type == "SourceChanged"
        assert record.snapshots[-1].read_build() == original_source
        assert not record.read_meta().finalized
        assert not record.ldr_path.exists()

    def test_revalidates_source_edited_outside_the_tools(
        self, record, in_process, floating_build, monkeypatch
    ):
        from brickwright.tools import validate

        monkeypatch.setattr(rendering, "render_available", lambda: False)
        validate(record)
        record.build_path.write_text(floating_build)
        report = finalize(record)
        assert not report.ok
        assert report.validation is not None and not report.validation.ok
        assert not record.read_meta().finalized
        assert not record.ldr_path.exists()

    def test_validation_and_export_share_one_build(
        self, record, in_process, monkeypatch
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: False)
        source = record.read_build().replace(
            "def create_build():",
            "builds = 0\n\ndef create_build():\n"
            "    global builds\n    builds += 1\n    assert builds == 1\n",
        )
        record.write_build(source)
        report = finalize(record)
        assert report.ok, report.error
        meta = record.read_meta()
        assert meta.finalized
        assert meta.exported_source_sha256 == meta.validated_source_sha256
        assert meta.exported_source_sha256 == record.source_sha256

    def test_failed_export_clears_previous_finalization(
        self, record, in_process, monkeypatch
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: False)
        assert finalize(record).ok
        monkeypatch.setattr(rendering, "render_available", lambda: True)

        def fail(*args, **kwargs):
            raise RuntimeError("render unavailable")

        monkeypatch.setattr(rendering, "build_mesh", fail)
        report = finalize(record)
        assert not report.ok
        assert not record.read_meta().finalized
        assert record.read_meta().exported_source_sha256 is None
        assert not record.ldr_path.exists()

    def test_renders_the_model_to_ldraw(
        self, make_record, in_process, valid_build, monkeypatch
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: False)
        rec = make_record(valid_build)
        report = finalize(rec)
        assert report.ok is True
        assert report.error is None
        assert rec.ldr_path.is_file()
        assert rec.ldr_path.read_text().strip()  # not empty
        assert (rec.out_dir / "diagnostics" / "build.ldr").is_file()
        assert not (rec.out_dir / "diagnostics" / "build.glb").exists()

    def test_renders_a_png_when_a_library_is_available(
        self, make_record, in_process, valid_build, fake_render_pipeline, monkeypatch
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: True)
        monkeypatch.setattr(
            rendering, "_convert", lambda ldr, glb: glb.write_bytes(b"mesh")
        )
        rec = make_record(valid_build)
        report = finalize(rec)
        assert report.ok is True
        assert report.error is None
        assert rec.ldr_path.is_file()
        assert rec.render_path.is_file()
        assert rec.render_path.stat().st_size > 0
        assert (rec.out_dir / "diagnostics" / "build.ldr").is_file()
        assert (rec.out_dir / "diagnostics" / "build.glb").is_file()

    def test_skips_rendering_when_no_library_is_available(
        self, make_record, in_process, valid_build, monkeypatch
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: False)
        rec = make_record(valid_build)
        report = finalize(rec)
        assert report.ok is True
        assert rec.ldr_path.is_file()
        assert not rec.render_path.is_file()

    def test_build_error_is_reported_and_nothing_is_rendered(
        self, make_record, in_process, raising_build, monkeypatch
    ):
        monkeypatch.setattr(rendering, "render_available", lambda: True)
        rec = make_record(raising_build)
        report = finalize(rec)
        assert report.ok is False
        assert report.error is not None
        assert "RuntimeError" in report.error
        assert "boom" in report.error
        assert not rec.ldr_path.is_file()
        assert not rec.render_path.is_file()

    @pytest.mark.parametrize(
        "exc", [TimeoutError("timed out"), RuntimeError("exited with code 1")]
    )
    def test_wraps_a_subprocess_failure(self, record, subprocess_raises, exc):
        # finalize turns either subprocess failure into a failed report
        subprocess_raises(exc)
        report = finalize(record)
        assert report.ok is False
        assert report.error is not None
        assert "Finalization failed" in report.error
        assert str(exc) in report.error
