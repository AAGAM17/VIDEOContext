"""Provider selection, device resolution, and the no-op engine.

Speech is the one stage whose provider can legitimately be missing on a working install, so
the interesting behaviour is not transcription — it is what happens when transcription cannot
run. Every test here is about degrading in a way the document can record honestly.

Nothing here loads a model or touches the network; :mod:`tests.integration` covers the real
Whisper path.
"""

from __future__ import annotations

import ctranslate2
import pytest

from videocontext.config import ASRConfig
from videocontext.errors import DependencyMissingError
from videocontext.interfaces import FrameContext
from videocontext.processing.asr import FALLBACK_ORDER, resolve_engine
from videocontext.processing.asr import faster_whisper as fw
from videocontext.processing.asr.null import NullASR


def ctx(duration: float = 62.4, language: str | None = "en") -> FrameContext:
    return FrameContext(duration=duration, fps=30.0, width=1280, height=720, language=language)


@pytest.fixture
def no_whisper(monkeypatch):
    """Simulate a machine where the speech extra was never installed."""
    monkeypatch.setattr(fw, "installed", lambda: False)


class TestNullEngine:
    def test_it_is_always_available(self):
        assert NullASR().available() is True

    def test_it_returns_an_empty_transcript_and_says_so(self):
        out = NullASR().transcribe("ignored.wav", ctx())
        assert out.utterances == []
        assert out.model is None
        assert out.warnings == ["transcription disabled (null ASR engine)"]

    def test_it_carries_the_context_forward(self):
        out = NullASR().transcribe("ignored.wav", ctx(duration=12.5, language="hi"))
        assert out.duration_s == 12.5
        assert out.language == "hi"

    def test_it_is_local(self):
        assert NullASR().remote is False


class TestResolveDevice:
    def test_an_explicit_device_is_respected(self):
        assert fw.resolve_device("cpu") == "cpu"
        assert fw.resolve_device("cuda") == "cuda"

    def test_auto_picks_cuda_when_a_gpu_is_present(self, monkeypatch):
        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 2)
        assert fw.resolve_device("auto") == "cuda"

    def test_auto_picks_cpu_when_no_gpu_is_present(self, monkeypatch):
        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
        assert fw.resolve_device("auto") == "cpu"

    def test_a_probe_that_raises_means_no_gpu(self, monkeypatch):
        # GPU is never mandatory (brief §21); a broken CUDA install must not fail the run.
        def boom():
            raise RuntimeError("libcudart.so not found")

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", boom)
        assert fw.resolve_device("auto") == "cpu"


class TestResolveComputeType:
    def test_an_explicit_type_is_respected(self):
        assert fw.resolve_compute_type("float32", "cpu") == "float32"

    def test_auto_prefers_int8_on_a_cpu_that_supports_it(self, monkeypatch):
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda _d: {"float32", "int8"}
        )
        assert fw.resolve_compute_type("auto", "cpu") == "int8"

    def test_auto_falls_back_to_float32_when_quantisation_is_unsupported(self, monkeypatch):
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda _d: {"float32"}
        )
        assert fw.resolve_compute_type("auto", "cpu") == "float32"

    def test_auto_prefers_float16_on_a_gpu(self, monkeypatch):
        monkeypatch.setattr(
            ctranslate2,
            "get_supported_compute_types",
            lambda _d: {"float32", "float16", "int8"},
        )
        assert fw.resolve_compute_type("auto", "cuda") == "float16"

    def test_nothing_recognised_defers_to_the_library(self, monkeypatch):
        monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda _d: {"bf16"})
        assert fw.resolve_compute_type("auto", "cpu") == "default"

    def test_a_probe_that_raises_defers_to_the_library(self, monkeypatch):
        def boom(_device):
            raise RuntimeError("unsupported device")

        monkeypatch.setattr(ctranslate2, "get_supported_compute_types", boom)
        assert fw.resolve_compute_type("auto", "cpu") == "default"

    def test_this_machine_resolves_to_something_ctranslate2_accepts(self):
        # Not a mock: the resolution has to be right on the machine running the tests, which
        # is the whole reason it asks instead of hardcoding.
        device = fw.resolve_device("auto")
        resolved = fw.resolve_compute_type("auto", device)
        assert resolved in ctranslate2.get_supported_compute_types(device) | {"default"}


class TestWhisperConfidence:
    def test_mean_log_prob_becomes_a_probability(self):
        # exp(-0.105) ≈ 0.9: the geometric mean of the token probabilities.
        assert fw._confidence(-0.105) == pytest.approx(0.9004, abs=1e-4)

    def test_certainty_is_one(self):
        assert fw._confidence(0.0) == 1.0

    def test_a_positive_log_prob_is_clamped(self):
        assert fw._confidence(0.5) == 1.0

    def test_unreported_stays_unreported(self):
        assert fw._confidence(None) is None

    def test_a_very_low_score_stays_in_range(self):
        assert 0.0 <= fw._confidence(-800.0) <= 1.0


class TestWhisperWithoutTheDependency:
    def test_it_reports_itself_unavailable(self, no_whisper):
        assert fw.FasterWhisperASR().available() is False

    def test_transcribe_raises_with_an_actionable_hint(self, no_whisper):
        with pytest.raises(DependencyMissingError) as excinfo:
            fw.FasterWhisperASR().transcribe("audio.wav", ctx())
        # The hint has to name the way out, not just the problem (errors.py docstring).
        assert "VIDEO_CONTEXT_ASR_PROVIDER" in (excinfo.value.hint or "")

    def test_version_says_unavailable_rather_than_guessing(self, no_whisper):
        assert fw.FasterWhisperASR().version == "unavailable"


class TestResolveEngine:
    def test_the_configured_provider_wins_when_available(self):
        engine, warnings = resolve_engine(ASRConfig(provider="null"))
        assert engine.name == "null"
        assert warnings == []

    def test_disabled_returns_the_no_op_without_consulting_the_chain(self):
        # A disabled stage that quietly resolved to Whisper would be the opposite of the ask.
        engine, warnings = resolve_engine(ASRConfig(enabled=False))
        assert engine.name == "null"
        assert warnings == []

    def test_a_missing_model_falls_back_to_subtitles(self, no_whisper):
        engine, warnings = resolve_engine(ASRConfig(provider="faster-whisper"))
        assert engine.name == "subtitles"
        assert len(warnings) == 1
        # The document must say what was asked for and what ran instead.
        assert "faster-whisper" in warnings[0] and "subtitles" in warnings[0]

    def test_subtitles_can_be_switched_off_leaving_only_the_no_op(self, no_whisper):
        engine, warnings = resolve_engine(
            ASRConfig(provider="faster-whisper", fallback_to_subtitles=False)
        )
        assert engine.name == "null"
        assert warnings and "null" in warnings[0]

    def test_an_unknown_provider_degrades_instead_of_raising(self):
        engine, warnings = resolve_engine(ASRConfig(provider="typo"))
        assert engine.name in FALLBACK_ORDER
        assert "unknown asr provider" in warnings[0]

    def test_a_substitution_warning_is_a_single_line(self):
        # These land in a JSON document's warnings list, not on a terminal, so the error's
        # multi-line hint must not be spliced in.
        _, warnings = resolve_engine(ASRConfig(provider="typo"))
        assert "\n" not in warnings[0]

    def test_the_video_path_reaches_the_subtitle_engine(self, no_whisper, tmp_path):
        track = tmp_path / "demo.srt"
        track.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n")
        engine, _ = resolve_engine(ASRConfig(), source=track)
        assert engine.name == "subtitles"
        assert engine.source == track

    def test_it_never_returns_an_unavailable_engine(self, no_whisper):
        for cfg in (
            ASRConfig(),
            ASRConfig(provider="typo"),
            ASRConfig(fallback_to_subtitles=False),
            ASRConfig(provider="null"),
        ):
            engine, _ = resolve_engine(cfg)
            assert engine.available() is True
