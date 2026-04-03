from rlm.video import (
    CallableAudioEventExtractor,
    CallableOCRExtractor,
    CallableSpeechRecognizer,
    CallableVisualSummarizer,
    TimeSpan,
    VideoMemoryBuilder,
)
from rlm.video.adapters import AudioEventExtractor, OCRExtractor, SpeechRecognizer, VisualSummarizer
from rlm.video.types import AudioEvent, OCRSpan, SpeechSpan, VisualSummarySpan


def test_callable_adapters_work_with_memory_builder():
    speech = CallableSpeechRecognizer(
        lambda path: [
            SpeechSpan(text=f"Transcript from {path}", time_span=TimeSpan(0.0, 3.0)),
        ]
    )
    visual = CallableVisualSummarizer(
        lambda path, spans: [
            VisualSummarySpan(
                summary=f"Visual summary for {path}",
                time_span=spans[0],
                granularity="scene",
                tags=["demo"],
            )
        ]
    )
    ocr = CallableOCRExtractor(
        lambda path: [OCRSpan(text="TITLE CARD", time_span=TimeSpan(0.0, 1.0))]
    )
    audio = CallableAudioEventExtractor(
        lambda path: [AudioEvent(label="applause", time_span=TimeSpan(2.0, 4.0))]
    )

    assert isinstance(speech, SpeechRecognizer)
    assert isinstance(visual, VisualSummarizer)
    assert isinstance(ocr, OCRExtractor)
    assert isinstance(audio, AudioEventExtractor)

    builder = VideoMemoryBuilder(
        speech_recognizer=speech,
        visual_summarizer=visual,
        ocr_extractor=ocr,
        audio_extractor=audio,
        scene_duration_seconds=10.0,
        segment_duration_seconds=5.0,
        clip_duration_seconds=2.5,
    )
    memory = builder.build(video_path="sample.mp4", duration_seconds=10.0, video_id="sample")

    scene = memory.get_node("sample_scene_001")
    assert scene.speech_spans
    assert scene.ocr_spans
    assert scene.audio_events
    assert "Visual summary" in scene.visual_summary
