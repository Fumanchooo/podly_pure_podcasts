from unittest.mock import MagicMock

import pytest

from app.models import Post, ProcessingJob, TranscriptSegment
from podcast_processor.ad_classifier import ClassificationIncompleteError
from podcast_processor.podcast_processor import PodcastProcessor
from shared.test_utils import create_standard_test_config


def test_incomplete_classification_stops_before_audio_processing() -> None:
    """An unclassified chunk must not lead to cut audio and a finalized episode."""
    transcription_manager = MagicMock()
    transcription_manager.transcribe.return_value = [
        TranscriptSegment(
            post_id=1, sequence_num=0, start_time=0.0, end_time=5.0, text="hello"
        )
    ]
    audio_processor = MagicMock()

    # Bypass __init__ to avoid beartype constructor checks on MagicMock doubles.
    processor = object.__new__(PodcastProcessor)
    processor.config = create_standard_test_config()
    processor.logger = MagicMock()
    processor.transcription_manager = transcription_manager
    processor.audio_processor = audio_processor
    processor.status_manager = MagicMock()
    # The real _classify_ad_segments runs; only the classifier itself is a double.
    processor.ad_classifier = MagicMock()
    processor.ad_classifier.classify.side_effect = ClassificationIncompleteError(
        "1 chunk(s) without a usable answer"
    )
    processor.get_user_prompt_template = MagicMock()
    processor.get_system_prompt = MagicMock(return_value="system prompt")
    processor._finalize_processing = MagicMock()

    post = Post(
        id=1,
        feed_id=1,
        guid="test-guid",
        title="Test Episode",
        download_url="https://example.com/test.mp3",
        unprocessed_audio_path="/tmp/input.mp3",
    )
    job = ProcessingJob(id="job-1", post_guid="test-guid", status="running")

    with pytest.raises(ClassificationIncompleteError):
        processor._perform_llm_based_processing(post, job, "/tmp/output.mp3")

    processor.ad_classifier.classify.assert_called_once()
    audio_processor.process_audio.assert_not_called()
    processor._finalize_processing.assert_not_called()
