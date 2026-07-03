"""Tests for RepetitionPenaltyReward logging and scoring."""

import logging

from atroposlib.envs.reward_fns.repetition_penalty_reward import RepetitionPenaltyReward

# A completion long enough to pass the min_words / min_sentences gate and reach the
# per-completion summary log, but with no repeated paragraphs, so scoring returns 0.0.
NON_REPETITIVE_TEXT = (
    "The quick brown fox jumps over lazy dogs today. "
    "A totally different second sentence with many other unique words here."
)


class _MessageRenderingHandler(logging.Handler):
    """Handler that renders every record and records any formatting error.

    ``logging`` swallows exceptions raised while formatting a record (it routes
    them through ``Handler.handleError``), so a broken ``logger.info`` call does
    not surface as a normal test failure. This handler forces the same rendering
    that a real handler performs and captures the resulting exception instead.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []
        self.format_errors = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception as exc:  # pragma: no cover - only hit on regression
            self.format_errors.append(exc)


class TestRepetitionPenaltyLogging:
    """The per-completion summary log must not raise on normal completions."""

    def _compute_with_capture(self, text):
        handler = _MessageRenderingHandler()
        logger = logging.getLogger(
            "atroposlib.envs.reward_fns.repetition_penalty_reward"
        )
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            rewards = RepetitionPenaltyReward().compute([text])
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        return rewards, handler

    def test_non_repetitive_completion_logs_without_error(self):
        rewards, handler = self._compute_with_capture(NON_REPETITIVE_TEXT)

        # The summary log line must render cleanly (a stray comma used to split
        # the f-string into a msg + args pair and raise TypeError here).
        assert handler.format_errors == []
        # The reward for non-repetitive text is unaffected and stays 0.0.
        assert rewards == [0.0]

    def test_summary_is_a_single_log_message(self):
        _, handler = self._compute_with_capture(NON_REPETITIVE_TEXT)

        summary = [m for m in handler.messages if m.startswith("Word rep:")]
        assert len(summary) == 1
        # All four fields belong to one concatenated message, not separate args.
        assert "Consecutive rep:" in summary[0]
        assert "Sentence rep:" in summary[0]
        assert "penalty:" in summary[0]
