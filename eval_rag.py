import sys
from unittest.mock import MagicMock
# Mock VertexAI module to bypass Ragas internal import error
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

from backend.evaluator import (
    FILE_TEST_QUESTIONS,
    DEFAULT_TEST_QUESTIONS,
    get_default_questions,
    run_ragas_evaluation
)
