'''
| Evaluator                                                                                          | Description                                                                                                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [AnswerExactMatchEvaluator](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator)       | Evaluates answers predicted by Haystack pipelines using ground truth labels. It checks character by character whether a predicted answer exactly matches the ground truth answer.                                                                |
| [ContextRelevanceEvaluator](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator)       | Uses an LLM to evaluate whether a generated answer can be inferred from the provided contexts.                                                                                                                                                   |
| [DeepEvalEvaluator](https://docs.haystack.deepset.ai/v2.0/docs/deepevalevaluator)                  | Use DeepEval to evaluate generative pipelines.                                                                                                                                                                                                   |
| [DocumentMAPEvaluator](https://docs.haystack.deepset.ai/docs/documentmapevaluator)                 | Evaluates documents retrieved by Haystack pipelines using ground truth labels. It checks to what extent the list of retrieved documents contains only relevant documents as specified in the ground truth labels or also non-relevant documents. |
| [DocumentMRREvaluator](https://docs.haystack.deepset.ai/docs/documentmrrevaluator)                 | Evaluates documents retrieved by Haystack pipelines using ground truth labels. It checks at what rank ground truth documents appear in the list of retrieved documents.                                                                          |
| [DocumentNDCGEvaluator](https://docs.haystack.deepset.ai/v2.7-unstable/docs/documentndcgevaluator) | Evaluates documents retrieved by Haystack pipelines using ground truth labels. It checks at what rank ground truth documents appear in the list of retrieved documents. This metric is called normalized discounted cumulative gain (NDCG).      |
| [DocumentRecallEvaluator](https://docs.haystack.deepset.ai/docs/documentrecallevaluator)           | Evaluates documents retrieved by Haystack pipelines using ground truth labels. It checks how many of the ground truth documents were retrieved.                                                                                                  |
| [FaithfulnessEvaluator](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator)               | Uses an LLM to evaluate whether a generated answer can be inferred from the provided contexts. Does not require ground truth labels.                                                                                                             |
| [LLMEvaluator](https://docs.haystack.deepset.ai/docs/llmevaluator)                                 | Uses an LLM to evaluate inputs based on a prompt containing user-defined instructions and examples.                                                                                                                                              |
| [RagasEvaluator](https://docs.haystack.deepset.ai/docs/ragasevaluator)                             | Use Ragas framework to evaluate a retrieval-augmented generative pipeline.                                                                                                                                                                       |
| [SASEvaluator](https://docs.haystack.deepset.ai/docs/sasevaluator)                                 | Evaluates answers predicted by Haystack pipelines using ground truth labels. It checks the semantic similarity of a predicted answer and the ground truth answer using a fine-tuned language model.                                              |
'''
from haystack.components.evaluators import (
    AnswerExactMatchEvaluator, ContextRelevanceEvaluator, DocumentMAPEvaluator,
    DocumentMRREvaluator, DocumentNDCGEvaluator, DocumentRecallEvaluator, FaithfulnessEvaluator,
    LLMEvaluator, SASEvaluator
)


from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator
from haystack_integrations.components.evaluators.ragas import RagasEvaluator


from abc import ABC, abstractmethod
from typing import Dict, Type


class EvaluatorFactory(ABC):
    """Abstract Factory for creating evaluators."""

    @abstractmethod
    def create_evaluator(self, **kwargs):
        """Create a specific evaluator."""
        pass


class AnswerExactMatchEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> AnswerExactMatchEvaluator:
        return AnswerExactMatchEvaluator(**kwargs)


class ContextRelevanceEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> ContextRelevanceEvaluator:
        return ContextRelevanceEvaluator(**kwargs)


class DeepEvalEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> DeepEvalEvaluator:
        return DeepEvalEvaluator(**kwargs)


class DocumentMAPEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> DocumentMAPEvaluator:
        return DocumentMAPEvaluator(**kwargs)


class DocumentMRREvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> DocumentMRREvaluator:
        return DocumentMRREvaluator(**kwargs)


class DocumentNDCGEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> DocumentNDCGEvaluator:
        return DocumentNDCGEvaluator(**kwargs)


class DocumentRecallEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> DocumentRecallEvaluator:
        return DocumentRecallEvaluator(**kwargs)


class FaithfulnessEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> FaithfulnessEvaluator:
        return FaithfulnessEvaluator(**kwargs)


class LLMEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> LLMEvaluator:
        return LLMEvaluator(**kwargs)


class RagasEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> RagasEvaluator:
        return RagasEvaluator(**kwargs)


class SASEvaluatorFactory(EvaluatorFactory):
    def create_evaluator(self, **kwargs) -> SASEvaluator:
        return SASEvaluator(**kwargs)


class EvaluatorManager:
    """Main factory class that determines which evaluator factory to use."""

    def __init__(self):
        self._factories: Dict[str, Type[EvaluatorFactory]] = {
            "exact_match": AnswerExactMatchEvaluatorFactory,
            "context_relevance": ContextRelevanceEvaluatorFactory,
            "deep_eval": DeepEvalEvaluatorFactory,
            "map": DocumentMAPEvaluatorFactory,
            "mrr": DocumentMRREvaluatorFactory,
            "ndcg": DocumentNDCGEvaluatorFactory,
            "recall": DocumentRecallEvaluatorFactory,
            "faithfulness": FaithfulnessEvaluatorFactory,
            "llm": LLMEvaluatorFactory,
            "ragas": RagasEvaluatorFactory,
            "semantic_answer_similarity": SASEvaluatorFactory
        }

    def get_evaluator(self, evaluator_type: str, **kwargs):
        """
        Get appropriate evaluator based on evaluator type.

        Args:
            evaluator_type: Type of evaluator to create
            **kwargs: Additional arguments to pass to the evaluator

        Returns:
            Appropriate evaluator instance

        Raises:
            ValueError: If evaluator type is not supported
        """
        factory_class = self._factories.get(evaluator_type.lower())

        if factory_class is None:
            raise ValueError(
                f"Unknown evaluator type: {evaluator_type}. "
                f"Available types: {', '.join(self._factories.keys())}"
            )

        factory = factory_class()
        return factory.create_evaluator(**kwargs)

    def register_evaluator(self, evaluator_name: str, factory: Type[EvaluatorFactory]):
        """
        Register a new evaluator factory.

        Args:
            evaluator_name: Name of the evaluator
            factory: Factory class for the evaluator
        """
        self._factories[evaluator_name.lower()] = factory

    def get_available_evaluators(self) -> list[str]:
        """Get list of all registered evaluator types."""
        return list(self._factories.keys())
