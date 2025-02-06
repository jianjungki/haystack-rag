from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Literal
from dataclasses import dataclass

from haystack.components.generators import (
    AzureOpenAIGenerator,
    HuggingFaceAPIGenerator,
    HuggingFaceLocalGenerator,
    OpenAIGenerator
)

from haystack.components.generators.chat import (
    AzureOpenAIChatGenerator,
    HuggingFaceAPIChatGenerator,
    HuggingFaceLocalChatGenerator,
    OpenAIChatGenerator,
)
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType


@dataclass
class GeneratorConfig:
    """Configuration for generator creation."""
    model_name: Optional[str] = None
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    streaming_callback: Optional[Callable] = None


class GeneratorFactory(ABC):
    """Abstract base factory for creating generators."""

    @abstractmethod
    def create_chat_generator(self, config: GeneratorConfig):
        """Create a chat-based generator."""
        pass

    @abstractmethod
    def create_completion_generator(self, config: GeneratorConfig):
        """Create a completion-based generator."""
        pass


class AzureGeneratorFactory(GeneratorFactory):
    def create_chat_generator(self, config: GeneratorConfig) -> AzureOpenAIChatGenerator:
        return AzureOpenAIChatGenerator(
            api_key=config.api_key,
            streaming_callback=config.streaming_callback
        )

    def create_completion_generator(self, config: GeneratorConfig) -> AzureOpenAIGenerator:

        return AzureOpenAIGenerator(
            api_key=config.api_key,
            streaming_callback=config.streaming_callback
        )


class HuggingFaceAPIGeneratorFactory(GeneratorFactory):
    def create_chat_generator(self, config: GeneratorConfig) -> HuggingFaceAPIChatGenerator:
        return HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": config.model_name},
            token=Secret.from_token(config.api_key),
            streaming_callback=config.streaming_callback
        )

    def create_completion_generator(self, config: GeneratorConfig) -> HuggingFaceAPIGenerator:
        return HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": config.model_name},
            token=Secret.from_token(config.api_key),
            streaming_callback=config.streaming_callback
        )


class HuggingFaceLocalGeneratorFactory(GeneratorFactory):
    def create_chat_generator(self, config: GeneratorConfig) -> HuggingFaceLocalChatGenerator:
        return HuggingFaceLocalChatGenerator(
            model=config.model_name,
            streaming_callback=config.streaming_callback
        )

    def create_completion_generator(self, config: GeneratorConfig) -> HuggingFaceLocalGenerator:
        return HuggingFaceLocalGenerator(
            model=config.model_name,
            streaming_callback=config.streaming_callback
        )


class OpenAIGeneratorFactory(GeneratorFactory):
    def create_chat_generator(self, config: GeneratorConfig) -> OpenAIChatGenerator:
        return OpenAIChatGenerator(
            model=config.model_name,
            api_base_url=config.api_base_url,
            api_key=Secret.from_token(config.api_key),
            streaming_callback=config.streaming_callback,
        )

    def create_completion_generator(self, config: GeneratorConfig) -> OpenAIGenerator:
        return OpenAIGenerator(
            model=config.model_name,
            api_base_url=config.api_base_url,
            api_key=Secret.from_token(config.api_key),
            streaming_callback=config.streaming_callback,
        )


class GeneratorManager:
    """Main factory class that manages generator creation."""

    def __init__(self):
        self._factories: Dict[str, GeneratorFactory] = {
            "azure": AzureGeneratorFactory(),
            "huggingface_api": HuggingFaceAPIGeneratorFactory(),
            "huggingface_local": HuggingFaceLocalGeneratorFactory(),
            "openai": OpenAIGeneratorFactory(),
        }

    def get_generator(
        self,
        generator_type: Literal["azure", "huggingface_api", "huggingface_local", "openai"],
        chat: bool = True,
        config: GeneratorConfig = None
    ):
        """
        Create a generator based on the specified type and configuration.

        Args:
            generator_type: Type of generator to create
            chat: Whether to create a chat-based generator
            **kwargs: Configuration parameters for the generator

        Returns:
            Appropriate generator instance

        Raises:
            ValueError: If generator type is not supported
        """
        factory = self._factories.get(generator_type)
        if not factory:
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available types: {', '.join(self._factories.keys())}"
            )

        if chat:
            return factory.create_chat_generator(config)
        return factory.create_completion_generator(config)

    def register_factory(self, generator_type: str, factory: GeneratorFactory):
        """Register a new generator factory."""
        self._factories[generator_type] = factory
