from haystack.components.embedders import *


from haystack.components.embedders import *
from typing import Type, Dict


class EmbedderFactory:
    """Factory for creating basic embedder instances."""
    @staticmethod
    def create_embedder(embedder_class: Type, embedding_model: str):
        """Create a new embedder instance."""
        return embedder_class(model=embedding_model)


class EmbedderManager:
    """Manager for handling embedder registration and creation."""
    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one manager instance exists."""
        if cls._instance is None:
            cls._instance = super(EmbedderManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the embedder registry with default embedders."""
        if not self._initialized:
            self.embedder_registry: Dict[str, Type] = {
                "azure.text": AzureOpenAITextEmbedder,
                "azure.document": AzureOpenAIDocumentEmbedder,
                "huggingface.document": HuggingFaceAPIDocumentEmbedder,
                "huggingface.text": HuggingFaceAPITextEmbedder,
                "openai.document": OpenAIDocumentEmbedder,
                "openai.text": OpenAITextEmbedder,
                "sentence_transformer.text": SentenceTransformersTextEmbedder,
                "sentence_transformer.document": SentenceTransformersDocumentEmbedder,
            }
            self.factory = EmbedderFactory()
            self._initialized = True

    def register_embedder(self, provider: str, embedder_type: str, embedder_class: Type) -> None:
        """Register a new embedder class."""
        key = f"{provider}.{embedder_type}"
        self.embedder_registry[key] = embedder_class

    def unregister_embedder(self, provider: str, embedder_type: str) -> None:
        """Remove an embedder from the registry."""
        key = f"{provider}.{embedder_type}"
        if key in self.embedder_registry:
            del self.embedder_registry[key]

    def get_embedder(self, provider: str, embedder_type: str, embedding_model: str):
        """Get an embedder instance based on provider and type."""
        key = f"{provider}.{embedder_type}"
        if key not in self.embedder_registry:
            raise ValueError(f"Unknown embedder type: {key}")

        embedder_class = self.embedder_registry[key]
        return self.factory.create_embedder(embedder_class, embedding_model)

    def list_available_embedders(self) -> list:
        """List all registered embedder types."""
        return list(self.embedder_registry.keys())
