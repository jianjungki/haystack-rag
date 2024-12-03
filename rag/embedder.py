
from haystack.components.embedders import *


class EmbedderFactory:
    def __init__(self):
        self.embedder_map = {
            "azure.text": AzureOpenAITextEmbedder,
            "azure.document": AzureOpenAIDocumentEmbedder,
            "huggingface.document": HuggingFaceAPIDocumentEmbedder,
            "huggingface.text": HuggingFaceAPITextEmbedder,
            "openai.document": OpenAIDocumentEmbedder,
            "openai.text": OpenAITextEmbedder,
            "sentence_transformer.text": SentenceTransformersTextEmbedder,
            "sentence_transformer.document": SentenceTransformersDocumentEmbedder,
        }

    """Abstract Factory for creating embedders."""
    @ staticmethod
    def create_embedder(provider, embedder_type, embedding_model):
        embedder_map = {
            "azure.text": AzureOpenAITextEmbedder,
            "azure.document": AzureOpenAIDocumentEmbedder,
            "huggingface.document": HuggingFaceAPIDocumentEmbedder,
            "huggingface.text": HuggingFaceAPITextEmbedder,
            "openai.document": OpenAIDocumentEmbedder,
            "openai.text": OpenAITextEmbedder,
            "sentence_transformer.text": SentenceTransformersTextEmbedder,
            "sentence_transformer.document": SentenceTransformersDocumentEmbedder,
        }
        if "{}.{}".format(provider, embedder_type) in embedder_map:
            return embedder_map["{}.{}".format(provider, embedder_type)](model=embedding_model)
        else:
            raise ValueError(f"Unknown embedder type: {provider}")
