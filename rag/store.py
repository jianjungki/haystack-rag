'''
InMemoryDocumentStore
AstraDocumentStore
AzureAISearchDocumentStore
ChromaDocumentStore
ElasticsearchDocumentStore
MilvusDocumentStore
MongoDBAtlasDocumentStore
Neo4jDocumentStore
OpenSearchDocumentStore
PgvectorDocumentStore
PineconeDocumentStore
QdrantDocumentStore
WeaviateDocumentStore
'''

# Document Store imports
from haystack.document_stores.in_memory import InMemoryDocumentStore


# Integration Document Store imports
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.document_stores.astra import AstraDocumentStore
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class StoreManager:
    """Manages different document store implementations"""

    @staticmethod
    def get_store(store_type: str, **kwargs) -> any:
        """
        Factory method to create and return a document store instance

        Args:
            store_type: Type of document store to create
            **kwargs: Additional arguments to pass to the document store constructor

        Returns:
            Document store instance
        """
        stores = {
            "memory": InMemoryDocumentStore,
            "qdrant": QdrantDocumentStore,
            "chroma": ChromaDocumentStore,
            "astra": AstraDocumentStore,
            "pgvector": PgvectorDocumentStore,
            "elasticsearch": ElasticsearchDocumentStore,
            "opensearch": OpenSearchDocumentStore,
            "pinecone": PineconeDocumentStore,
            "weaviate": WeaviateDocumentStore,
            "mongodb": MongoDBAtlasDocumentStore
        }

        if store_type.lower() not in stores:
            raise ValueError(f"Unsupported store type: {store_type}. "
                             f"Supported types are: {', '.join(stores.keys())}")

        return stores[store_type.lower()](**kwargs)
