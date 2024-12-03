'''

| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [AstraEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/astraretriever "https://docs.haystack.deepset.ai/docs/astraretriever")                                           | An embedding-based Retriever compatible with the AstraDocumentStore.                                     |
| [ChromaEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/chromaembeddingretriever "https://docs.haystack.deepset.ai/docs/chromaembeddingretriever")                      | An embedding-based Retriever compatible with the Chroma Document Store.                                  |
| [ChromaQueryTextRetriever](https://docs.haystack.deepset.ai/docs/chromaqueryretriever "https://docs.haystack.deepset.ai/docs/chromaqueryretriever")                              | A Retriever compatible with the Chroma Document Store that uses the Chroma query API.                    |
| [ElasticsearchEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/elasticsearchembeddingretriever "https://docs.haystack.deepset.ai/docs/elasticsearchembeddingretriever") | An embedding-based Retriever compatible with the Elasticsearch Document Store.                           |
| [ElasticsearchBM25Retriever](https://docs.haystack.deepset.ai/docs/elasticsearchbm25retriever "https://docs.haystack.deepset.ai/docs/elasticsearchbm25retriever")                | A keyword-based Retriever that fetches Documents matching a query from the Elasticsearch Document Store. |
| [InMemoryBM25Retriever](https://docs.haystack.deepset.ai/docs/inmemorybm25retriever "https://docs.haystack.deepset.ai/docs/inmemorybm25retriever")                               | A keyword-based Retriever compatible with the InMemoryDocumentStore.                                     |
| [InMemoryEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/inmemoryembeddingretriever "https://docs.haystack.deepset.ai/docs/inmemoryembeddingretriever")                | An embedding-based Retriever compatible with the InMemoryDocumentStore.                                  |
| [FilterRetriever](https://docs.haystack.deepset.ai/docs/filterretriever "https://docs.haystack.deepset.ai/docs/filterretriever")                                                 | A special Retriever to be used with any Document Store to get the Documents that match specific filters. |
| [MongoDBAtlasEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/mongodbatlasembeddingretriever "https://docs.haystack.deepset.ai/docs/mongodbatlasembeddingretriever")    | An embedding Retriever compatible with the MongoDB Atlas Document Store.                                 |
| [OpenSearchBM25Retriever](https://docs.haystack.deepset.ai/docs/opensearchbm25retriever "https://docs.haystack.deepset.ai/docs/opensearchbm25retriever")                         | A keyword-based Retriever that fetches Documents matching a query from an OpenSearch Document Store.     |
| [OpenSearchEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/opensearchembeddingretriever "https://docs.haystack.deepset.ai/docs/opensearchembeddingretriever")          | An embedding-based Retriever compatible with the OpenSearch Document Store.                              |
| [PgvectorEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/pgvectorembeddingretriever "https://docs.haystack.deepset.ai/docs/pgvectorembeddingretriever")                | An embedding-based Retriever compatible with the Pgvector Document Store.                                |
| [PgvectorKeywordRetriever](https://docs.haystack.deepset.ai/docs/pgvectorkeywordretriever "https://docs.haystack.deepset.ai/docs/pgvectorkeywordretriever")                      | A keyword-based Retriever that fetches documents matching a query from the Pgvector Document Store.      |
| [PineconeEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/pineconedenseretriever "https://docs.haystack.deepset.ai/docs/pineconedenseretriever")                        | An embedding-based Retriever compatible with the Pinecone Document Store.                                |
| [QdrantEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/qdrantretriever "https://docs.haystack.deepset.ai/docs/qdrantretriever")                                        | An embedding-based Retriever compatible with the Qdrant Document Store.                                  |
| [QdrantSparseEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/qdrantsparseembeddingretriever "https://docs.haystack.deepset.ai/docs/qdrantsparseembeddingretriever")    | A sparse embedding-based Retriever compatible with the Qdrant Document Store.                            |
| [QdrantHybridRetriever](https://docs.haystack.deepset.ai/docs/qdranthybridretriever "https://docs.haystack.deepset.ai/docs/qdranthybridretriever")                               | A Retriever based both on dense and sparse embeddings, compatible with the Qdrant Document Store.        |
| [SentenceWindowRetriever](https://docs.haystack.deepset.ai/docs/sentencewindowretrieval "https://docs.haystack.deepset.ai/docs/sentencewindowretrieval")                         | Retrieves neighboring sentences around relevant sentences to get the full context.                       |
| [SnowflakeTableRetriever](https://docs.haystack.deepset.ai/docs/snowflaketableretriever "https://docs.haystack.deepset.ai/docs/snowflaketableretriever")                         | Connects to a Snowflake database to execute an SQL query.                                                |
| [WeaviateBM25Retriever](https://docs.haystack.deepset.ai/docs/weaviatebm25retriever "https://docs.haystack.deepset.ai/docs/weaviatebm25retriever")                               | A keyword-based Retriever that fetches Documents matching a query from the Weaviate Document Store.      |
| [WeaviateEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/weaviateembeddingretriever "https://docs.haystack.deepset.ai/docs/weaviateembeddingretriever")                | An embedding Retriever compatible with the Weaviate Document Store.                                      |

'''
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Literal, Any
from dataclasses import dataclass

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


# Base Retriever imports
from haystack.components.retrievers import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever
)

# Integration Retriever imports
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchBM25Retriever,
    ElasticsearchEmbeddingRetriever
)
from haystack_integrations.components.retrievers.qdrant import (
    QdrantHybridRetriever,
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever
)
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.components.retrievers.weaviate import (
    WeaviateBM25Retriever,
    WeaviateEmbeddingRetriever
)
from haystack_integrations.components.retrievers.chroma import (
    ChromaQueryTextRetriever,
    ChromaEmbeddingRetriever
)
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
    OpenSearchBM25Retriever
)
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever
)


@dataclass
class RetrieverConfig:
    """Configuration for retriever creation."""
    embedding_model: Optional[Any] = None
    document_store: Optional[Any] = None


class RetrieverFactory(ABC):
    """Abstract base factory for creating retrievers."""

    @abstractmethod
    def create_embedding_retriever(self, config: RetrieverConfig):
        """Create embedding-based retriever."""
        pass

    @abstractmethod
    def create_bm25_retriever(self, config: RetrieverConfig):
        """Create BM25-based retriever."""
        pass

    def create_hybrid_retriever(self, config: RetrieverConfig):
        """Create hybrid retriever."""
        raise NotImplementedError(
            "Hybrid retriever not supported for this document store")

    def create_sparse_retriever(self, config: RetrieverConfig):
        """Create sparse retriever."""
        raise NotImplementedError(
            "Sparse retriever not supported for this document store")

    def create_query_retriever(self, config: RetrieverConfig):
        """Create query-based retriever."""
        raise NotImplementedError(
            "Query retriever not supported for this document store")

    def create_keyword_retriever(self, config: RetrieverConfig):
        """Create keyword-based retriever."""
        raise NotImplementedError(
            "Keyword retriever not supported for this document store")

    @abstractmethod
    def supports_retriever_type(self, retriever_type: str) -> bool:
        """Check if this factory supports the given retriever type."""
        pass


class InMemoryRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return InMemoryEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        return InMemoryBM25Retriever(document_store=config.document_store)

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type in ["embedding", "bm25"]


class ElasticsearchRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return ElasticsearchEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        return ElasticsearchBM25Retriever(document_store=config.document_store)

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type in ["embedding", "bm25"]


class OpenSearchRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return OpenSearchEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        return OpenSearchBM25Retriever(document_store=config.document_store)

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type in ["embedding", "bm25"]


class PineconeRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return PineconeEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        raise NotImplementedError("BM25 not supported for Pinecone")

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type == "embedding"


class QdrantRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return QdrantEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        raise NotImplementedError("BM25 not supported for Qdrant")

    def create_hybrid_retriever(self, config: RetrieverConfig):
        return QdrantHybridRetriever(
            document_store=config.document_store
        )

    def create_sparse_retriever(self, config: RetrieverConfig):
        return QdrantSparseEmbeddingRetriever(
            document_store=config.document_store
        )

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type in ["embedding", "hybrid", "sparse"]


class WeaviateRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return WeaviateEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        return WeaviateBM25Retriever(document_store=config.document_store)

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type in ["embedding", "bm25"]


class ChromaRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return ChromaEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        raise NotImplementedError("BM25 not supported for Chroma")

    def create_query_retriever(self, config: RetrieverConfig):
        return ChromaQueryTextRetriever(document_store=config.document_store)

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type in ["embedding", "query"]


class MongoDBAtlasRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return MongoDBAtlasEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        raise NotImplementedError("BM25 not supported for MongoDB Atlas")

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type == "embedding"


class AstraRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return AstraEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        raise NotImplementedError("BM25 not supported for Astra")

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type == "embedding"


class PgvectorRetrieverFactory(RetrieverFactory):
    def create_embedding_retriever(self, config: RetrieverConfig):
        return PgvectorEmbeddingRetriever(
            document_store=config.document_store
        )

    def create_bm25_retriever(self, config: RetrieverConfig):
        raise NotImplementedError("BM25 not supported for Pgvector")

    def create_keyword_retriever(self, config: RetrieverConfig):
        return PgvectorKeywordRetriever(
            document_store=config.document_store
        )

    def supports_retriever_type(self, retriever_type: str) -> bool:
        return retriever_type in ["embedding", "keyword"]


class RetrieverManager:
    """Main factory class that manages retriever creation."""

    def __init__(self):
        self._factories: Dict[str, RetrieverFactory] = {
            'InMemoryDocumentStore': InMemoryRetrieverFactory(),
            'elasticsearch': ElasticsearchRetrieverFactory(),
            'opensearch': OpenSearchRetrieverFactory(),
            'pinecone': PineconeRetrieverFactory(),
            'qdrant': QdrantRetrieverFactory(),
            'weaviate': WeaviateRetrieverFactory(),
            'chroma': ChromaRetrieverFactory(),
            'mongo_atlas': MongoDBAtlasRetrieverFactory(),
            'astra': AstraRetrieverFactory(),
            'pgvector': PgvectorRetrieverFactory(),
        }

    def get_retriever(
        self,
        retriever_type: Literal["embedding", "bm25", "hybrid", "query", "sparse", "keyword"],
        document_store,
        embedding_model: Optional[Any] = None
    ):
        """
        Create a retriever based on the document store type and retriever type.

        Args:
            retriever_type: Type of retriever to create
            document_store: Document store instance
            embedding_model: Optional embedding model for embedding-based retrievers

        Returns:
            Appropriate retriever instance

        Raises:
            ValueError: If combination of document store and retriever type is not supported
        """
        factory = self._factories.get(type(document_store).__name__)
        if not factory:
            raise ValueError(
                f"Unsupported document store type: {type(document_store).__name__}")

        if not factory.supports_retriever_type(retriever_type):
            raise ValueError(
                f"Retriever type '{retriever_type}' is not supported for "
                f"document store {type(document_store).__name__}"
            )

        config = RetrieverConfig(
            embedding_model=embedding_model,
            document_store=document_store
        )

        try:
            if retriever_type == "embedding":
                return factory.create_embedding_retriever(config)
            elif retriever_type == "bm25":
                return factory.create_bm25_retriever(config)
            elif retriever_type == "hybrid":
                return factory.create_hybrid_retriever(config)
            elif retriever_type == "sparse":
                return factory.create_sparse_retriever(config)
            elif retriever_type == "query":
                return factory.create_query_retriever(config)
            else:
                return factory.create_keyword_retriever(config)
        except NotImplementedError as e:
            raise ValueError(
                f"Retriever type '{retriever_type}' is not implemented for "
                f"document store {type(document_store)}: {str(e)}"
            )

    def register_factory(self, document_store_class, factory: RetrieverFactory):
        """Register a new factory for a document store type."""
        self._factories[document_store_class] = factory

    def get_supported_retriever_types(self, document_store) -> list[str]:
        """Get list of supported retriever types for a document store."""
        factory = self._factories.get(type(document_store))
        if not factory:
            return []

        return [
            rt for rt in ["embedding", "bm25", "hybrid", "query", "sparse", "keyword"]
            if factory.supports_retriever_type(rt)
        ]
