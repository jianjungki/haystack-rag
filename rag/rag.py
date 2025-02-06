from .converter import ConverterManager
from .retriever import RetrieverManager
from .generator import GeneratorManager, GeneratorConfig
from .embedder import EmbedderManager

from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from IPython.display import Image
from pathlib import Path
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


class RAGPipeline:
    def __init__(self, embedding_provider, embedding_model, llm_provider, llm_model, api_key=None, base_url=None):
        self.document_store = InMemoryDocumentStore()

        embedder = EmbedderManager().get_embedder(
            provider=embedding_provider, embedder_type="document", embedding_model=embedding_model)

        llm_config = GeneratorConfig(
            model_name=llm_model,
            api_key=api_key,
            temperature=0.5,
            max_tokens=4096
        )
        if base_url is not None:
            llm_config = GeneratorConfig(
                api_base_url=base_url,
                model_name=llm_model,
                api_key=api_key,
                temperature=0.5,
                max_tokens=4096
            )
        generator = GeneratorManager().get_generator(llm_provider, False, llm_config)
        print(generator)
        retriever = RetrieverManager().get_retriever(
            "embedding", self.document_store, embedder)
        converter = ConverterManager().get_converter("new.pdf")

        # Initialize the pipeline
        self.indexing = Pipeline()
        self.indexing.add_component(
            "converter", converter)
        self.indexing.add_component("cleaner", DocumentCleaner())
        self.indexing.add_component("splitter", DocumentSplitter(
            split_by="sentence", split_length=1))

        self.indexing.add_component("doc_embedder", embedder)
        self.indexing.add_component(
            "writer", DocumentWriter(self.document_store))

        self.indexing.connect("converter", "cleaner")
        self.indexing.connect("cleaner", "splitter")
        self.indexing.connect("splitter", "doc_embedder")
        self.indexing.connect("doc_embedder", "writer")

        self.indexing.draw("indexing.png")
        Image(filename='indexing.png')

        self.indexing.run(
            {"converter": {"sources": [Path("new.pdf")]}})

        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
        self.prompt_builder = PromptBuilder(template=template)

        self.generator = generator

        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store)

        text_embedder = EmbedderManager().get_embedder(
            embedding_provider, "text", embedding_model=embedding_model)
        self.text_embedder = text_embedder

        self.basic_rag_pipeline = Pipeline()
        self.basic_rag_pipeline.add_component(
            "text_embedder", self.text_embedder)
        self.basic_rag_pipeline.add_component("retriever", retriever)
        self.basic_rag_pipeline.add_component(
            "prompt_builder", self.prompt_builder)
        self.basic_rag_pipeline.add_component("llm", generator)

        self.basic_rag_pipeline.connect("text_embedder.embedding",
                                        "retriever.query_embedding")
        self.basic_rag_pipeline.connect(
            "retriever", "prompt_builder.documents")
        self.basic_rag_pipeline.connect("prompt_builder", "llm")

    def run(self, question):
        response = self.basic_rag_pipeline.run(
            {"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
        return response["llm"]["replies"][0]


class RAGChatPipeline:
    def __init__(self, embedding_provider, embedding_model, llm_provider, llm_model, api_key=None, base_url=None):
        self.document_store = InMemoryDocumentStore()

        embedder = EmbedderManager().get_embedder(
            provider=embedding_provider, embedder_type="document", embedding_model=embedding_model)

        llm_config = GeneratorConfig(
            model_name=llm_model,
            api_key=api_key,
            temperature=0.5,
            max_tokens=4096
        )
        if base_url is not None:
            llm_config = GeneratorConfig(
                api_base_url=base_url,
                model_name=llm_model,
                api_key=api_key,
                temperature=0.5,
                max_tokens=4096
            )
        generator = GeneratorManager().get_generator(llm_provider, False, llm_config)
        print(generator)
        retriever = RetrieverManager().get_retriever(
            "embedding", self.document_store, embedder)
        converter = ConverterManager().get_converter("new.pdf")

        # Initialize the pipeline
        self.indexing = Pipeline()
        self.indexing.add_component(
            "converter", converter)
        self.indexing.add_component("cleaner", DocumentCleaner())
        self.indexing.add_component("splitter", DocumentSplitter(
            split_by="sentence", split_length=1))

        self.indexing.add_component("doc_embedder", embedder)
        self.indexing.add_component(
            "writer", DocumentWriter(self.document_store))

        self.indexing.connect("converter", "cleaner")
        self.indexing.connect("cleaner", "splitter")
        self.indexing.connect("splitter", "doc_embedder")
        self.indexing.connect("doc_embedder", "writer")

        self.indexing.draw("indexing.png")
        Image(filename='indexing.png')

        self.indexing.run(
            {"converter": {"sources": [Path("new.pdf")]}})

        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
        self.prompt_builder = PromptBuilder(template=template)

        self.generator = generator

        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store)

        text_embedder = EmbedderManager().get_embedder(
            embedding_provider, "text", embedding_model=embedding_model)
        self.text_embedder = text_embedder

        self.basic_rag_pipeline = Pipeline()
        self.basic_rag_pipeline.add_component(
            "text_embedder", self.text_embedder)
        self.basic_rag_pipeline.add_component("retriever", retriever)
        self.basic_rag_pipeline.add_component(
            "prompt_builder", self.prompt_builder)
        self.basic_rag_pipeline.add_component("llm", generator)

        self.basic_rag_pipeline.connect("text_embedder.embedding",
                                        "retriever.query_embedding")
        self.basic_rag_pipeline.connect(
            "retriever", "prompt_builder.documents")
        self.basic_rag_pipeline.connect("prompt_builder", "llm")

    def run(self, question):
        response = self.basic_rag_pipeline.run(
            {"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
        return response["llm"]["replies"][0]
