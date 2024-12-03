from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from IPython.display import Image
from pathlib import Path
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from rag.converter import ConverterFactory
from rag.retriever import RetrieverManager
from rag.generator import GeneratorFactory, GeneratorManager, GeneratorConfig
from rag.embedder import EmbedderFactory


generator_manager = GeneratorManager()
# Register custom generator


class RAGPipeline:
    def __init__(self, embedder_type, generator_type, embedding_model, llm_model, api_key=None, base_url=None):
        self.document_store = InMemoryDocumentStore()

        embedder = EmbedderFactory.create_embedder(
            embedder_type, "document", embedding_model=embedding_model)

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
        generator = GeneratorManager().get_generator(generator_type, False, llm_config)
        print(generator)
        retriever = RetrieverManager().get_retriever(
            "embedding", self.document_store, embedder)
        converter = ConverterFactory.get_converter("doc9338.pdf")

        # Initialize the pipeline
        self.indexing = Pipeline()
        self.indexing.add_component(
            "converter", converter)
        self.indexing.add_component("cleaner", DocumentCleaner())
        self.indexing.add_component("splitter", DocumentSplitter(
            split_by="word", split_length=150, split_overlap=50))

        self.indexing.add_component("doc_embedder", embedder)
        self.indexing.add_component(
            "writer", DocumentWriter(self.document_store))

        self.indexing.connect("converter", "cleaner")
        self.indexing.connect("cleaner", "splitter")
        self.indexing.connect("splitter", "doc_embedder")
        self.indexing.connect("doc_embedder", "writer")

        self.indexing.draw("indexing.png")
        Image(filename='indexing.png')

        self.indexing.run({"converter": {"sources": [Path("doc9338.pdf")]}})

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

        text_embedder = EmbedderFactory.create_embedder(
            embedder_type, "text", embedding_model=embedding_model)
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


        # Usage
rag_pipeline = RAGPipeline(
    embedder_type="sentence_transformer",  # Add these parameters
    generator_type="openai",         # Add these parameters
    embedding_model="malenia1/ternary-weight-embedding",
    llm_model="meta-llama/llama-3.1-70b-instruct:free",
    api_key="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",
    base_url="https://openrouter.ai/api/v1")
question = "烘焙制品中允许加哪些添加剂"
print(rag_pipeline.run(question))
