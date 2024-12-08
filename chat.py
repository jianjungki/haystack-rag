import os
from typing import List

from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack import Pipeline
from IPython.display import Image

from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter

from rag.converter import ConverterManager
from rag.retriever import RetrieverManager
from rag.generator import GeneratorManager, GeneratorConfig
from rag.embedder import EmbedderManager


import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Initialize document store and components
    document_store = InMemoryDocumentStore()

    # Setup embedder
    embedder = EmbedderManager().get_embedder(
        provider="sentence_transformer",
        embedder_type="document",
        embedding_model="malenia1/ternary-weight-embedding"
    )

    # Setup converter and indexing pipeline first
    converter = ConverterManager().get_converter(file.path)

    indexing = Pipeline()
    indexing.add_component("converter", converter)
    indexing.add_component("cleaner", DocumentCleaner())
    indexing.add_component("splitter", DocumentSplitter(
        split_by="sentence", split_length=1))
    indexing.add_component("doc_embedder", embedder)
    indexing.add_component("writer", DocumentWriter(document_store))

    indexing.connect("converter", "cleaner")
    indexing.connect("cleaner", "splitter")
    indexing.connect("splitter", "doc_embedder")
    indexing.connect("doc_embedder", "writer")

    # Run indexing
    await cl.Message(content=f"Indexing `{file.name}`...").send()
    indexing.run({"converter": {"sources": [file.path]}})

    # Setup LLM and RAG pipeline
    llm_config = GeneratorConfig(
        api_base_url="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3.1-8b-instruct:free",
        api_key="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",
        temperature=0.5,
        max_tokens=4096
    )
    generator = GeneratorManager().get_generator("openai", False, llm_config)

    # Setup RAG components
    template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
    """
    prompt_builder = PromptBuilder(template=template)
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    text_embedder = EmbedderManager().get_embedder(
        "sentence_transformer",
        "text",
        embedding_model="malenia1/ternary-weight-embedding"
    )

    # Setup RAG pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", generator)

    rag_pipeline.connect("text_embedder.embedding",
                         "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    # Store pipeline in session
    cl.user_session.set("chain", rag_pipeline)

    await cl.Message(
        content=f"Ready! You can now ask questions about `{file.name}`.").send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    res = chain.run(
        {
            "text_embedder": {"text": message.content},
            "prompt_builder": {"question": message.content}
        }
    )

    await cl.Message(content=res["llm"]["replies"][0]).send()
