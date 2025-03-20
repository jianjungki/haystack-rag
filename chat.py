from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack import Pipeline

from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter

from rag.converter import ConverterManager
from rag.retriever import RetrieverManager
from rag.store import StoreManager
from rag.generator import GeneratorManager, GeneratorConfig
from rag.embedder import EmbedderManager

from rag.evaluator import EvaluatorManager


import chainlit as cl

import nltk

nltk.download('punkt_tab')


@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}").send()
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a file to begin!",
            accept=["text/plain",
                    "application/pdf",
                    "text/html"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Initialize document store and components
    document_store = StoreManager().get_store("memory")

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
        split_by="period", split_length=1))
    indexing.add_component("doc_embedder", embedder)
    indexing.add_component("writer", DocumentWriter(document_store))

    indexing.connect("converter", "cleaner")
    indexing.connect("cleaner", "splitter")
    indexing.connect("splitter", "doc_embedder")
    indexing.connect("doc_embedder", "writer")

    # Run indexing
    await cl.Message(content=f"Indexing `{file.name}`...").send()
    print(f"Indexing `{file.path}`...")
    indexing.run({"converter": {"sources": [file.path]}})

    # Setup LLM and RAG pipeline
    llm_config = GeneratorConfig(
        api_base_url="https://api.siliconflow.cn/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_key="sk-aqilqiwnnwgdbzvqyriehstlzluawydzuimomxwvmlqfgfzk",
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
    retriever = RetrieverManager().get_retriever(
        "embedding",
        document_store=document_store, embedding_model="malenia1/ternary-weight-embedding")

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

    eval_pipeline = Pipeline()
    eval_pipeline.add_component(
        "doc_mrr_evaluator", EvaluatorManager().get_evaluator("mmr"))
    eval_pipeline.add_component(
        "faithfulness", EvaluatorManager().get_evaluator("faithfulness"))
    eval_pipeline.add_component("sas_evaluator",
                                EvaluatorManager().get_evaluator("semantic_answer_similarity",
                                                                 model="sentence-transformers/all-MiniLM-L6-v2"))

    rag_pipeline.connect("text_embedder.embedding",
                         "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm", "evaluator")

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


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
