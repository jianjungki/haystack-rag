from rag.rag import RAGPipeline

import logging
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)

# to enable tracing/logging content (inputs/outputs)
tracing.tracer.is_content_tracing_enabled = True
tracing.enable_tracing(LoggingTracer())

# Usage
rag_pipeline = RAGPipeline(
    embedding_provider="sentence_transformer",  # Add these parameters
    llm_provider="openai",         # Add these parameters
    embedding_model="malenia1/ternary-weight-embedding",
    llm_model="qwen/qwen-2.5-7b-instruct",
    api_key="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",
    base_url="https://openrouter.ai/api/v1")
question = "孤独症应该如何评估"
print(rag_pipeline.run(question))
