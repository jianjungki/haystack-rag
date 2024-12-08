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
question = "输出内容的标题, 著作者, 文号以及日期"
print(rag_pipeline.run(question))
