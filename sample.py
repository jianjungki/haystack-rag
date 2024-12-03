from rag.rag import RAGPipeline

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
