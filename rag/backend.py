from haystack import Document
from haystack import EmbeddingRetriever, FARMReader


class HaystackInitializer:
    def __init__(self, document_store, embedding_model_name="deepset/sentence_bert", llm_model_name="deepset/roberta-base-squad2", use_gpu=True):
        self.document_store = document_store
        self.embedding_model = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=embedding_model_name,
            model_format="sentence_transformers",
            use_gpu=use_gpu
        )
        self.llm_model = FARMReader(
            model_name_or_path=llm_model_name,
            use_gpu=use_gpu
        )

    def embed_documents(self, texts):
        documents = [Document(content=text) for text in texts]
        embeddings = self.embedding_model.embed_documents(documents)
        return embeddings

    def query(self, question):
        prediction = self.llm_model.predict(question)
        return prediction


# Example usage
if __name__ == "__main__":
    # Replace with your actual document store
    document_store = None
    haystack_initializer = HaystackInitializer(document_store)

    # Example embedding
    texts = ["This is a sample document.", "Another document for embedding."]
    embeddings = haystack_initializer.embed_documents(texts)
    print("Embeddings:", embeddings)

    # Example query
    answer = haystack_initializer.query("What is a sample document?")
    print("Answer:", answer)
