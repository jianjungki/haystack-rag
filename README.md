# haystack-rag

Welcome to the haystack-rag project! This toolkit is designed to facilitate the implementation of Retrieval-Augmented Generation (RAG) using the **Haystack 2.x** framework. It provides a set of tools and utilities to streamline the process of building RAG-based applications.

## Table of Contents

- [haystack-rag](#haystack-rag)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Thanks to](#thanks-to)

## Features

- Easy integration with the Haystack framework
- Support for various document stores and retrievers
- Flexible configuration options for different use cases
- Example scripts and notebooks for quick start
- Comprehensive documentation

## Installation

To install the RAG Toolkit, clone the repository and install the required dependencies:

```bash
git clone https://github.com/jianjungki/haystack-rag.git
cd haystack-rag
pip install -r requirements.txt
```

## Usage

Here’s a quick example of how to use the haystack rag Toolkit:

```python
from rag.rag import RAGPipeline

# Usage
rag_pipeline = RAGPipeline(
    embedder_type="sentence_transformer",  # Add these parameters
    generator_type="openai",         # Add these parameters
    embedding_model="malenia1/ternary-weight-embedding",
    llm_model="meta-llama/llama-3.1-70b-instruct:free",
    api_key="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",
    base_url="https://openrouter.ai/api/v1")
question = "What preservatives are allowed in bakery products"
print(rag_pipeline.run(question))
```

For more detailed usage instructions, please refer to the [Documentation](https://github.com/jianjungki/haystack-rag/wiki).

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, please fork the repository and submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Thanks to

- [Haystack](https://haystack.deepset.ai/) for providing a powerful framework for building RAG applications.
- [Chainlit](https://docs.chainlit.io/) for chat interface with pipeline analyze and monitor
- All contributors and users for their support and feedback.

---

Feel free to reach out if you have any questions or suggestions!
