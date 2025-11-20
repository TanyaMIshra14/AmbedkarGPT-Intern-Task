# AmbedkarGPT

**Advanced RAG-based Q&A System for Dr. Ambedkar's Works**

AmbedkarGPT is an interactive question-answering system that allows you to explore the speeches and writings of Dr. Bhimrao Ramji Ambedkar (1891-1956), the principal architect of the Indian Constitution, social reformer, and champion of Dalit rights.

## Features

- **Local AI Processing**: Runs entirely on your machine using Ollama and Hugging Face models
- **Semantic Search**: Uses ChromaDB for efficient document retrieval
- **Source Attribution**: Shows which documents were used to generate answers
- **Interactive CLI**: Easy-to-use command-line interface
- **Configurable Chunking**: Adjustable text processing parameters for optimal performance
- **Privacy-First**: No data sent to external APIs

## System Architecture

```
Documents → Text Splitting → Embeddings → Vector Store → RAG Chain → Answers
    ↓            ↓              ↓           ↓            ↓
  Corpus/    Chunking     HuggingFace   ChromaDB    Ollama LLM
```

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. Ollama Setup
Install Ollama and download the Mistral model:

```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
# Then pull the required model:
ollama pull mistral

# Start Ollama server
ollama serve
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ambedkargpt.git
cd ambedkargpt
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare your corpus:**
   - Create a `corpus/` directory in the project root
   - Add Dr. Ambedkar's text files (`.txt` format) to this directory
   - Example structure:
   ```
   corpus/
   ├── speech1.txt
   ├── speech2.txt
   ├── speech3.txt
   └── ...
   ```

## Usage

### Basic Usage

Run the interactive Q&A system:

```bash
python main.py
```

### Example Session

```
Starting AmbedkarGPT...
Loading documents from: corpus
Successfully loaded 6 document(s)
Splitting documents (size=500, overlap=50)...
Generated 234 text chunks
Building vector database...
Vector database ready with 234 embeddings
Configuring QA system...
QA system ready for questions!

AmbedkarGPT is ready to answer your questions!

======================================================================
AmbedkarGPT - Interactive Q&A Assistant
======================================================================
Ask questions about Dr. Bhimrao Ramji Ambedkar's speeches and writings
Commands: 'quit', 'exit', 'q' to stop | Ctrl+C for quick exit
======================================================================

Your Question: What is the real remedy for caste system according to Ambedkar?

Processing your question...

==================================================
Answer:
According to Dr. Ambedkar, the real remedy for the caste system lies in destroying the belief in the sanctity of the shastras...

Sources:
  [1] speech1.txt
  [2] speech3.txt
  [3] speech6.txt
--------------------------------------------------
```

## Configuration

You can customize the system by modifying the constants in `main.py`:

```python
DEFAULT_CHUNK_SIZE = 500        # Size of text chunks
DEFAULT_CHUNK_OVERLAP = 50      # Overlap between chunks
DEFAULT_CORPUS_PATH = "corpus"  # Path to documents
DEFAULT_RETRIEVAL_K = 3         # Number of documents to retrieve
```

## Project Structure

```
ambedkargpt/
├── main.py                 # Main application file
├── evaluation.py           # Evaluation framework
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── test_dataset.json      # Test questions and answers
├── README.md              # This file
├── corpus/                # Document corpus
│   ├── speech1.txt
│   ├── speech2.txt
│   └── ...
├── chroma_db/             # Vector database (auto-generated)
└── test_results_*.json    # Evaluation results
```

## Evaluation Framework

The project includes a comprehensive evaluation system to test different chunking strategies:

```bash
python evaluation.py
```

This will:
- Test different chunk sizes (Small: 250, Medium: 550, Large: 900)
- Evaluate using multiple metrics (Hit Rate, MRR, ROUGE-L, BLEU, etc.)
- Generate detailed performance reports

### Evaluation Metrics

- **Hit Rate**: Percentage of questions where relevant documents were retrieved
- **MRR (Mean Reciprocal Rank)**: Quality of document ranking
- **ROUGE-L**: Overlap between generated and ground truth answers
- **BLEU**: N-gram based similarity score
- **Cosine Similarity**: Semantic similarity between answers
- **Faithfulness**: How well answers stick to source documents
- **Answer Relevance**: Relevance of generated answers to questions

## Performance Results

Based on evaluation with 25 test questions:

| Configuration | Hit Rate | MRR   | ROUGE-L | Faithfulness |
|--------------|----------|-------|---------|--------------|
| Small (250)  | 0.84     | 0.79  | 0.27    | 0.30         |
| Medium (550) | **0.88** | 0.79  | 0.27    | 0.45         |
| Large (900)  | 0.88     | 0.79  | **0.27**| **0.58**     |

**Recommendation**: Medium chunk size (550 tokens) provides the best balance of retrieval and generation performance.

## Dependencies

- `langchain`: LLM application framework
- `langchain-community`: Community extensions for LangChain
- `chromadb`: Vector database for embeddings
- `sentence-transformers`: Embedding models
- `ollama`: Local LLM inference
- `huggingface-hub`: Access to Hugging Face models

## Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   ollama serve
   ```

2. **Mistral model not found**
   ```bash
   ollama pull mistral
   ```

3. **Corpus directory missing**
   - Create `corpus/` directory
   - Add `.txt` files with Dr. Ambedkar's works

4. **Memory issues**
   - Reduce `DEFAULT_CHUNK_SIZE` 
   - Use fewer documents in corpus

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB for models and vector database
- **CPU**: Multi-core processor recommended for faster embedding generation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dr. Bhimrao Ramji Ambedkar for his invaluable contributions to Indian society
- LangChain team for the RAG framework
- Ollama team for local LLM inference
- Hugging Face for embedding models
- ChromaDB team for the vector database

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ambedkargpt2024,
  title={AmbedkarGPT: RAG-based Q&A System for Dr. Ambedkar's Works},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ambedkargpt}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

---

**"Education is the right weapon to cut the social slavery and it is the only way to enlighten the downtrodden masses."** - Dr. B.R. Ambedkar
