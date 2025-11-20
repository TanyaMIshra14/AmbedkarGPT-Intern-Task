#!/usr/bin/env python3
"""
AmbedkarGPT - Advanced RAG-based Q&A System
Interactive question-answering system powered by LangChain, ChromaDB, and Ollama
for Dr. Ambedkar's speeches and writings
"""

import os
import shutil
from typing import Dict, List, Optional, Any


# Core LangChain imports
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Community extensions
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


# Default configuration constants
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_CORPUS_PATH = "corpus"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_RETRIEVAL_K = 3


class AmbedkarGPT:
    """Advanced RAG-based Q&A system for Dr. Ambedkar's works"""

    def __init__(
        self,
        corpus_path: str = DEFAULT_CORPUS_PATH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """Initialize the RAG system with configurable parameters

        Args:
            corpus_path: Directory or file path containing documents
            chunk_size: Maximum size of text chunks for processing
            chunk_overlap: Number of overlapping characters between chunks
        """
        # Core configuration
        self.corpus_path = corpus_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # System components
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[Ollama] = None

        # Initialize core components
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize embedding and language models"""
        print("Initializing AI models...")

        # Setup embeddings model (local, no API required)
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)

        # Setup Ollama LLM (local inference)
        print("Connecting to Ollama...")
        self.llm = Ollama(model=DEFAULT_LLM_MODEL, temperature=DEFAULT_TEMPERATURE)

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into manageable chunks for processing"""
        print(
            f"Splitting documents (size={self.chunk_size}, overlap={self.chunk_overlap})..."
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(documents)
        print(f"Generated {len(chunks)} text chunks")
        return chunks

    def load_documents(self) -> List[Any]:
        """Load and validate documents from the specified corpus path"""
        print(f"Loading documents from: {self.corpus_path}")

        # Determine loader type based on path
        if os.path.isdir(self.corpus_path):
            loader = DirectoryLoader(
                self.corpus_path, glob="*.txt", loader_cls=TextLoader
            )
        else:
            loader = TextLoader(self.corpus_path)

        documents = loader.load()
        print(f"Successfully loaded {len(documents)} document(s)")
        return documents

    def create_vectorstore(
        self, chunks: List[Any], persist_directory: str = "./chroma_db"
    ) -> Chroma:
        """Build and persist the vector database for semantic search"""
        print("Building vector database...")

        # Clean existing database to ensure fresh start
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Cleaned existing database at {persist_directory}")

        # Create new vector store with embeddings
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )

        print(f"Vector database ready with {len(chunks)} embeddings")
        return self.vectorstore

    def _create_system_prompt(self) -> PromptTemplate:
        """Create the specialized prompt template for Dr. Ambedkar's works"""
        template = """You are AmbedkarGPT, an expert AI assistant specializing in Dr. Bhimrao Ramji Ambedkar (1891-1956) - the visionary architect of India's Constitution, renowned social reformer, brilliant jurist, economist, and tireless champion of Dalit rights.

YOUR EXPERTISE:
• Dr. Ambedkar's philosophical works and constitutional contributions
• His speeches, writings, and social reform initiatives
• His role in India's independence movement and nation-building
• His advocacy for social justice and human dignity

RESPONSE GUIDELINES:
Answer questions using ONLY the provided context from Dr. Ambedkar's authentic documents. Maintain scholarly accuracy and respectful tone. If the context doesn't contain sufficient information, respond with: "I cannot provide a complete answer based on the available documents about Dr. Ambedkar."

Context Documents:
{context}

User Question: {question}

Expert Response:"""

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def setup_qa_chain(self) -> None:
        """Configure the retrieval-augmented question-answering system"""
        print("Configuring QA system...")

        # Create specialized prompt for Dr. Ambedkar context
        system_prompt = self._create_system_prompt()

        # Build retrieval QA chain with custom configuration
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": DEFAULT_RETRIEVAL_K}
            ),
            chain_type_kwargs={"prompt": system_prompt},
            return_source_documents=True,
        )

        print("QA system ready for questions!")

    def ask(self, question: str) -> Dict[str, Any]:
        """Process a question and return an AI-generated response with sources

        Args:
            question: User's inquiry about Dr. Ambedkar

        Returns:
            Dictionary containing 'result' (answer) and 'source_documents'

        Raises:
            ValueError: If system hasn't been initialized
        """
        if self.qa_chain is None:
            raise ValueError("System not initialized. Please call initialize() first.")

        print("Processing your question...")
        result = self.qa_chain.invoke({"query": question})
        return result

    def initialize(self) -> None:
        """Execute the complete system initialization pipeline"""
        print("Starting AmbedkarGPT initialization...\n")

        # Step 1: Load source documents
        documents = self.load_documents()

        # Step 2: Process documents into chunks
        chunks = self.split_documents(documents)

        # Step 3: Build vector database
        self.create_vectorstore(chunks)

        # Step 4: Configure QA system
        self.setup_qa_chain()

        print("\nAmbedkarGPT is ready to answer your questions!\n")

    def interactive_mode(self) -> None:
        """Launch interactive question-answering session"""
        self._display_welcome_banner()

        while True:
            try:
                question = input("\nYour Question: ").strip()

                if self._should_exit(question):
                    print(
                        "\nThank you for exploring Dr. Ambedkar's works with AmbedkarGPT!"
                    )
                    break

                if not question:
                    print("Please enter a question about Dr. Ambedkar.")
                    continue

                # Process question and display response
                response = self.ask(question)
                self._display_response(response)

            except KeyboardInterrupt:
                print("\n\nSession ended by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nError processing question: {e}")

    def _display_welcome_banner(self) -> None:
        """Show the welcome message and instructions"""
        print("=" * 70)
        print("AmbedkarGPT - Interactive Q&A Assistant")
        print("=" * 70)
        print("Ask questions about Dr. Bhimrao Ramji Ambedkar's speeches and writings")
        print("Commands: 'quit', 'exit', 'q' to stop | Ctrl+C for quick exit")
        print("=" * 70)

    def _should_exit(self, user_input: str) -> bool:
        """Check if user wants to exit the session"""
        return user_input.lower() in ["quit", "exit", "q", "bye"]

    def _display_response(self, response: Dict[str, Any]) -> None:
        """Format and display the AI response with sources"""
        print("\n" + "=" * 50)
        print("Answer:")
        print(response["result"])

        print("\nSources:")
        for idx, document in enumerate(response["source_documents"], 1):
            source_file = document.metadata.get("source", "Unknown Document")
            print(f"  [{idx}] {os.path.basename(source_file)}")
        print("\n" + "-" * 50)


def validate_environment() -> bool:
    """Verify that required components are available"""
    if not os.path.exists(DEFAULT_CORPUS_PATH):
        print(f"Corpus directory '{DEFAULT_CORPUS_PATH}' not found!")
        print("Please create the corpus folder with Dr. Ambedkar's text files")
        return False
    return True


def display_troubleshooting_guide() -> None:
    """Show troubleshooting information for common issues"""
    print("\nTroubleshooting Guide:")
    print("1. Ensure Ollama is running: 'ollama serve'")
    print("2. Verify Mistral model: 'ollama pull mistral'")
    print("3. Check corpus directory exists with .txt files")
    print("4. Verify internet connection for model downloads")


def main() -> None:
    """Main application entry point"""
    print("Starting AmbedkarGPT...")

    # Validate prerequisites
    if not validate_environment():
        return

    try:
        # Initialize and run the system
        system = AmbedkarGPT(
            corpus_path=DEFAULT_CORPUS_PATH,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

        system.initialize()
        system.interactive_mode()

    except Exception as error:
        print(f"\nUnexpected error: {error}")
        display_troubleshooting_guide()


if __name__ == "__main__":
    main()
