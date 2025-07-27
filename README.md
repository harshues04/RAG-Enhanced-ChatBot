# Career Guidance Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Flask and LangChain that provides guidance on placements, internships, and higher studies for university students.

## Overview

This project implements a chatbot that uses RAG architecture to retrieve information from a knowledge base and generate contextually relevant responses. The chatbot is specifically designed to answer questions related to career guidance for university students, including information about internships, placements, and higher education options.

## Features

- **Web-based Chat Interface**: User-friendly interface for interacting with the chatbot
- **Retrieval-Augmented Generation**: Combines retrieval of relevant information with language model generation
- **Context-Aware Responses**: Provides answers based on retrieved context from the knowledge base
- **Domain-Specific Knowledge**: Focused on student career guidance topics

## Technology Stack

- **Flask**: Web framework for handling HTTP requests
- **LangChain**: Framework for building LLM applications
- **HuggingFace**: For accessing language models and embeddings
- **FAISS**: Vector database for efficient similarity search
- **Sentence Transformers**: For generating text embeddings

## Project Structure

```
rag_chatbot/
├── app.py                 # Main Flask application
├── doc_rag.txt            # Knowledge base document
├── templates/
│   └── bot_1.html         # HTML template for the chat interface
├── data/
│   └── dataprovider.py    # File for storing API keys
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harshues04/RAG-Enhanced-ChatBot.git
cd rag-chatbot
```

2. Install the required dependencies:
```bash
pip install flask langchain langchain_community faiss-cpu huggingface_hub sentence-transformers
```

3. Set up your API keys:
   - Create a `data` directory if it doesn't exist
   - Create a `dataprovider.py` file inside the data directory with the following content:
   ```python
   # API keys
   key = "your_openai_api_key_here"  # Not required for current implementation
   hg_key = "your_huggingface_api_key"  # Get this from huggingface.co
   ```
   - Replace `your_huggingface_api_key` with your actual HuggingFace API key

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Enter your questions in the chat interface about student career topics:
   - Internship search strategies
   - Placement preparation
   - Higher studies options
   - Career decision making

## Customization

### Modifying the Knowledge Base

The chatbot's knowledge is contained in the `doc_rag.txt` file. You can customize this file with any information you want the chatbot to have access to:

1. Open `doc_rag.txt`
2. Modify or add information in a structured format
3. Save the file
4. Restart the application to apply changes

### Adjusting the Model Parameters

To adjust how the language model generates responses, you can modify the following parameters in `app.py`:

```python
model_kwargs={
    "max_new_tokens": 512,  # Controls response length
    "top_k": 30,            # Number of top tokens to consider
    "temperature": 0.1,     # Lower for more deterministic responses
    "repetition_penalty": 1.03,  # Prevents repetition
}
```

## Troubleshooting

### Common Issues

1. **HuggingFace API Authentication Errors**:
   - Verify that your HuggingFace API key is correct
   - Check that your account has access to the required models

2. **Slow Response Times**:
   - Initial queries may be slow as embeddings are generated
   - Consider using smaller chunks in the text splitter for faster processing

3. **Module Not Found Errors**:
   - Ensure all dependencies are properly installed
   - Check that the import statements match the installed package versions

## Limitations

- The chatbot's knowledge is limited to the information in the `doc_rag.txt` file
- Processing times may vary based on the model and hardware used
- The quality of responses depends on the quality and structure of the knowledge base

## Future Improvements

- Add support for multiple knowledge bases
- Implement conversation memory to maintain context across multiple queries
- Add user authentication for personalized responses
- Integrate alternative language models for comparison

## License

This project is licensed under the MIT License.

## Acknowledgments

- This project uses the Zephyr language model from HuggingFace
- Built with LangChain framework for language model applications
