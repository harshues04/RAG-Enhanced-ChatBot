from flask import Flask, request, render_template, jsonify
from data.dataprovider import key, hg_key
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
import re
import os

app = Flask(__name__)

# Print working directory for debugging
print(f"Current working directory: {os.getcwd()}")

try:
    # Use a smaller model for faster responses
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",  # Smaller model
        task="text2text-generation",
        model_kwargs={
            "max_length": 200,
            "temperature": 0.1,
        },
        huggingfacehub_api_token=hg_key,
    )
    print("HuggingFace model loaded successfully")
except Exception as e:
    print(f"Error loading HuggingFace model: {str(e)}")

# Define your rag chatbot function
def chat_with_rag(message):
    try:
        # Use absolute path to locate the file
        file_path = os.path.join(os.getcwd(), "doc_rag.txt")
        print(f"Looking for file at: {file_path}")
        
        if not os.path.exists(file_path):
            return "Error: Knowledge base file not found. Please create the doc_rag.txt file in the project directory."
        
        full_text = open(file_path, "r", encoding="utf-8").read()
        print(f"File loaded successfully. Size: {len(full_text)} characters")
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(full_text)
        print(f"Split into {len(texts)} chunks")

        try:
            print("Loading embeddings model...")
            embeddings = HuggingFaceEmbeddings()
            print("Embeddings model loaded successfully")
        except Exception as e:
            print(f"Error loading embeddings model: {str(e)}")
            return f"Error loading embeddings model: {str(e)}"

        try:
            print("Creating vector database...")
            db = FAISS.from_texts(texts, embeddings)
            retriever = db.as_retriever()
            print("Vector database created successfully")
        except Exception as e:
            print(f"Error creating vector database: {str(e)}")
            return f"Error creating vector database: {str(e)}"
            
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        model = llm

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        try:
            print("Building and running chain...")
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )    
            
            result = chain.invoke(message)
            print(f"Chain completed successfully. Response length: {len(result)}")
            return result
        except Exception as e:
            print(f"Error running chain: {str(e)}")
            return f"Error generating response: {str(e)}"
            
    except Exception as e:
        print(f"Error in RAG processing: {str(e)}")
        return f"I'm having trouble processing your request. Error: {str(e)}"

# Define your Flask routes
@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'response': 'This is a test response. The server is running correctly.'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("Received chat request")
        user_message = request.form['user_input']
        print(f"Processing message: {user_message}")
        
        # Simple response for debugging
        if user_message.lower() == "test":
            return jsonify({'response': 'Test response successful!'})
        
        bot_message = chat_with_rag(user_message)
        print(f"Got raw response: {bot_message[:100]}...")  # Print first 100 chars
        
        # Define the regex pattern to extract the answer
        pattern = r"Answer:\s*(.*)"
        # Search for the pattern in the text
        match = re.search(pattern, bot_message, re.DOTALL)

        if match:
            answer = match.group(1).strip()
            print(f"Extracted Answer: {answer[:100]}...")
            return jsonify({'response': answer})
        else:
            print("Answer pattern not found, returning full message")
            return jsonify({'response': bot_message})
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({'response': f"I encountered an error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)