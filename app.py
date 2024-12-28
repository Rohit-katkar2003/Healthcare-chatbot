import os
import warnings
import tensorflow as tf
import requests  # For API requests
from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_community.llms import HuggingFacePipeline
import traceback 
import re 
from src.helper import download_hugging_face_embeddings, load_faiss_with_metadata

app = Flask(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Initialize components
embeddings = download_hugging_face_embeddings()
vectorstore = load_faiss_with_metadata(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# Load fine-tuned model
model_path = "flan-t5-finetuned-new"
tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
model = T5ForConditionalGeneration.from_pretrained(model_path)

pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_path,
    task="text2text-generation",
    pipeline_kwargs={
        "max_length": 50,
        "min_length": 15,
        "top_k": 25,
        "temperature": 0.4,
        "top_p": 0.85,
        "repetition_penalty": 1.7, 
        "do_sample" : True , 
        "num_beams": 3 , # Reduced beams for speed
    }
)

PROMPT = PromptTemplate(
    template=(
        "You are a helpful and expert Health-care assistant. Analyze the context and provide a detailed, accurate, and structured response to the question.\n\n"
        "If you don't know the answer just say I don't know."
        "Context: {context}\n\n"
        "Question: {question}\n"
        "Answer with explanations where applicable:"
    ),
    input_variables=["context", "question"]
)


qa = RetrievalQA.from_chain_type(
    llm=pipeline,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)

# Clean up the context
def clean_context(context):
    return context.replace('\n', ' ').strip()

# Post-process the model output
def post_process_response(response):
    response = re.sub(r'\b(\w+\s+){0,2}(\w+)(\s+\2)+\b', r'\1\2', response)  # Remove repetitive words
    response = re.sub(r'(state of being )+', 'state of being ', response, flags=re.IGNORECASE)
    sentences = [sentence.strip().capitalize() for sentence in response.split('.') if sentence.strip()]
    return ". ".join(sentences[:3]) + "." if sentences else "I don't know."

# Main function for getting output
def get_output(input_text):
    # Reduced retrieval step size
    retrieved_documents = retriever.invoke(input_text)
    context = " ".join(clean_context(doc.page_content) for doc in retrieved_documents)
    query_input = f"Context: {context}\n\nQuestion: {input_text}\n"
    result = qa({"query": query_input})
    return post_process_response(result['result'])

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    print(f"User input: {msg}")
    input_text = msg
    try:
        if "and" in input_text:
            sub_queries = input_text.lower().split("and")
            responses = [get_output(sub_query.strip()) for sub_query in sub_queries]
            return str("\n".join(responses)) 
        else: 
            return str(get_output(input_text)) 
        

    except Exception as e:
        print(traceback.format_exc())
        return f"Error processing the request: {e}"

if __name__ == '__main__':
    app.run(debug=True)
