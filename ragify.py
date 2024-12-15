import time
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import pandas as pd


class Ragify:
    def __init__(self, pdf_paths, llm_name="llama3.2:1b", embedding_name="nomic-embed-text", chunk_size=1000):
        self.pdf_paths = pdf_paths
        self.llm_name = llm_name
        self.embedding_name = embedding_name
        self.chunk_size = chunk_size

        # Load and split documents
        self.loader = [PyPDFLoader(path) for path in self.pdf_paths]
        self.all_data = [doc for loader in self.loader for doc in loader.load()]
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=200)
        self.all_splits = self.text_splitter.split_documents(self.all_data)

        # Initialize embeddings and vector store
        self.local_embeddings = OllamaEmbeddings(model=self.embedding_name)
        self.vectorstore = FAISS.from_documents(documents=self.all_splits, embedding=self.local_embeddings)

        # Initialize LLM and prompt template
        self.model = ChatOllama(model=self.llm_name)
        self.rag_prompt = ChatPromptTemplate.from_template("""
            You are an assistant specialized in providing information about the 'Rules and Regulations Governing Graduate Studies' at METU.
            Use ONLY the provided context to answer the question. Do not use outside knowledge.

            If you cannot find relevant information in the context, or if the question is unrelated to METU or its graduate regulations, respond exactly with:
            "I'm sorry, I cannot assist with that question."

            ### Examples of questions you can answer:
            - "What are the graduation requirements for a Master's at METU?"
            - "How do I apply for a thesis extension at METU?"

            ### Examples of questions you CANNOT answer:
            - "What are the graduation requirements at Bilkent University?"
            - "What is the capital of Turkey?"
            - Any question that is not related to METU's graduate regulations.

            Always respond with: "I'm sorry, I cannot assist with that question." if the question is unrelated to METU or if the context does not contain relevant information.

            <context>
            {context}
            </context>

            Question: {question}
        """)


        self.retriever = self.vectorstore.as_retriever()
        self.chat_history = []  # Initialize empty chat history

        # Define QA chain
        self.qa_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.model
            | StrOutputParser()
        )

    def format_docs(self, docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def clean_text(self, text):
        """Clean text by removing control characters and excessive whitespace."""
        if not text or not isinstance(text, str):
            return text  # Return as-is if invalid or not a string
        # Remove control characters and normalize whitespace
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control characters
        text = re.sub(r'\s+', ' ', text).strip()          # Normalize whitespace
        return text

    def update_chat_history(self, question, response):
        """Update chat history with the latest interaction."""
        max_history_length = 5  # Limit to the last 5 interactions
        self.chat_history.insert(0, f"User: {question}\nAssistant: {response}")
        self.chat_history = self.chat_history[:max_history_length]

    def generate_response(self, question):
        """Generate a response to a question."""
        if not question or not isinstance(question, str):
            raise ValueError("The question must be a non-empty string.")

        chat_history_str = "\n".join(self.chat_history) if self.chat_history else "No previous interactions."
        start_time = time.time()

        # Retrieve relevant documents
        try:
            print(f"Retrieving documents for question: {question}")
            retrieved_docs = self.retriever.invoke(question)
            if not retrieved_docs:
                print("No documents retrieved.")
                return "I'm sorry, I couldn't find any relevant information.", 0.0
            context = self.format_docs(retrieved_docs)
            context = self.clean_text(context)  # Clean the context
            if not context.strip():
                print("Retrieved context is empty after cleaning.")
                return "I'm sorry, the retrieved context is empty.", 0.0
            print(f"Retrieved context: {context[:200]}...")
        except Exception as e:
            raise RuntimeError(f"Error retrieving documents: {e}")

        # Combine context, question, and chat history into a single string
        combined_input = f"Context:\n{context}\n\nChat History:\n{chat_history_str}\n\nQuestion:\n{question}"
        print(f"Combined Input: {combined_input[:500]}...")  # Debug combined input

        # Generate response
        try:
            print(f"Generating response for question: {question}")
            raw_response = self.model.invoke(combined_input)  # Pass as a single string
            # Check if response has a 'content' attribute
            if hasattr(raw_response, 'content'):
                response = raw_response.content
            else:
                response = str(raw_response)  # Fallback in case of unexpected format
            print(f"Response: {response}")
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")

        time_taken = time.time() - start_time
        self.update_chat_history(question, response)

        return response, time_taken

    def reset_chat_history(self):
        """Reset the chat history."""
        self.chat_history = []

    def evaluate_responses(self, questions, reference_responses, grouped_reference_chunks):
        """Evaluate chatbot responses."""
        chatbot_responses = []
        response_times = []

        for q in questions:
            response, time_taken = self.generate_response(q)
            chatbot_responses.append(response)
            response_times.append(time_taken)

        precision_k = self.calculate_precision_k(questions, grouped_reference_chunks, top_k=3)
        rouge_scores = self.calculate_rouge_scores(chatbot_responses, reference_responses)
        bleu_score = self.calculate_bleu_scores(chatbot_responses, reference_responses)

        return chatbot_responses, response_times, precision_k, rouge_scores, bleu_score

    def calculate_precision_k(self, questions, grouped_reference_chunks, top_k=3):
        """
        Calculate Precision@k for retrieved chunks vs. grouped reference chunks.
        """
        match = 0

        for i, (question, ref_chunks) in enumerate(zip(questions, grouped_reference_chunks)):

            # Retrieve relevant chunks for the question
            retrieved_docs = self.retriever.get_relevant_documents(question)
            retrived_chunks = ""
            for doc in retrieved_docs[:min(top_k, len(retrieved_docs))]:
                retrived_chunks += doc.page_content.strip().lower().replace("\n", "")

            # Normalize reference chunks
            ref_chunks_list = [chunk for chunk in ref_chunks.strip().lower().split()]

            if ref_chunks.strip().lower() in retrived_chunks:
                match += 1

        return {"mean": match / len(questions)}

    def calculate_rouge_scores(self, chatbot_responses, reference_responses):
        rouge_evaluator = Rouge()
        metrics = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

        for hyp, ref in zip(chatbot_responses, reference_responses):
            if hyp and ref:
                scores = rouge_evaluator.get_scores(hyp.strip().lower(), ref.strip().lower(), avg=False)
                for metric in metrics.keys():
                    metrics[metric].append(scores[0][metric])

        return {
            metric: {
                "mean": {
                    "precision": np.mean([score["p"] for score in metrics[metric]]),
                    "recall": np.mean([score["r"] for score in metrics[metric]]),
                    "f1": np.mean([score["f"] for score in metrics[metric]]),
                },
                "std": {
                    "precision": np.std([score["p"] for score in metrics[metric]]),
                    "recall": np.std([score["r"] for score in metrics[metric]]),
                    "f1": np.std([score["f"] for score in metrics[metric]]),
                },
            }
            for metric in metrics
        }

    def calculate_bleu_scores(self, chatbot_responses, reference_responses):
        scores = []
        for hyp, ref in zip(chatbot_responses, reference_responses):
            if hyp and ref:
                score = sentence_bleu([ref.strip().lower().split()], hyp.strip().lower().split(),
                                      smoothing_function=SmoothingFunction().method4)
                scores.append(score)
        return {"mean": np.mean(scores), "std": np.std(scores)}


if __name__ == "__main__":
    # Initialize Ragify
    rag_pipeline = Ragify(
        pdf_paths=[
            r"documents/METU_Regulation.pdf",
            r"documents/ISStudentGuide_2023-2024_v1.5.pdf"
        ],
        llm_name="llama3.2:latest",
        embedding_name="nomic-embed-text",
        chunk_size=1000
    )

    print("Hi there! Need help with the rules and regs of Informatics at METU? I’ve got you covered. Let’s do this! 💻✨")
    print("Type your question below (type 'history' to view chat history or 'exit' to quit):\n")

    while True:
        # Take user input
        user_input = input("Q: ").strip()
        if user_input.lower() == "exit":
            print("Stay curious, and don’t forget: ODTÜ’de başarı bir gelenektir! 🌟")
            break
        elif user_input.lower() == "history":
            # Display chat history
            if rag_pipeline.chat_history:
                print("\nChat History:")
                for i, interaction in enumerate(rag_pipeline.chat_history[::-1], 1):
                    print(f"{i}. {interaction}")
                print()
            else:
                print("No chat history available.\n")
            continue

        # Generate response
        try:
            response, time_taken = rag_pipeline.generate_response(user_input)
            print(f"A: {response}")
            print(f"(Response time: {time_taken:.2f} seconds)\n")
        except Exception as e:
            print(f"Error: {e}\n")
