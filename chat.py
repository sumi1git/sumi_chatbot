print("Loading IDEAM Chatbot...")

#import gradio as gr
#from transformers import pipeline
#from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------

DATA_FILE = "ideam_qna_dataset_v2.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Please make sure it’s in the same folder.")

df = pd.read_csv(DATA_FILE)

if not {"question", "answer"}.issubset(df.columns):
    raise ValueError("Dataset must contain 'question' and 'answer' columns.")

questions = df["question"].astype(str).tolist()
answers = df["answer"].astype(str).tolist()

# ----------------------------------------------------
# 2. Load Models
# ----------------------------------------------------

# Embedding model (semantic understanding)
# Force CPU if you want to avoid CUDA tensor mismatch
embedder = SentenceTransformer("all-mpnet-base-v2")

# Text generation model for fluent responses
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# Encode all dataset questions
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

# ----------------------------------------------------
# 3. Chat Function
# ----------------------------------------------------

def chat_with_bot(user_query):
    """
    Find the most semantically similar question in the dataset
    and generate a refined, fluent answer using Flan-T5.
    """
    if len(questions) == 0:
        return "The dataset is empty. Please provide some Q&A data.", None

    # Encode the user's question
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarity (and move to CPU for numpy)
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings).cpu().numpy()
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[0][best_idx])

    # If similarity is too low, prompt user to teach
    if best_score < 0.70:
        return "I'm not sure about that yet. You can teach me the correct answer.", None


    # Retrieve best-matching answer
    best_answer = answers[best_idx]
    prompt = f"Question: {user_query}\nAnswer: {best_answer}"

    # Generate a refined, natural response
    result = qa_model(prompt, max_length=150)[0]["generated_text"]
    return result, best_idx

# ----------------------------------------------------
# 4. Add / Correct Answers
# ----------------------------------------------------

def correct_answer(user_query, correct_text):
    """
    Add a new Q&A pair to the dataset and rebuild embeddings.
    """
    global df, questions, answers, question_embeddings

    new_row = {"question": user_query, "answer": correct_text}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    # Update local memory
    questions.append(user_query)
    answers.append(correct_text)
    question_embeddings = embedder.encode(questions, convert_to_tensor=True)

    return f"New knowledge added:\n{user_query} → {correct_text}"

# ----------------------------------------------------
# 5. Build Gradio Interface
# ----------------------------------------------------

with gr.Blocks(title="IDEAM Chatbot") as chat_ui:
    gr.Markdown("## IDEAM Chatbot\nAsk a question below. If the answer is incorrect, provide the correct one to teach the model.")

    with gr.Row():
        user_input = gr.Textbox(lines=2, label="Your Question")
        response_output = gr.Textbox(label="Chatbot Response")

    with gr.Row():
        correction_input = gr.Textbox(lines=2, label="Correct or Add Answer (optional)")
        submit_button = gr.Button("Get Answer")
        correct_button = gr.Button("Teach / Correct")

    status_output = gr.Markdown("")

    # Button functions
    def respond(query):
        answer, _ = chat_with_bot(query)
        return answer

    submit_button.click(fn=respond, inputs=user_input, outputs=response_output)
    correct_button.click(fn=correct_answer, inputs=[user_input, correction_input], outputs=status_output)

chat_ui.launch(share=True)