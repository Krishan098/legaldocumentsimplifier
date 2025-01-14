import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st
from rouge_score import rouge_scorer
import textstat
import numpy as np
import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

# Load BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('G:/6th sem/GDG/model_outputs/models/billsum')
tokenizer = BartTokenizer.from_pretrained('G:/6th sem/GDG/model_outputs/models/billsum')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Preprocessing functions for simplification
def remove_citations(text):
    return re.sub(r'\(\d+\s+[A-Za-z\.]+\s+\d+\)', '', text)

def split_long_sentences(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 50:
            splits = []
            current_split = []
            for token in sentence.split():
                current_split.append(token)
                if token == ',' or token == ';':
                    splits.append(' '.join(current_split))
                    current_split = []
            if current_split:
                splits.append(' '.join(current_split))
            processed_sentences.extend(splits)
        else:
            processed_sentences.append(sentence)
    return ' '.join(processed_sentences)

def replace_legal_terms(text):
    legal_terms = {
        'hereinafter': 'from now on',
        'pursuant to': 'according to',
        'whereas': 'since',
        'notwithstanding': 'despite',
        'forthwith': 'immediately',
    }
    for term, replacement in legal_terms.items():
        text = re.sub(r'\b' + term + r'\b', replacement, text, flags=re.IGNORECASE)
    return text

def preprocess(text):
    text = remove_citations(text)
    text = split_long_sentences(text)
    text = replace_legal_terms(text)
    return text

# Text summarization function
def generate_summary(text, max_length=150, min_length=50):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ROUGE score calculation
def calculate_rouge_score(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(generated, reference)

# FKGL calculation
def calculate_fkgl(text):
    return textstat.flesch_kincaid_grade(text)

# Streamlit UI
st.title("Legal Text Simplification and Summary Generator")

text = st.text_area("Enter legal text:", height=200)

if st.button("Generate"):
    if text.strip():
        # Simplify the input text
        simplified_text = preprocess(text)
        
        # Generate the summary from the simplified text
        summary = generate_summary(simplified_text)
        
        # Calculate evaluation metrics
        rouge_score = calculate_rouge_score(summary, text)
        fkgl_score = calculate_fkgl(summary)
        
        # Display results
        st.subheader("Simplified Text")
        st.write(simplified_text)
        
        st.subheader("Generated Summary")
        st.write(summary)
        
        st.subheader("Evaluation Metrics")
        st.text("ROUGE Scores:")
        st.json(rouge_score)
        st.write(f"Flesch-Kincaid Grade Level (FKGL) Score: **{fkgl_score:.2f}**")
    else:
        st.warning("Please enter some text to process.")
