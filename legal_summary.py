import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
)
import streamlit as st
from rouge_score import rouge_scorer
import numpy as np

model = BartForConditionalGeneration.from_pretrained('G:/6th sem/GDG/model_outputs/models/billsum')
tokenizer = BartTokenizer.from_pretrained('G:/6th sem/GDG/model_outputs/models/billsum')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

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

def calculate_rouge_score(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(generated, reference)
    return score

st.title("Legal Text Summary Generator")

text = st.text_area("Enter legal text:", height=200)

if st.button("Generate Summary"):
    if text.strip():  
        summary = generate_summary(text)
        rouge_score = calculate_rouge_score(summary, text)
        
        st.subheader("Generated Summary")
        st.write(summary)
        
        st.subheader("ROUGE Scores")
        st.json(rouge_score)
    else:
        st.warning("Please enter some text to summarize.")
