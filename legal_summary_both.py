import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
)
import streamlit as st
from rouge_score import rouge_scorer
import numpy as np
import textstat  # Library to calculate FKGL
import nltk
from nltk.tokenize import sent_tokenize
import re

nltk.download('punkt')  # Download punkt tokenizer

# Initialize model and tokenizer
model = BartForConditionalGeneration.from_pretrained('G:/6th sem/GDG/model_outputs/models/billsum')
tokenizer = BartTokenizer.from_pretrained('G:/6th sem/GDG/model_outputs/models/billsum')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Functions for simplification process
def preprocess(text):
    text = remove_citations(text)
    text = split_long_sentences(text)
    text = replace_legal_terms(text)
    text = standardize_structure(text)
    return text

def postprocess(text):
    text = fix_formatting(text)
    text = ensure_consistency(text)
    text = add_paragraph_breaks(text)
    return text

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
        'inter alia': 'among other things',
        'ab initio': 'from the beginning',
        'ipso facto': 'by that fact itself',
        'mutatis mutandis': 'with the necessary changes',
        'de facto': 'in fact',
        'de jure': 'by law',
        'quid pro quo': 'something for something',
        'sub judice': 'under judicial consideration',
        'prima facie': 'at first glance',
        'pro rata': 'in proportion',
        'ultra vires': 'beyond the powers',
        'res judicata': 'a matter already judged',
        'a fortiori': 'even more so',
        'ex parte': 'by one party',
        'actus reus': 'guilty act',
        'mens rea': 'guilty mind',
        'nolo contendere': 'no contest',
        'stare decisis': 'to stand by decided cases',
        'in loco parentis': 'in the place of a parent',
        'per curiam': 'by the court',
        'amicus curiae': 'friend of the court',
        'sui generis': 'unique',
        'caveat emptor': 'let the buyer beware',
        'habeas corpus': 'you shall have the body',
        'ex post facto': 'after the fact',
        'in situ': 'in its original place',
        'pari passu': 'on equal footing',
        'lex loci': 'law of the place',
        'contra proferentem': 'against the drafter',
        'pro bono': 'for the public good',
        'ad hoc': 'for this specific purpose',
        'ex officio': 'by virtue of office',
        'jus cogens': 'compelling law',
        'locus standi': 'right to bring action',
        'nullum crimen sine lege': 'no crime without law',
    }

    for term, replacement in legal_terms.items():
        text = re.sub(r'\b' + term + r'\b', replacement, text, flags=re.IGNORECASE)
    return text

def standardize_structure(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Section \d+\.', lambda m: '\n' + m.group(0) + '\n', text)
    return text

def fix_formatting(text):
    text = '. '.join(s.strip().capitalize() for s in text.split('. '))
    text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1 \2', text)
    return text

def ensure_consistency(text):
    return text

def add_paragraph_breaks(text):
    text = re.sub(r'([.!?])\s+(?=[A-Z])', r'\1\n\n', text)
    return text

def process_document(text):
    simplified_text = preprocess(text)
    model_output = bart_model(simplified_text)
    final_text = postprocess(model_output)
    return final_text

# BART Model for summarization
def bart_model(input_text):
    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=512, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

# Functions for evaluating ROUGE and FKGL
def calculate_rouge_score(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(generated, reference)
    return score

def calculate_fkgl(text):
    return textstat.flesch_kincaid_grade(text)

# Streamlit Interface
st.title("Legal Text Simplification and Summary Generator")

text = st.text_area("Enter legal text:", height=200)

if st.button("Generate Summary"):
    if text.strip():  
        # Simplify text first
        simplified_text = process_document(text)
        
        # Generate summary of simplified text
        summary = generate_summary(simplified_text)
        
        # Calculate ROUGE scores for the simplified summary
        rouge_score = calculate_rouge_score(summary, simplified_text)
        
        # Calculate FKGL score for the generated summary
        fkgl_score = calculate_fkgl(summary)
        
        # Display results
        st.subheader("Generated Summary")
        st.write(summary)
        
        st.subheader("ROUGE Scores")
        st.json(rouge_score)
        
        st.subheader("Flesch-Kincaid Grade Level (FKGL) Score")
        st.write(f"The FKGL score for the summary is: **{fkgl_score:.2f}**")
    else:
        st.warning("Please enter some text to summarize.")
