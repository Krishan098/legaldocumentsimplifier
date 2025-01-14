import streamlit as st
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt_tab')

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def bart_model(input_text):
    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=512, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

def preprocess(text):
    text = remove_citations(text)
    text = split_long_sentences(text)
    text = replace_legal_terms(text)
    text = standardize_structure(text)
    return text

def postprocess(text):
    text = fix_formatting(text)
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
        'inter alia': 'among other things'
        #add more terms from simple_legal.ipynb
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

def add_paragraph_breaks(text):
    text = re.sub(r'([.!?])\s+(?=[A-Z])', r'\1\n\n', text)
    return text

def process_document(text):
    simplified_text = preprocess(text)
    model_output = bart_model(simplified_text)
    final_text = postprocess(model_output)
    return final_text


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge_scores(generated, reference):
    scores = []
    for gen, ref in zip(generated, reference):
        score = scorer.score(gen, ref)
        scores.append(score)
    avg_scores = {
        'rouge1': np.mean([score['rouge1'].fmeasure for score in scores]),
        'rouge2': np.mean([score['rouge2'].fmeasure for score in scores]),
        'rougeL': np.mean([score['rougeL'].fmeasure for score in scores])
    }
    return avg_scores

#streamlit from here
st.title("Summary Generator with Pre- and Post-Simplification ROUGE Scores")


input_text = st.text_area("Enter the text to be summarized:", height=200)
if st.button("Generate Summary"):
    if input_text.strip():

        pre_simplified_summary = bart_model(input_text)


        post_simplified_text = process_document(input_text)
        post_simplified_summary = bart_model(post_simplified_text)


        pre_rouge_scores = calculate_rouge_scores([pre_simplified_summary], [input_text])
        post_rouge_scores = calculate_rouge_scores([post_simplified_summary], [input_text])

        #add fkgl score for both pre and post 
        # and increase the area of output text box
        
        st.subheader("Pre-Simplification Results")
        st.write("Summary:")
        st.text(pre_simplified_summary)
        st.write("ROUGE Scores:")
        st.json(pre_rouge_scores)

        st.subheader("Post-Simplification Results")
        st.write("Summary:")
        st.text(post_simplified_summary)
        st.write("ROUGE Scores:")
        st.json(post_rouge_scores)
    else:
        st.warning("Please enter some text to summarize.")
