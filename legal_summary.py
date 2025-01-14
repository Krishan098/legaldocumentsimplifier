import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
)
import streamlit as st
from rouge_score import rouge_scorer
import numpy as np

model = BartForConditionalGeneration.from_pretrained('models/billsum')
tokenizer = BartTokenizer.from_pretrained('models/billsum')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

from summarygenerator import generate_summary
st.title("Legal Text Summary Generator")
text = st.text_area("Enter legal text:", height=100)
if st.button("Generate Summary"):
    if text.strip():  
        summary = generate_summary(text)
        st.subheader("Generated Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
