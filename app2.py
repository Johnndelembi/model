import streamlit as st
import pandas as pd
import TensorFlow
from transformers import pipeline

model = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
df = pd.read_csv("vgsales.csv")
table = df.astype(str)

st.title("Chat with your data")
st.write(table.head())

query = st.text_area("Ask me anything about the dataset and i will answer you")
output = model(table=table, query=query)['answer']
st.write(output)
    

