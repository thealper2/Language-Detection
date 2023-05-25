import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

st.title("Language Detection")

text = st.text_area("Text")

if st.button("Detect"):
	test = cv.transform([text]).toarray()
	res = model.predict(test)
	print(res)
	st.success("Detected Language: " + str(res[0]))
