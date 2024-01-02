import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
nltk.download('stopwords')

def generate_summary(text):
    text = text.replace('\n', '')
    sentences = re.split('\. |\.', text)

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokenized = (tokenizer.tokenize(s.lower()) for s in sentences)
    tokenized_sentences = [list(token) for token in tokenized]

    listStopword = set(stopwords.words('indonesian'))
    important_token = []
    for sent in tokenized_sentences:
        filtered = [s for s in sent if s not in listStopword]
        important_token.append(filtered)

    sw_removed = [' '.join(t) for t in important_token]

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_sent = [stemmer.stem(sent) for sent in sw_removed]

    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(stemmed_sent)

    lsa = TruncatedSVD(n_components=5, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    lsa_sum = lsa_matrix.sum(axis=1)

    total_sentences = len(stemmed_sent)
    top_n_percent = 0.25
    top_n = int(total_sentences * top_n_percent)
    top_sentences_indices = np.argsort(lsa_sum)[-top_n:]

    summary = [sentences[i] for i in top_sentences_indices]
    return summary


<<<<<<< HEAD
def main(): 
=======
def main():
>>>>>>> 76d01ca6bf381134e87912745b657ab925aa76e6
    st.title("Peringkas Berita")

    text_input = st.text_area("Masukkan teks:", "Input Text")

    if st.button("Hasilkan Ringkasan"):

        summary = generate_summary(text_input)

        st.subheader("Hasil Ringkasan:")
        for sentence in summary:
            st.write(sentence)

if __name__ == "__main__":
    main()
