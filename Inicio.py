import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("🔍 Demo TF-IDF en Español")

# Documentos de ejemplo
default_docs = """La paella se prepara con arroz, azafrán y mariscos frescos.
El chocolate caliente se hace con leche y cacao en polvo.
La ensalada lleva lechuga, tomate, pepino y aceite de oliva.
El pollo asado se hornea con ajo, limón y hierbas aromáticas.
La sopa de lentejas se cocina con verduras y especias.
El pan artesanal se elabora con harina, agua, sal y levadura."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "arroz azafrán mariscos")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")

    if st.button("arroz azafrán mariscos", use_container_width=True):
        st.session_state.question = "arroz azafrán mariscos"
        st.rerun()

    if st.button("leche cacao chocolate", use_container_width=True):
        st.session_state.question = "leche cacao chocolate"
        st.rerun()

    if st.button("lechuga tomate ensalada", use_container_width=True):
        st.session_state.question = "lechuga tomate ensalada"
        st.rerun()

    if st.button("pollo ajo limón horno", use_container_width=True):
        st.session_state.question = "pollo ajo limón horno"
        st.rerun()

    if st.button("lentejas verduras sopa", use_container_width=True):
        st.session_state.question = "lentejas verduras sopa"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )

        X = vectorizer.fit_transform(documents)

        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 🎯 Respuesta")
        st.markdown(f"**Tu pregunta:** {question}")

        if best_score > 0.01:
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")
