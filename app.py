import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Passos Mágicos",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Passos Mágicos")
st.success("Streamlit funcionando corretamente!")

st.subheader("🤖 Teste de previsão com o modelo")

try:
    modelo = joblib.load("modelo_risco.pkl")
    st.write("✅ Modelo carregado com sucesso!")
    st.write("Tipo do objeto:", type(modelo))

    # Exemplo de entrada fictícia (ajuste depois para bater com as features reais)
    exemplo = np.zeros((1, modelo.n_features_in_))  # vetor de zeros só para teste
    prob = modelo.predict_proba(exemplo)[0][1]      # probabilidade da classe positiva

    st.write("🔍 Teste de previsão com dados fictícios:")
    st.write(f"Probabilidade de risco: {prob:.2f}")

except Exception as e:
    st.error("❌ Erro ao carregar ou prever com o modelo.")
    st.exception(e)
