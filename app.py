import streamlit as st
import joblib
import pandas as pd

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

    # Criar um DataFrame fictício com as mesmas colunas que o modelo espera
    # (por enquanto vamos usar apenas uma linha com valores nulos ou zeros)
    # Ajuste depois para bater com as features reais da base
    colunas = modelo.feature_names_in_  # pega os nomes das features esperadas
    exemplo_df = pd.DataFrame([[0]*len(colunas)], columns=colunas)

    prob = modelo.predict_proba(exemplo_df)[0][1]  # probabilidade da classe positiva

    st.write("🔍 Teste de previsão com dados fictícios:")
    st.write(f"Probabilidade de risco: {prob:.2f}")

except Exception as e:
    st.error("❌ Erro ao carregar ou prever com o modelo.")
    st.exception(e)
