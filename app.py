import streamlit as st
import joblib

st.set_page_config(
    page_title="Passos Mágicos",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Passos Mágicos")
st.success("Streamlit funcionando corretamente!")

st.subheader("🤖 Teste de carregamento do modelo")

try:
    # Carrega o modelo
    modelo = joblib.load("modelo_risco.pkl")

    st.write("✅ Modelo carregado com sucesso!")
    st.write("Tipo do objeto:", type(modelo))

except Exception as e:
    st.error("❌ Erro ao carregar o modelo.")
    st.exception(e)
