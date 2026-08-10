import streamlit as st
import pandas as pd

# Configuração inicial da página
st.set_page_config(
    page_title="Passos Mágicos",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Passos Mágicos")
st.success("Streamlit funcionando corretamente!")

# --- Etapa 2: Carregar a base ---
st.subheader("📊 Carregando a base consolidada")

try:
    # Lê o CSV
    df = pd.read_csv("base_datathon_consolidada.csv")

    # Mostra informações básicas
    st.write("✅ Base carregada com sucesso!")
    st.write(f"Quantidade de registros: {len(df)}")
    st.write("Visualização inicial da base:")
    st.dataframe(df.head())

except Exception as e:
    st.error("❌ Erro ao carregar a base.")
    st.exception(e)
