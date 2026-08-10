import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Passos Mágicos", page_icon="📚", layout="wide")

# --- Carregar base e modelo ---
@st.cache_data
def load_data():
    df = pd.read_csv("base_datathon_consolidada.csv")

    # Corrigir colunas numéricas com vírgula
    for col in ["inde","ian","ida","ieg","iaa","ips","ipp","ipv","matematica","portugues","ingles","cg","cf","ct"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace("nan", None)
                .astype(float)
            )
    return df

@st.cache_resource
def load_model():
    return joblib.load("modelo_risco.pkl")

df = load_data()
modelo = load_model()

# --- Menu lateral ---
menu = st.sidebar.radio("Navegação", ["📊 Visão Geral", "🎯 Previsão de Risco", "👥 Risco por Aluno", "🤖 Sobre o Modelo"])

# --- 📊 Visão Geral ---
if menu == "📊 Visão Geral":
    st.title("📊 Visão Geral da Base")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de alunos", len(df))
    col2.metric("% em defasagem", f"{df['em_defasagem'].mean()*100:.1f}%")
    col3.metric("Idade média", f"{df['idade'].mean():.1f} anos")
    col4.metric("INDE médio", f"{df['inde'].mean():.2f}")

    st.subheader("Distribuição por Pedra")
    st.bar_chart(df["pedra"].value_counts())

    st.subheader("Amostra da base")
    st.dataframe(df.head())

# --- 🎯 Previsão de Risco ---
elif menu == "🎯 Previsão de Risco":
    st.title("🎯 Previsão de Risco Individual")

    st.write("Preencha os indicadores do aluno:")

    entrada = {}
    # Exemplo: tratar alguns campos categóricos
    entrada["genero"] = st.selectbox("Gênero", ["Masculino","Feminino"])
    entrada["pedra"] = st.selectbox("Pedra", df["pedra"].unique())
    entrada["situacao"] = st.selectbox("Situação", df["situacao"].dropna().unique())

    # Campos numéricos
    for col in ["idade","inde","ian","ida","ieg","iaa","ips","ipp","ipv","matematica","portugues","ingles"]:
        entrada[col] = st.number_input(col, value=0.0)

    if st.button("Calcular risco"):
        entrada_df = pd.DataFrame([entrada])
        prob = modelo.predict_proba(entrada_df)[0][1]
        risco = "Baixo"
        if prob >= 0.35 and prob < 0.6:
            risco = "Atenção"
        elif prob >= 0.6:
            risco = "Alto"
        st.success(f"Probabilidade de risco: {prob:.2f} → {risco}")

# --- 👥 Risco por Aluno ---
elif menu == "👥 Risco por Aluno":
    st.title("👥 Ranking de Risco por Aluno")

    try:
        probs = modelo.predict_proba(df[modelo.feature_names_in_])[:,1]
        df["prob_risco"] = probs
        df["nivel_risco"] = pd.cut(df["prob_risco"], bins=[0,0.35,0.6,1], labels=["Baixo","Atenção","Alto"])

        filtro = st.selectbox("Filtrar por nível de risco", ["Todos","Baixo","Atenção","Alto"])
        df_filtrado = df if filtro=="Todos" else df[df["nivel_risco"]==filtro]

        st.dataframe(df_filtrado[["ra","nome","idade","pedra","prob_risco","nivel_risco"]].sort_values("prob_risco",ascending=False))

    except Exception as e:
        st.error("Erro ao calcular risco para os alunos.")
        st.exception(e)

# --- 🤖 Sobre o Modelo ---
elif menu == "🤖 Sobre o Modelo":
    st.title("🤖 Sobre o Modelo")
    st.markdown("""
    - Modelo: Random Forest Classifier em Pipeline
    - Validação temporal: treino 2022→2023, teste 2023→2024
    - Métricas:
        - ROC-AUC: 0,855
        - PR-AUC: 0,570
        - Recall: 77,3%
        - Precision: 44,5%
        - F1: 56,5%
    - Threshold: 35%
    """)
