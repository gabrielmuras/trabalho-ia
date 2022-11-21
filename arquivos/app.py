import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import sklearn

DATASET_PATH = "data/heart_2020_cleaned.csv"
LOG_MODEL_PATH = "model/logistic_regression.pkl"


def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pl.read_csv(DATASET_PATH)
        heart_df = heart_df.to_pandas()
        heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                                index=heart_df.index,
                                columns=heart_df.columns)
        return heart_df


    def user_input_features() -> pd.DataFrame:
        race = st.sidebar.selectbox("Raça", options=("Indígena","Asiatico","Preta","Hispânico","Branca","Outros",))
        if race == "Indígena":
            race = 'American Indian/Alaskan Native'
        elif race == "Asiatico":
            race = 'Asian'
        elif race == "Preta":
            race = "Black"
        elif race == "Hispânico":
            race = "Hispanic"
        elif race == "Branca":
            race = "White"
        elif race == "Outros":
            race = "Other"
        sex = st.sidebar.selectbox("Sexo", options=("",""))
        print(heart.Sex.unique())
        age_cat = st.sidebar.selectbox("Idade",
                                       options=(age_cat for age_cat in heart.AgeCategory.unique()))
        print(heart.AgeCategory.unique())
        bmi_cat = st.sidebar.selectbox("IMC",
                                       options=(bmi_cat for bmi_cat in heart.BMICategory.unique()))
        print(heart.BMICategory.unique())
        sleep_time = st.sidebar.number_input("Quantas horas você dorme por dia?", 0, 24, 7)
        gen_health = st.sidebar.selectbox("Como você pode definir sua saúde geral?",
                                          options=("","","","",""))
        print(heart.GenHealth.unique())
        phys_health = st.sidebar.number_input("Por quantos dias durante os últimos 30 dias"
                                              " sua saúde física não estava boa?", 0, 30, 0)
        ment_health = st.sidebar.number_input("Por quantos dias durante os últimos 30 dias"
                                              " sua saúde mental não estava boa?", 0, 30, 0)
        phys_act = st.sidebar.selectbox("Você praticou algum tipo de esporte (corrida, academia, etc.)"
                                        " no ultimo mes?", options=("Não", "Sim"))
        if phys_act == "Sim":
            phys_act = "Yes"
        else:
            phys_act = "No"
        smoking = st.sidebar.selectbox("Você fumou pelo menos 100 cigarros em"
                                       " toda a sua vida (aprox. 5 pacotes)?)",
                                       options=("Não", "Sim"))
        if smoking == "Sim":
            smoking = "Yes"
        else:
            smoking = "No"        
        alcohol_drink = st.sidebar.selectbox("Você toma mais de 14 doses de álcool (homens)"
                                             " ou mais de 7 (mulheres) em uma semana?", options=("Não", "Sim"))
        if alcohol_drink == "Sim":
            alcohol_drink = "Yes"
        else:
            alcohol_drink = "No"        
        stroke = st.sidebar.selectbox("Você teve um derrame?", options=("Não", "Sim"))
        if stroke == "Sim":
            stroke = "Yes"
        else:
            stroke = "No"        
        diff_walk = st.sidebar.selectbox("Você tem sérias dificuldades para andar"
                                         " ou subir escadas?", options=("Não", "Sim"))
        if diff_walk == "Sim":
            diff_walk = "Yes"
        else:
            diff_walk = "No"        
        diabetic = st.sidebar.selectbox("Você já teve diabetes?",
                                        options=(diabetic for diabetic in heart.Diabetic.unique()))
        print(heart.Diabetic.unique())
        asthma = st.sidebar.selectbox("Você tem asma?", options=("Não", "Sim"))
        if asthma == "Sim":
            asthma = "Yes"
        else:
            asthma = "No"        
        kid_dis = st.sidebar.selectbox("Você tem doença renal?", options=("Não", "Sim"))
        if kid_dis == "Sim":
            kid_dis = "Yes"
        else:
            kid_dis = "No"        
        skin_canc = st.sidebar.selectbox("Você tem câncer de pele?", options=("Não", "Sim"))
        if skin_canc == "Sim":
            skin_canc = "Yes"
        else:
            skin_canc = "No"        

        features = pd.DataFrame({
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "SleepTime": [sleep_time],
            "BMICategory": [bmi_cat],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [phys_act],
            "GenHealth": [gen_health],
            "Asthma": [asthma],
            "KidneyDisease": [kid_dis],
            "SkinCancer": [skin_canc]
        })

        return features


    st.set_page_config(
        page_title="Previsão de doenças cardíacas",
        page_icon="images/heart-fav.png"
    )

    st.title("Previsão de doenças cardíacas")
    st.subheader("Nesse semestre nosso grupo teve como objetivo fazer uma analise de uma tabela e realizar um relatorio com o que foi analisado! "
                 )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="Vou te ajudar a diagnosticar a saúde do seu coração!",
                 width=150)
        submit = st.button("Prever")
    with col2:
        st.markdown("""
        Descrição da tabela aqui 
        
        Para usar a ferramenta é fácil! 
        1. Insira os dados que melhor descreve você
        2. Pressione o botão "Prever" e aguarde o resultado. 
        """)

    heart = load_dataset()

    # st.sidebar.title("Feature Selection")
    # st.sidebar.image("images/heart-sidebar.png", width=100)

    input_df = user_input_features()
    df = pd.concat([input_df, heart], axis=0)
    df = df.drop(columns=["HeartDisease"])

    cat_cols = ["BMICategory", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
                "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

    df = df[:1]
    df.fillna(0, inplace=True)

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    if submit:
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        if prediction == 0:
            st.markdown(f"**A probabilidade de você ter"
                        f" doença cardíaca é {round(prediction_prob[0][1] * 100, 2)}%."
                        f" Você é saudável!**")
            st.image("images/heart-okay.jpg",
                     caption="Seu coração parece estar bem! - Mas lembre-se isso não é um diagnostico, procure sempre um medico!")
        else:
            st.markdown(f"**A probabilidade de você ter"
                        f" doença cardíaca é {round(prediction_prob[0][1] * 100, 2)}%."
                        f" Parece que você não está saudável.**")
            st.image("images/heart-bad.jpg",
                     caption="Você deve tomar cuidado, há indicios de que pode ter algum problema de coração! - Procure um medico e sempre o mantanha informado sobre suas condições!")

if __name__ == "__main__":
    main()
