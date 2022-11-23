import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
#import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
DATASET_PATH = "data/heart_2020_cleaned.csv"


def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pl.read_csv(DATASET_PATH)
        heart_df = heart_df.to_pandas()
        heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                                index=heart_df.index,
                                columns=heart_df.columns)
        return heart_df
    def user_input_features():
        race = st.sidebar.selectbox("Raça", options=("Indígena","Asiatico","Preta","Hispânico","Branca","Outros",))
        if race == "Indígena":
            AmericanIndianAlaskanNative=1
            Asian=0
            Black=0
            Hispanic=0
            OtherRace=0
            White=0
        elif race == "Asiatico":
            AmericanIndianAlaskanNative=0
            Asian=1
            Black=0
            Hispanic=0
            OtherRace=0
            White=0
        elif race == "Preta":
            AmericanIndianAlaskanNative=0
            Asian=0
            Black=1
            Hispanic=0
            OtherRace=0
            White=0
        elif race == "Hispânico":
            AmericanIndianAlaskanNative=0
            Asian=0
            Black=0
            Hispanic=1
            OtherRace=0
            White=0
        elif race == "Branca":
            AmericanIndianAlaskanNative=0
            Asian=0
            Black=0
            Hispanic=0
            OtherRace=0
            White=1
        elif race == "Outros":
            AmericanIndianAlaskanNative=0
            Asian=0
            Black=0
            Hispanic=0
            OtherRace=1
            White=0
        sex = st.sidebar.selectbox("Sexo", options=("Masculino","Feminino"))
        if sex == "Masculino":
            Male = 1
        else:
            Male = 0
        age_cat = st.sidebar.number_input("Idade", 18, 90, 18)
        if age_cat <= 18:
            AgeCategory = 18
        elif age_cat > 18 and age_cat <= 25:
            AgeCategory = 25
        elif age_cat > 26 and age_cat <= 30:
            AgeCategory = 30
        elif age_cat > 31 and age_cat <= 35:
            AgeCategory = 35
        elif age_cat > 36 and age_cat <= 40:
            AgeCategory = 40
        elif age_cat > 41 and age_cat <= 45:
            AgeCategory = 45
        elif age_cat > 46 and age_cat <= 50:
            AgeCategory = 50
        elif age_cat > 51 and age_cat <= 55:
            AgeCategory = 55
        elif age_cat > 56 and age_cat <= 60:
            AgeCategory = 60
        elif age_cat > 61 and age_cat <= 65:
            AgeCategory = 65
        elif age_cat > 66 and age_cat <= 70:
            AgeCategory = 70
        elif age_cat > 71 and age_cat <= 75:
            AgeCategory = 75
        elif age_cat > 76 and age_cat <= 999:
            AgeCategory = 80
        altura = st.sidebar.number_input("Altura em metros", min_value=0.6, value=1.50, max_value=2.40)
        peso = st.sidebar.number_input("Peso em kg", min_value=1,value=50, max_value=600)
        try:
          BMI = round(peso/(altura * altura))
          st.write('Seu IMC é: ', BMI)
        except:
          st.write('Inserir uma altura e peso válidos')
        SleepTime = st.sidebar.number_input("Quantas horas você dorme por dia?", 0, 24, 7)
        gen_health = st.sidebar.selectbox("Como você pode definir sua saúde geral?",
                                          options=("Excelente","Muito bom","Bom","Razoável","Ruim"))
        if gen_health == "Excelente":
            ExcellentHealth=1
            FairHealth=0
            GoodHealth=0
            PoorHealth=0
            VeryGoodHealth=0
        elif gen_health == "Muito bom":
            ExcellentHealth=0
            FairHealth=0
            GoodHealth=0
            PoorHealth=0
            VeryGoodHealth=1
        elif gen_health == "Bom":
            ExcellentHealth=0
            FairHealth=0
            GoodHealth=1
            PoorHealth=0
            VeryGoodHealth=0
        elif gen_health == "Razoável":
            ExcellentHealth=0
            FairHealth=1
            GoodHealth=0
            PoorHealth=0
            VeryGoodHealth=0
        elif gen_health == "Ruim":
            ExcellentHealth=0
            FairHealth=0
            GoodHealth=0
            PoorHealth=1
            VeryGoodHealth=0
        PhysicalHealth = st.sidebar.number_input("Por quantos dias durante os últimos 30 dias"
                                              " sua saúde física não estava boa?", 0, 30, 0)
        MentalHealth = st.sidebar.number_input("Por quantos dias durante os últimos 30 dias"
                                              " sua saúde mental não estava boa?", 0, 30, 0)
        phys_act = st.sidebar.selectbox("Você praticou algum tipo de esporte (corrida, academia, etc.)"
                                        " no ultimo mes?", options=("Não", "Sim"))
        if phys_act == "Sim":
            PhysicalActivity = 1
        else:
            PhysicalActivity = 0
        smoking = st.sidebar.selectbox("Você fumou pelo menos 100 cigarros em"
                                       " toda a sua vida (aprox. 5 pacotes)?)",
                                       options=("Não", "Sim"))
        if smoking == "Sim":
            Smoking = 1
        else:
            Smoking = 0
        alcohol_drink = st.sidebar.selectbox("Você toma mais de 14 doses de álcool (homens)"
                                             " ou mais de 7 (mulheres) em uma semana?", options=("Não", "Sim"))
        if alcohol_drink == "Sim":
            AlcoholDrinking = 1
        else:
            AlcoholDrinking = 0
        stroke = st.sidebar.selectbox("Você teve um derrame?", options=("Não", "Sim"))
        if stroke == "Sim":
            Stroke = 1
        else:
            Stroke = 0
        diff_walk = st.sidebar.selectbox("Você tem sérias dificuldades para andar"
                                         " ou subir escadas?", options=("Não", "Sim"))
        if diff_walk == "Sim":
            DiffWalking = 1
        else:
            DiffWalking = 0
        diabetic = st.sidebar.selectbox("Você tem diabetes?",
                                        options=("Não", "Sim"))
        if diabetic == "Sim":
            Diabetes = 1
        else:
            Diabetes = 0
        diabetsPregnancy = st.sidebar.selectbox("(Mulher) Você tem diabetes gestacional?", options=("Não", "Sim"))
        if diabetsPregnancy == "Sim":
            DiabetsDuringPregnancy = 1
        else:
            DiabetsDuringPregnancy = 0
        borderlinediabete = st.sidebar.selectbox("Você tem pré-diabetes?", options=("Não", "Sim"))
        if borderlinediabete == "Sim":
            BorderlineDiabetes = 1
        else:
            BorderlineDiabetes = 0
        asthma = st.sidebar.selectbox("Você tem asma?", options=("Não", "Sim"))
        if asthma == "Sim":
            Asthma = 1
        else:
            Asthma = 0
        kid_dis = st.sidebar.selectbox("Você tem doença renal?", options=("Não", "Sim"))
        if kid_dis == "Sim":
            KidneyDisease = 1
        else:
            KidneyDisease = 0
        skin_canc = st.sidebar.selectbox("Você tem câncer de pele?", options=("Não", "Sim"))
        if skin_canc == "Sim":
            SkinCancer = 1
        else:
            SkinCancer = 0

        respostas = [BMI, PhysicalHealth, MentalHealth, AgeCategory, SleepTime,
               Male, Asthma, AlcoholDrinking, Stroke,
               DiffWalking, BorderlineDiabetes, Diabetes,
               DiabetsDuringPregnancy, PhysicalActivity, KidneyDisease,
               SkinCancer, Smoking, ExcellentHealth, FairHealth,
               GoodHealth, PoorHealth, VeryGoodHealth,
               AmericanIndianAlaskanNative, Asian, Black, Hispanic,
               OtherRace, White]
        return respostas


    st.set_page_config(
        page_title="Previsão de doenças cardíacas",
        page_icon="images/heart-fav.png"
    )

    st.title("Previsão de doenças cardíacas")
    st.subheader("Nesse semestre nosso grupo teve como objetivo fazer uma análise de um dataset e realizar um relatório com o que foi analisado! Nesse site utilizamos inteligência artificial para fazer uma regressão logística e tentar prever a chance de ter alguma doença cardíaca "
                 )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="Vou te ajudar a diagnosticar a saúde do seu coração!",
                 width=150)
        submit = st.button("Prever")
    with col2:
        st.markdown("""
        Contribuir com a prevenção de doenças cardíacas através do mapeamento das principais características identificadas como ofensivas da incidência deste tipo de doença.

        Para usar a ferramenta é fácil!
        1. Insira os dados que melhor te descrevem.
        2. Pressione o botão "Prever" e aguarde o resultado.
        """)
    heart = load_dataset()

    x = user_input_features()

    if submit:
        heart = pd.read_csv('./data/base_v12.csv')

        heart.drop('Unnamed: 0', axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(heart.drop('HeartDisease',axis=1), heart['HeartDisease'], test_size=0.40, random_state=101)
        logging.info('Iniciando os testes')
        #smote equalizando a base
        smt = SMOTE()
        logging.info('Aplicando o SMOTE')
        X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
        logging.info('Aplicando a regressão logistica')
        logmodel = LogisticRegression(solver='lbfgs',max_iter=1000)

        logmodel.fit(X_train_sm,y_train_sm)

        predictions = logmodel.predict(X_test)

        conf_mat = confusion_matrix(y_test, predictions)
        logging.info('Predicitons finalizadas')


        EXEMPLO = np.array(x).reshape((1,-1))
        print("INIT",x,"FINISH")


        print("EXEMPLO: {}".format(logmodel.predict(EXEMPLO)[0]))
        prediction = logmodel.predict(EXEMPLO)[0]

        if prediction == 0:
            st.markdown(f"**Aparentemente você NÃO tem indicios de doença cardiaca")
            st.image("images/heart-okay.jpg",
                     caption="Seu coração parece estar bem! - Mas lembre-se isso não é um diagnostico, procure sempre um medico!")
            st.markdown("Classification Report da regressão logística")
            st.table(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())
            logging.info("Output gerado")
        else:
            st.markdown(f"**Aparentemente você TEM indicios de doença cardiaca")
            st.image("images/heart-bad.jpg",
                     caption="Você deve tomar cuidado, há indicios de que pode ter algum problema de coração! - Procure um medico e sempre o mantanha informado sobre suas condições!")
            st.markdown("Classification Report da regressão logística")
            st.table(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())
            logging.info("Output gerado")
if __name__ == "__main__":
    main()
