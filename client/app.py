import streamlit as st
import requests

st.title("Кредитная карта Premium")
st.write("Новая кредитная карта с мгновенным одобрением и ставкой под ___3% годовых!___")

education_types = [
    "Нет образования", 
    "Среднее", 
    "Среднее-специальное", 
    "Неоконченное высшее/Бакалавриат",
    "Магистратура",
    "Аспирантура"
]

with st.form("Подать заявку"):
    age = st.number_input("Ваш возраст", min_value=18)
    education = st.radio(
        "Ваш уровень образования",
        education_types,
        width=500
    )
    education = education_types.index(education)
    income = st.number_input("Ваш ежемесячный доход", min_value=0)
    has_work = st.checkbox("Вы трудоустроены?")
    has_car = st.checkbox("У вас есть личный автомобиль?")
    has_passport = st.checkbox("У вас есть заграничный паспорт?")
    submit = st.form_submit_button("Подать заявку")

if submit:
    data = {
        "age": age,
        "education": education,
        "income": income,
        "has_car": int(has_car),
        "has_passport": int(has_passport),
        "has_work": int(has_work)
    }

    response = requests.post(
        url="http://127.0.0.1:8000/score",
        json=data
    ).json()

    if response["approved"]:
        st.success("Ваша заявка одобрена")
    else:
        st.success("Мы подобрали для вас альтернативный вариант - карта с 5% кэшбеком во всех супермаркетах Санкт-Петербурга")

