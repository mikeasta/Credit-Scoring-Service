import os
import requests
import streamlit as st

# Get backend url
BACKEND_HOST = os.getenv("BACKEND_HOST")
BACKEND_PORT = os.getenv("BACKEND_PORT")
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Draw frontend
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
    name_surname = st.text_input("Ваше имя и фамилия")
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
        "name_surname": name_surname,
        "age": age,
        "education": education,
        "income": income,
        "has_car": int(has_car),
        "has_work": int(has_work),
        "has_passport": int(has_passport),
    }

    try:
        response = requests.post(
            url=f"{BACKEND_URL}/score",
            json=data,
            timeout=10
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        st.success(result["message"])
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при подключении к серверу: {str(e)}")
    except ValueError as e:
        st.error(f"Ошибка при обработке ответа сервера: {str(e)}")
    