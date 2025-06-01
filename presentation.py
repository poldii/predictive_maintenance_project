# presentation.py

import streamlit as st


def presentation_page():
    st.title("Презентация проекта")

    # Используем tabs для организации слайдов
    tabs = st.tabs([
        "Введение",
        "Методология",
        "Результаты",
        "Демонстрация"
    ])

    # --- Слайд 1: Введение ---
    with tabs[0]:
        st.header("Прогнозирование отказов оборудования")
        st.markdown("""
        - **Задача**: Предсказание отказов промышленного оборудования  
        - **Цель**: Снижение простоев и затрат на обслуживание  
        - **Датасет**: 10,000 записей с 14 признаками  
        """)
        # Отображаем сохранённый ROC-график из раздела «Анализ и модель»
        st.image(
            "images/roc_curve.png",
            caption="ROC-кривые",
            use_container_width=True
        )

    # --- Слайд 2: Методология ---
    with tabs[1]:
        st.header("Методология")
        st.markdown("""
        1. Загрузка и предобработка данных  
        2. Обучение моделей классификации  
        3. Оценка метрик качества  
        4. Развертывание Streamlit-приложения  
        """)
        st.code("""
# Пример кода обучения RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)
        """)

    # --- Слайд 3: Результаты ---
    with tabs[2]:
        st.header("Результаты")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Лучшая модель", "XGBoost")
            st.metric("Accuracy", "0.9865")
        with col2:
            st.metric("ROC-AUC", "0.9771")
            st.metric("Время обучения", "≈5 сек")
        # Отображаем сохранённую матрицу ошибок
        st.image(
            "images/confusion_matrix.png",
            caption="Матрица ошибок",
            use_container_width=True
        )
        # Дополнительно можно добавить диаграмму сравнения:
        st.bar_chart({
            "Logistic Regression": 0.9005,
            "Random Forest": 0.9641,
            "XGBoost": 0.9771,
            "SVM": 0.9501
        })

    # --- Слайд 4: Демонстрация ---
    with tabs[3]:
        st.header("Демонстрация")
        # Воспроизводим локальное видео (положите demo.mp4 в папку video/)
        st.video("video/demo.mp4")
        # Ссылка-якорь на страницу «Анализ и модель» (название страницы в sidebar)
        st.markdown(
            "[Перейти к анализу и модели ➡️](#анализ-и-модель)",
            unsafe_allow_html=True
        )
