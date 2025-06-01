# analysis_and_model.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo
import os

def analysis_and_model_page():
    st.title("Анализ данных и построение модели")

    # 1) Выбор источника данных
    data_option = st.radio("Источник данных:", ("Встроенный датасет", "Загрузить CSV"))

    if data_option == "Встроенный датасет":
        try:
            ds = fetch_ucirepo(id=601)
            data = pd.concat([ds.data.features, ds.data.targets], axis=1)
        except Exception:
            st.error("Не удалось загрузить встроенный датасет. Проверьте интернет-соединение.")
            return
    else:
        uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
        if uploaded_file is None:
            st.info("Ожидание загрузки файла...")
            return
        data = pd.read_csv(uploaded_file)

    # 2) Удаляем лишние столбцы, если они есть
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors='ignore')

    # 3) Проверяем обязательные колонки
    if 'Type' not in data.columns or 'Machine failure' not in data.columns:
        st.error("В данных отсутствуют столбцы 'Type' или 'Machine failure'.")
        st.write("Найдены столбцы:", data.columns.tolist())
        return

    # 4) Кодируем Type
    le = LabelEncoder()
    data['Type'] = le.fit_transform(data['Type'])

    # 5) Заполняем пропуски (если есть)
    if data.isnull().sum().sum() > 0:
        data = data.fillna(data.mean())

    # 6) Переименуем все колонки, чтобы убрать пробелы, скобки, %, <, > и т. п.
    def sanitize_col(col_name: str) -> str:
        return (col_name
                .replace(' ', '_')
                .replace('[', '')
                .replace(']', '')
                .replace('(', '')
                .replace(')', '')
                .replace('%', '')
                .replace('<', '')
                .replace('>', ''))

    data = data.rename(columns=lambda x: sanitize_col(x))

    # 7) Показываем переименованные столбцы (для наглядности)
    #st.write("Переименованные столбцы:", data.columns.tolist())

    # 8) Список кандидатов на 5 числовых признаков (без единиц и с единицами)
    candidates = {
        'Air_temperature': ['Air_temperature', 'Air_temperature_K'],
        'Process_temperature': ['Process_temperature', 'Process_temperature_K'],
        'Rotational_speed': ['Rotational_speed', 'Rotational_speed_rpm'],
        'Torque': ['Torque', 'Torque_Nm'],
        'Tool_wear': ['Tool_wear', 'Tool_wear_min']
    }

    # 9) Для каждого признака выбираем первое найденное название из кандидатов
    present_num_cols = []
    missing_features = []
    for feature_key, options in candidates.items():
        found = next((opt for opt in options if opt in data.columns), None)
        if found:
            present_num_cols.append(found)
        else:
            missing_features.append(feature_key)

    if missing_features:
        st.error(f"Отсутствуют ожидаемые числовые признаки: {missing_features}")
        st.write("Найдены столбцы:", data.columns.tolist())
        return

    # 10) Масштабируем выбранные числовые признаки
    scaler = StandardScaler()
    data[present_num_cols] = scaler.fit_transform(data[present_num_cols])

    # 11) Предпросмотр готовых данных
    st.subheader("Предпросмотр данных (после предобработки)")
    st.dataframe(data.head())

    # 12) Разделение на X и y
    X = data.drop(columns=['Machine_failure'])
    y = data['Machine_failure']

    # 13) Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 14) Обучение моделей
    st.subheader("Обучение моделей")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        ),

        "SVM": SVC(probability=True, random_state=42)
    }

    best_model = None
    best_auc = 0
    results = []

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results.append((name, acc, auc))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # 15) Таблица с метриками
    res_df = pd.DataFrame(results, columns=["Модель", "Accuracy", "ROC-AUC"])
    st.table(res_df)

    # 16) ROC-кривые
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые")
    plt.legend()
    # Сохраняем ROC-кривую
    if not os.path.isdir("images"):
        os.mkdir("images")
    plt.savefig("images/roc_curve.png", dpi=150, bbox_inches="tight")
    st.pyplot(plt)

    st.success(f"🟢 Лучшая модель: {best_model.__class__.__name__} (ROC-AUC={best_auc:.3f})")

    # 17) Матрица ошибок
    st.subheader("Матрица ошибок (лучшая модель)")
    y_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_best)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинные")
    ax.set_title("Матрица ошибок")
    # Сохраняем матрицу ошибок
    fig.savefig("images/confusion_matrix.png", dpi=150, bbox_inches="tight")
    st.pyplot(fig)

    # 18) Отчет о классификации
    st.subheader("Отчет о классификации (лучшая модель)")
    cr = classification_report(y_test, y_best, output_dict=True)
    st.table(pd.DataFrame(cr).transpose())

    # 19) Форма для ручного предсказания
    st.subheader("Предсказание для нового наблюдения")
    with st.form("prediction_form"):
        type_val = st.selectbox("Type", options=le.classes_.tolist())
        # Значения на входе мы всё равно просим в привычных единицах (K, rpm, Nm, min)
        air_temp = st.number_input("Air temperature [K]", value=300.0)
        process_temp = st.number_input("Process temperature [K]", value=310.0)
        rotational_speed = st.number_input("Rotational speed [rpm]", value=1500)
        torque = st.number_input("Torque [Nm]", value=40.0)
        tool_wear = st.number_input("Tool wear [min]", value=100)
        submit = st.form_submit_button("Предсказать")

    if submit:
        # 20) Создаём DataFrame с теми же именами столбцов, что и в тренировке
        #    ‘present_num_cols’ уже содержит реальные названия, которые мы нашли в data:
        #    например: ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        #
        #    Поэтому для каждого индекса i из present_num_cols мы берём соответствующее значение
        #    (air_temp, process_temp, ...) по порядку.
        new_df_raw = pd.DataFrame({
            'Type': [le.transform([type_val])[0]],
            present_num_cols[0]: [air_temp],
            present_num_cols[1]: [process_temp],
            present_num_cols[2]: [rotational_speed],
            present_num_cols[3]: [torque],
            present_num_cols[4]: [tool_wear]
        })

        # 21) Сразу масштабируем эти же колонки через тот же scaler
        try:
            new_df_raw[present_num_cols] = scaler.transform(new_df_raw[present_num_cols])
        except Exception as e:
            st.error(f"Ошибка при масштабировании: {e}")
            st.write("present_num_cols:", present_num_cols)
            st.write("Колонки new_df_raw:", new_df_raw.columns.tolist())
            return

        # 22) Предсказание
        #    Теперь в new_df_raw ровно такие же признаки, какие были у X_train после предобработки
        pred = best_model.predict(new_df_raw)[0]
        proba = best_model.predict_proba(new_df_raw)[0][1]
        st.write(f"**Результат:** {'Отказ (1)' if pred == 1 else 'Без отказа (0)'}")
        st.write(f"**Вероятность отказа:** {proba:.2%}")
