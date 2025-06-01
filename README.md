# Проект: Прогнозирование отказов оборудования

## Описание
Данное приложение предсказывает, произойдет ли отказ промышленного оборудования (метка 1) или нет (метка 0). 
Используется датасет AI4I 2020 Predictive Maintenance.

## Установка и запуск
```bash
git clone https://github.com/poldii/predictive_maintenance_project.git
cd predictive_maintenance_project
pip install -r requirements.txt
streamlit run app.py
```

## Структура
- `app.py` — основной скрипт с навигацией  
- `analysis_and_model.py` — страница анализа данных и обучения моделей  
- `presentation.py` — страница-презентация   
- `requirements.txt` — зависимости  
- `data/predictive_maintenance.csv` — локальный датасет (можно загружать через UCI)  
- `video/demo.mp4` — видео-демонстрация 

## Датасет
Используется **AI4I Predictive Maintenance**  
- Источник: https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset  
- Включены признаки:  
  - `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`  

## Видео-демонстрация
Файл `video/demo.mp4` содержит короткое (1–2 минуты) видео, показывающее:
1. Клонирование репозитория  
2. Установку зависимостей  
3. Запуск Streamlit  
4. Загрузку/просмотр данных  
5. Обучение модели и вывод метрик  
6. Ручное предсказание на новых данных
