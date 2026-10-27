# Churn service

Сервис предсказывает, уйдёт ли клиент. Целевая переменная `churn`: `1` — клиент ушёл, `0` — остался. Модель обучается на `data/churn_dataset.csv` и сохраняется в `models/`. После запуска доступны предсказание, обучение и проверка состояния.

## Структура

- `main.py` — приложение FastAPI и загрузка сохранённой модели при старте
- `api/` — маршруты и формат ошибок
- `ml/` — чтение датасета и pipeline обучения
- `schemas/` — схемы запросов и ответов
- `core/` — пути к файлам, логирование и ошибки сервиса
- `models/` — сохранённые pipeline, карточка модели и история метрик
- `data/churn_dataset.csv` — обучающая выборка

## Датасет

Файл `data/churn_dataset.csv`, одна строка — один клиент.

| Колонка | Тип | Роль |
| --- | --- | --- |
| `monthly_fee` | число | признак |
| `usage_hours` | число | признак |
| `support_requests` | целое | признак |
| `account_age_months` | целое | признак |
| `failed_payments` | целое | признак |
| `autopay_enabled` | целое, 0 или 1 | признак |
| `region` | строка | признак |
| `device_type` | строка | признак |
| `payment_method` | строка | признак |
| `churn` | целое, 0 или 1 | целевая переменная |

В запросе `/predict` колонки `churn` нет: её как раз нужно предсказать.

## Запуск локально

Из корня проекта:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn main:app --reload
```

Сервис слушает http://127.0.0.1:8000. Документация: http://127.0.0.1:8000/docs. Проверка состояния: http://127.0.0.1:8000/health.

Тесты:

```bash
pytest
```

## Запуск в Docker

Команды тоже из корня проекта. Внутри контейнера сервис слушает порт 8000.

```bash
docker build -t churn-service .
docker run --rm -p 8000:8000 churn-service
```

На машине, где порт 8000 уже занят, опубликуйте другой порт хоста: `-p 8001:8000`. Тогда адрес будет http://127.0.0.1:8001.

## Примеры запросов

Обучить логистическую регрессию:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type":"logreg","hyperparameters":{"C":1.0,"max_iter":1000,"random_state":42}}'
```

Ответ содержит `accuracy`, `f1` и `roc_auc` на тестовой части датасета.

Предсказать отток для одного клиента:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 9.99,
    "usage_hours": 27.92,
    "support_requests": 1,
    "account_age_months": 14,
    "failed_payments": 1,
    "region": "america",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
```

`prediction` — класс `0` или `1`. `probabilities` — вероятности обоих классов. В тело можно передать и список таких объектов: тогда ответ тоже будет списком.
