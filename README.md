# Finance QA Assistant (RAG) — AI for Finance Hack (Raiffeisen)

Ассистент на базе LLM для ответов на финансовые вопросы (вклады, облигации, деривативы и т.д.).
Основан на Retrieval-Augmented Generation (RAG) поверх базы знаний из `train_data.csv` (Markdown-статьи).

## Problem
Обработка вопросов через колл-центры/чат-боты дорога и плохо масштабируется.
Важно обеспечивать корректность ответов — в финтехе ошибка критична.

## Data
- `train_data.csv` — база знаний (≈350 статей в Markdown)
- `questions.csv` — вопросы, для которых нужно сформировать ответы
- Output: `submission.csv`

## Solution overview (RAG + FAISS)
1. Ingestion: чтение `train_data.csv`, очистка и нормализация Markdown
2. Chunking: разбиение документов на чанки фиксированного размера с overlap
3. Indexing: эмбеддинги чанков + FAISS индекс
4. Retrieval: поиск top-k релевантных чанков по вопросу
5. Generation: генерация ответа LLM строго по retrieved-контексту (anti-hallucination prompt)
6. Export: сохранение `submission.csv`

> Rerank: пробовали, но в финальной версии отключён (стоимость/скорость vs выигрыш качества).

## How to run
### 1) Install
```bash
pip install -r requirements.txt```

### 2) Setup env
```cp .env.example .env
# заполнить API ключи```

 ### 3) Build index (once)
```python -m src.faiss_index --data train_data.csv```

### 4) Generate submission
```python -m src.run --questions questions.csv --out submission.csv```


## Anti-hallucination

LLM получает только retrieved-контекст
Если ответа нет в контексте — модель обязана сказать, что информации недостаточно
Ответ должен быть кратким и прикладным (без “воды”)
