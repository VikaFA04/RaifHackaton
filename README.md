# Finance QA Assistant (RAG) — RaifHackaton

LLM-ассистент для ответов на вопросы о банковских продуктах и финансовых терминах.
Использует Retrieval-Augmented Generation (RAG) поверх базы знаний из `train_data.csv` (350 статей в Markdown).

## Problem
Клиенты задают вопросы по вкладам, кредитам, комиссиям, инвестициям и страхованию.
Классические каналы поддержки (колл-центр/чат-боты) дорогие и масштабируются плохо.
Ключевое требование: **корректность** ответов и контроль hallucinations.

## Data
- `train_data.csv` — база знаний (350 статей, Markdown)
- `questions.csv` — вопросы для генерации ответов
- `baseline.py` — предоставленное стартовое решение

## Solution overview (RAG)
1. Ingestion: загрузка и очистка markdown-текста
2. Chunking: нарезка документов на чанки с overlap
3. Indexing: эмбеддинги чанков + векторный индекс
4. Retrieval: top-k релевантных чанков по запросу
5. Generation: ответ LLM строго по retrieved-контексту (anti-hallucination prompt)
6. Output: `submission.csv` с ответами на `questions.csv`

## How to run
1) Install:
```bash
pip install -r requirements.txt
