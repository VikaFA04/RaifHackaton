# Finance QA Assistant (RAG) — AI for Finance Hack (Raiffeisen)

Ассистент на базе LLM для ответов на финансовые вопросы  
(вклады, облигации, деривативы и др.).

Решение основано на **Retrieval-Augmented Generation (RAG)**  
и использует базу знаний из `train_data.csv` (статьи в формате Markdown).


## Problem

Клиенты банков ежедневно задают большое количество вопросов о продуктах и услугах.  
Обработка таких запросов через колл-центры и чат-боты требует значительных ресурсов и плохо масштабируется.

При этом в финтех-домене **критически важна корректность ответов** — ошибка может привести к финансовым и юридическим рискам.


## Data

- `train_data.csv` — база знаний (≈350 статей на финансовые темы, формат Markdown)
- `questions.csv` — список пользовательских вопросов
- `submission.csv` — итоговый файл с сгенерированными ответами


## Solution overview (RAG + FAISS)

1. **Ingestion**  
   Загрузка и предварительная очистка текстов из `train_data.csv`

2. **Chunking**  
   Разбиение документов на чанки фиксированного размера с overlap

3. **Indexing**  
   Построение эмбеддингов для чанков и сохранение их в **FAISS-индекс**

4. **Retrieval**  
   Поиск top-k наиболее релевантных чанков по пользовательскому вопросу

5. **Generation**  
   Генерация ответа LLM **строго на основе retrieved-контекста**  
   (anti-hallucination prompt)

6. **Export**  
   Формирование файла `submission.csv` с ответами на вопросы из `questions.csv`

> **Rerank:** рассматривался на этапе экспериментов,  
> но в финальной версии отключён из-за trade-off между качеством и стоимостью API.


## How to run

### 1) Install dependencies
```
pip install -r requirements.txt
```
### 2)Set up environment variables
```
cp .env.example .env
# заполнитmь API-ключи
```
### 3) Build the FAISS index (один раз)
```
python -m src.faiss_index --data train_data.csv
```

### 4) Generate submission
```
python -m src.run --questions questions.csv --out submission.csv bash
```
---

## Anti-hallucination strategy

LLM получает только контекст, извлечённый на этапе retrieval

Если информации нет в retrieved-контексте, модель обязана явно указать,
что ответа нет в базе знаний

Ответы формируются в кратком и прикладном виде, без «воды»

## Tech stack
- Python
- LLM API
- FAISS
- Retrieval-Augmented Generation (RAG)

## Example answer

**Question:**  
Как изменение ключевой ставки влияет на цену облигаций?

**Model answer:**  
При повышении ключевой ставки рыночные процентные ставки растут, 
что снижает текущую стоимость будущих купонных выплат по облигации. 
В результате цена облигации, как правило, падает.
