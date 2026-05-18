import os
import json
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI

MODEL_ID = 'gpt-4o-mini'
EMBED_MODEL = 'all-MiniLM-L6-v2'


def load_rag_index(models_dir: str, processed_dir: str):
    index = faiss.read_index(str(Path(models_dir) / 'nhtsa_faiss.index'))
    df = pd.read_csv(Path(processed_dir) / 'nhtsa_complaints_processed.csv')
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, df['text'].tolist(), embedder


def retrieve(query: str, index, texts: list, embedder, k: int = 3) -> list[str]:
    q_emb = embedder.encode([query]).astype('float32')
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, k)
    return [texts[i][:400] for i in indices[0] if i < len(texts)]


def generate_report(
    cv_result: dict,
    ml_result: dict,
    vehicle_info: dict,
    index,
    texts: list,
    embedder,
    api_key: str
) -> dict:
    client = OpenAI(api_key=api_key)

    query = f"{cv_result['damage_class']} {vehicle_info.get('make', '')} {vehicle_info.get('location', '')}"
    context = '\n---\n'.join(retrieve(query, index, texts, embedder))

    system_prompt = "You are an insurance claims assessor. Always respond with valid JSON only. No text outside the JSON."

    user_prompt = f"""Generate a structured vehicle damage report as JSON.

VEHICLE INFORMATION:
Make: {vehicle_info.get('make', 'Unknown')}
Model: {vehicle_info.get('model', 'Unknown')}
Year: {vehicle_info.get('year', 'Unknown')}
Vehicle value: ${vehicle_info.get('value_usd', 0):,}
Damage type: {cv_result['damage_class']} (AI confidence: {cv_result['confidence']:.0%})
Damage location: {vehicle_info.get('location', 'unspecified')}
Estimated repair cost: ${ml_result['estimated_cost_usd']:,.0f} (range: ${ml_result['cost_range_low']:,.0f}–${ml_result['cost_range_high']:,.0f})

RELEVANT HISTORICAL CONTEXT:
{context}

Return this exact JSON:
{{
  "vehicle": {{"make": "", "model": "", "year": 0, "value_usd": 0}},
  "damage": {{"type": "", "location": "", "severity": "", "description": ""}},
  "assessment": {{"repair_recommendation": "", "estimated_cost_usd": 0, "cost_range": "", "confidence_level": ""}},
  "notes": ""
}}"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=700,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        response_format={'type': 'json_object'}
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {'raw_output': raw, 'error': 'JSON parse failed'}
