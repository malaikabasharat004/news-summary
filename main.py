from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
from newspaper import Article
import aiohttp
import asyncio
from gtts import gTTS
from io import BytesIO
from starlette.responses import StreamingResponse

app = FastAPI()



# Replace with your actual values
API_KEY = "AIzaSyC0GwA3L5EELNiGvEZeVPZR0Xf1dkZfI-A"
PROJECT_ID = "932776620592"
MODEL_ID = "gemini-pro"
VERTEX_ENDPOINT = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/{MODEL_ID}:predict?key={API_KEY}"

article_storage = {}

class ArticleRequest(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = "en"

class QuestionRequest(BaseModel):
    question: str
    language: str

class SummaryRequest(BaseModel):
    max_words: int = 200
    language: str

async def fetch_article_text(url):
    loop = asyncio.get_event_loop()
    try:
        article = Article(url)
        await loop.run_in_executor(None, article.download)
        await loop.run_in_executor(None, article.parse)
        return article.text, article.authors, article.source_url
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to fetch article")

async def fetch_article_by_title(title):
    search_key = "YOUR_GOOGLE_SEARCH_API_KEY"
    search_engine_id = "YOUR_CUSTOM_SEARCH_ENGINE_ID"
    search_url = f"https://www.googleapis.com/customsearch/v1?q={requests.utils.quote(title)}&key={search_key}&cx={search_engine_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                response.raise_for_status()
                search_results = await response.json()

                if 'items' not in search_results or not search_results['items']:
                    return None, None, None

                article_url = search_results['items'][0].get("link")
                return await fetch_article_text(article_url)
    except Exception:
        raise HTTPException(status_code=500, detail="Error fetching articles")

async def call_vertex_ai(prompt: str, system_message: str = "", temperature=0.7, max_tokens=512):
    payload = {
        "instances": [{
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        }],
        "parameters": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(VERTEX_ENDPOINT, json=payload) as resp:
            data = await resp.json()
            try:
                return data["predictions"][0]["content"]
            except Exception:
                raise HTTPException(status_code=500, detail=f"LLM response error: {data}")

async def translate_text(text, target_language):
    return await call_vertex_ai(
        prompt=text,
        system_message=f"Translate the following text to {target_language}."
    )

def generate_audio(text: str, language: str) -> StreamingResponse:
    try:
        tts = gTTS(text=text, lang=language)
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        return StreamingResponse(audio_stream, media_type="audio/mpeg")
    except Exception:
        raise HTTPException(status_code=500, detail="Error generating audio")

@app.post("/article")
async def analyze_article(request: ArticleRequest):
    if not request.url and not request.title:
        raise HTTPException(status_code=400, detail="URL or title must be provided")

    if request.url:
        article_text, authors, source_url = await fetch_article_text(request.url)
    else:
        article_text, authors, source_url = await fetch_article_by_title(request.title)

    if request.language != "en":
        article_text = await translate_text(article_text, request.language)

    article_storage['text'] = article_text

    return {
        "article_text": article_text,
        "authors": authors,
        "source_url": source_url
    }

@app.post("/article_audio")
async def article_audio():
    if 'text' not in article_storage:
        raise HTTPException(status_code=400, detail="No article text available")

    return generate_audio(article_storage['text'], "en")

@app.post("/summary_audio")
async def summary_audio():
    if 'summary' not in article_storage:
        raise HTTPException(status_code=400, detail="No summary available")

    return generate_audio(article_storage['summary'], "en")

@app.post("/question")
async def ask_question(request: QuestionRequest):
    if 'text' not in article_storage:
        raise HTTPException(status_code=400, detail="No article text available")

    prompt = f"{article_storage['text']}\n\nQuestion: {request.question}"
    answer = await call_vertex_ai(
        prompt=prompt,
        system_message="You are an AI assistant who knows everything."
    )

    if request.language != "en":
        answer = await translate_text(answer, request.language)

    return {"answer": answer}

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    if 'text' not in article_storage:
        raise HTTPException(status_code=400, detail="No article text available")

    summary = await call_vertex_ai(
        prompt=article_storage['text'],
        system_message="Summarize the following article in a concise manner:",
        max_tokens=512
    )

    if request.language != "en":
        summary = await translate_text(summary, request.language)

    article_storage['summary'] = summary

    return {"summary": summary}
