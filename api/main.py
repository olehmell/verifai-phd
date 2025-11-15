from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
import sys
import os
import uuid

# Add the parent directory to sys.path so we can import verifai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze_content
from verifai.utils.logging import get_logger

app = FastAPI(
    title="VerifAI API",
    description="API for analyzing content with VerifAI multi-agent system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000, description="Content to analyze")

    @field_validator('content')
    @classmethod
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v


class AnalyzeResponse(BaseModel):
    manipulation: bool
    techniques: list[str]
    explanation: str
    disinfo: list[str]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/analyze-test", response_model=AnalyzeResponse)
async def analyze_test(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Mock endpoint for testing extension - returns example data immediately
    """
    import asyncio
    # Simulate a small delay
    await asyncio.sleep(0.5)
    
    return {
        "manipulation": True,
        "techniques": [
            "emotional_manipulation",
            "fear_appeals",
            "selective_truth"
        ],
        "explanation": "Контент містить високу ймовірність маніпуляції (0.950), використовуючи техніки емоційного маніпулювання, залякування та вибіркової правди. Основний наратив полягає в тому, що Офіс Президента нібито ініціює обшуки у Віталія Кличка та його оточення як форму політичної розправи, що має на меті створити негативне сприйняття дій ОП. Перевірка фактів виявила, що контент містить елементи правди: факт підтримки Кличком мера Атрошенка підтверджений, як і його власні звинувачення на адресу Офісу Президента щодо сприяння обшукам у посадовців мерії Києва. Однак значна частина тверджень, зокрема прямий наказ Офісу Президента силовикам, конкретні дати обшуків, точні мотиви ОП та деталі внутрішніх рішень, є непідтвердженими або є спекулятивними інтерпретаціями, поданими як беззаперечні факти. Таким чином, контент змішує підтверджені події зі значною кількістю неперевірених припущень та інтерпретацій, щоб сформувати наратив політичного тиску та переслідування, що робить його недостовірним та маніпулятивним.",
        "disinfo": [
            "Офис Президента дал силовикам приказ для начала обыска у Кличка и его окружения: Це твердження, представлене як факт, не має прямих незалежних підтверджень. Воно базується на звинуваченнях Кличка щодо тиску, але не підтверджує прямий наказ ОП силовикам.",
            "С ближайшего понедельника маски-шоу планируют посетить мэра Киева Виталия Кличко: Конкретна дата початку обшуків не підтверджена жодними джерелами.",
            "Причина: Кличко своими международными связями с ЕС и США, а также влиянием в Киеве раздражает Офис Президента. Кроме того, позиции ОП в столице не усиливаются – глава городской военной администрации Попко до сих пор почти не влияет на процессы в столице: Зазначені мотиви Офісу Президента (роздратування міжнародними зв'язками Кличка, неефективність Попка) є спекулятивними інтерпретаціями та не підтверджені наданими джерелами. Також немає інформації щодо ефективності Попка.",
            "Последней каплей для ОП стало то, что Кличко поднял три десятка мэров: Хоча факт приїзду мерів на підтримку Атрошенка підтверджений, точна кількість 'три десятка' не підтверджена, а твердження про те, що це була 'остання крапля' для ОП, є інтерпретацією, а не фактом.",
            "Именно поэтому на совещании в Офисе Президента решили передать «привет» Кличко. Была дана команда на Кличко для обысков, а также задача пройтись с обысками по домам его окружения: Внутрішні рішення Офісу Президента, деталі 'передачі привіту' та 'дана команда' не підтверджені незалежними джерелами. Це інтерпретація, що подається як відомий факт."
        ]
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Analyze content for manipulation techniques and disinformation
    """
    logger = get_logger()
    content_id = str(uuid.uuid4())
    
    try:
        logger.log_step(
            step_name="api_request",
            content_id=content_id,
            input_data={"content_length": len(request.content)}
        )
        
        result = analyze_content(request.content, content_id=content_id)
        
        logger.log_step(
            step_name="api_response",
            content_id=content_id,
            output_data={
                "manipulation": result.get("manipulation", False),
                "techniques_count": len(result.get("techniques", []))
            }
        )
        
        return {
            "manipulation": result.get("manipulation", False),
            "techniques": result.get("techniques", []),
            "explanation": result.get("explanation", ""),
            "disinfo": result.get("disinfo", [])
        }
    except Exception as e:
        logger.log_step(
            step_name="api_error",
            content_id=content_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)