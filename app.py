from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
from typing import Dict,Any, Optional
import os
from dotenv import load_dotenv

from scraper import LSEScraper
from extractor import LLMExtractor,PostProcessor
from proximity_extractor import ProximityExtractor
from nlp_extractor import NLPExtractor
from hybrid_extractor import HybridExtractor
from schema import Form83Schema
import time

load_dotenv()

app = FastAPI(title="Form 8.3 Extractor API",version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
scraper=LSEScraper()
proximity_extractor = ProximityExtractor()
nlp_extractor=NLPExtractor()
api_key=os.getenv("GEMINI_API_KEY")
llm_extractor=None
hybrid_extractor=None
if api_key:
    llm_extractor=LLMExtractor(api_key=api_key)
    hybrid_extractor=HybridExtractor(api_key=api_key)


class ExtractionRequest(BaseModel):
    news_id: str
    method: str = "hybrid"


class TextExtractionRequest(BaseModel):
    raw_text: str
    method: str = "hybrid"

class ScraperTestRequest(BaseModel):
    news_id: str


def compare_extractions(proximity_data: Dict[str, Any], llm_data: Dict[str, Any]) -> Dict[str, Any]:

    def get_nested_value(data: dict, keys: list) -> str:
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, "")
            else:
                return ""
        return str(data)

    comparison_fields = [
        (["key_information", "discloser_name"], "Discloser Name"),
        (["key_information", "offeror_offeree_name"], "Offeror/Offeree"),
        (["key_information", "transaction_date"], "Transaction Date"),
        (["positions", "interests", "security_class"], "Security Class"),
        (["positions", "interests", "long_positions", "number"], "Long Position Number"),
        (["positions", "interests", "long_positions", "percentage"], "Long Position %"),
        (["other_information", "contact_name"], "Contact Name"),
        (["other_information", "disclosure_date"], "Disclosure Date"),
    ]

    matches = 0
    total_fields = 0
    field_comparison = []

    for field_path, field_name in comparison_fields:
        prox_val = get_nested_value(proximity_data, field_path)
        llm_val = get_nested_value(llm_data, field_path)
        prox_normalized = prox_val.strip().lower()
        llm_normalized = llm_val.strip().lower()

        is_match = prox_normalized == llm_normalized if prox_normalized and llm_normalized else False
        both_empty = not prox_normalized and not llm_normalized

        if prox_normalized or llm_normalized:
            total_fields += 1
            if is_match:
                matches += 1

        field_comparison.append({
            "field": field_name,
            "proximity_value": prox_val,
            "llm_value": llm_val,
            "match": is_match,
            "both_empty": both_empty
        })
    similarity_score = (matches / total_fields * 100) if total_fields > 0 else 0
    prox_dealings = len(proximity_data.get("dealings", {}).get("purchases_sales", []))
    llm_dealings = len(llm_data.get("dealings", {}).get("purchases_sales", []))

    return {
        "overall_similarity": round(similarity_score, 2),
        "matching_fields": matches,
        "total_compared_fields": total_fields,
        "field_by_field": field_comparison,
        "dealings_count": {
            "proximity": prox_dealings,
            "llm": llm_dealings,
            "match": prox_dealings == llm_dealings
        }
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/test-scraper")
async def test_scraper_endpoint(request: ScraperTestRequest):
    raw_data = await scraper.fetch_form_data(request.news_id)
    text_content = scraper.extract_text_content(raw_data)

    return JSONResponse({
        "success": True,
        "news_id": request.news_id,
        "raw_api_response": raw_data,
        "extracted_text": text_content[:2000] + "..." if len(text_content) > 2000 else text_content,
        "text_length": len(text_content)
    })


@app.post("/api/extract")
async def extract_from_news_id(request: ExtractionRequest):

    raw_text = await scraper.scrape_and_extract(request.news_id)

    results = {
        "success": True,
        "news_id": request.news_id,
        "method": request.method,
    }

    if request.method in ["proximity","both"]:
        start_time=time.time()
        proximity_data=proximity_extractor.extract(raw_text)
        proximity_time=time.time()-start_time

        results["proximity_extraction"]={
            "data":proximity_data,
            "time_seconds":round(proximity_time,3),
            "method":"Proximity-based (Rule-based IR)"
        }

    if request.method in ["nlp","both"]:
        start_time = time.time()
        nlp_data=nlp_extractor.extract(raw_text)
        nlp_time = time.time()-start_time

        results["nlp_extraction"]={
            "data": nlp_data,
            "time_seconds": round(nlp_time,3),
            "method": "NLP-based (spaCy + NLTK + TF-IDF)"
        }

    if request.method in ["llm","both"]:
        if not llm_extractor:
            raise HTTPException(
                status_code=400,
                detail="LLM extraction requires GEMINI_API_KEY in .env file")

        start_time=time.time()
        llm_data=await llm_extractor.extract(raw_text,temperature=0.0)
        llm_time=time.time()-start_time

        results["llm_extraction"]={
            "data":llm_data,
            "time_seconds":round(llm_time,3),
            "method":"LLM-based (Gemini 2.5 Flash)",
            "model":"gemini-2.5-flash"
        }

    if request.method=="hybrid":
        if not hybrid_extractor:
            raise HTTPException(
                status_code=400,
                detail="Hybrid extraction requires GEMINI_API_KEY in .env file")

        start_time = time.time()
        hybrid_data=await hybrid_extractor.extract(raw_text,verbose=False)
        hybrid_time=time.time()-start_time

        results["hybrid_extraction"]={
            "data":hybrid_data,
            "time_seconds": round(hybrid_time,3),
            "method":"Hybrid (NLP + LLM)",
            "breakdown":{
                "nlp_fields":["key_information","other_information"],
                "llm_fields":["positions","dealings"]
            }
        }

    if request.method == "both" and "proximity_extraction" in results and "llm_extraction" in results:
        results["comparison"] = compare_extractions(
            results["proximity_extraction"]["data"],
            results["llm_extraction"]["data"]
        )
        results["comparison"]["speed_difference"] = {
            "proximity_time": results["proximity_extraction"]["time_seconds"],
            "llm_time": results["llm_extraction"]["time_seconds"],
            "speedup": round(results["llm_extraction"]["time_seconds"] /
                           results["proximity_extraction"]["time_seconds"], 2)
        }

    results["metadata"] = {
        "raw_text_length": len(raw_text),
    }

    return JSONResponse(results)


@app.post("/api/extract-from-text")
async def extract_from_text(request: TextExtractionRequest):
    """
    Extract from raw text directly (bypass scraper)
    Supports: hybrid (recommended), nlp, llm, or proximity methods
    """
    results = {
        "success": True,
        "method": request.method,
    }

    if request.method in ["proximity", "both"]:
        start_time = time.time()
        proximity_data = proximity_extractor.extract(request.raw_text)
        proximity_time = time.time() - start_time

        results["proximity_extraction"] = {
            "data": proximity_data,
            "time_seconds": round(proximity_time, 3),
            "method": "Proximity-based (Rule-based IR)"
        }

    if request.method in ["nlp", "both"]:
        start_time = time.time()
        nlp_data = nlp_extractor.extract(request.raw_text)
        nlp_time = time.time() - start_time

        results["nlp_extraction"] = {
            "data": nlp_data,
            "time_seconds": round(nlp_time, 3),
            "method": "NLP-based (spaCy + NLTK + TF-IDF)"
        }

    if request.method in ["llm", "both"]:
        if not llm_extractor:
            raise HTTPException(
                status_code=400,
                detail="LLM extraction requires GEMINI_API_KEY in .env file"
            )

        start_time = time.time()
        llm_data = await llm_extractor.extract(request.raw_text, temperature=0.0)
        llm_time = time.time() - start_time

        results["llm_extraction"] = {
            "data": llm_data,
            "time_seconds": round(llm_time, 3),
            "method": "LLM-based (Gemini 2.5 Flash)",
            "model": "gemini-2.5-flash"
        }

    if request.method == "hybrid":
        if not hybrid_extractor:
            raise HTTPException(
                status_code=400,
                detail="Hybrid extraction requires GEMINI_API_KEY in .env file"
            )

        start_time = time.time()
        hybrid_data = await hybrid_extractor.extract(request.raw_text, verbose=False)
        hybrid_time = time.time() - start_time

        results["hybrid_extraction"] = {
            "data": hybrid_data,
            "time_seconds": round(hybrid_time, 3),
            "method": "Hybrid (NLP + LLM)",
            "breakdown": {
                "nlp_fields": ["key_information", "other_information"],
                "llm_fields": ["positions", "dealings"]
            }
        }

    if request.method == "both" and "proximity_extraction" in results and "llm_extraction" in results:
        results["comparison"] = compare_extractions(
            results["proximity_extraction"]["data"],
            results["llm_extraction"]["data"]
        )
        results["comparison"]["speed_difference"] = {
            "proximity_time": results["proximity_extraction"]["time_seconds"],
            "llm_time": results["llm_extraction"]["time_seconds"],
            "speedup": round(results["llm_extraction"]["time_seconds"] /
                           results["proximity_extraction"]["time_seconds"], 2)
        }

    results["metadata"] = {
        "raw_text_length": len(request.raw_text),
    }

    return JSONResponse(results)


@app.get("/api/schema")
async def get_schema():
    return JSONResponse(Form83Schema.model_json_schema())


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
