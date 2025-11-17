import arxiv
import json
import os
import logging
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_research_server")

PAPER_DIR = "papers"

# Initialize FastAPI app
app = FastAPI(
    title="Research Server API",
    description="A web API for searching and analyzing research papers from arXiv",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    topic: str
    max_results: int = 5

class SearchResponse(BaseModel):
    success: bool
    paper_ids: List[str]
    topic: str
    message: str

class AnalyzeRequest(BaseModel):
    topic: str

class AnalyzeResponse(BaseModel):
    success: bool
    analysis: str
    topic: str

# Serve the HTML interface
@app.get("/")
async def serve_homepage():
    """Serve the main HTML interface"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {"message": "Research Server API is running", "status": "healthy", "docs_url": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "research-server", "version": "1.0.0"}

def search_papers_core(topic: str, max_results: int = 5) -> List[str]:
    """
    Core function to search for papers on arXiv based on a topic and store their information.
    """
    logger.info(f"Starting paper search for topic: '{topic}' with max_results: {max_results}")
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    papers_info = {}
    paper_ids = []
    
    try:
        for paper in papers:
            paper_id = paper.entry_id.split('/')[-1]
            paper_ids.append(paper_id)
            
            papers_info[paper_id] = {
                "title": paper.title,
                "authors": [str(author) for author in paper.authors],
                "abstract": paper.summary,
                "url": paper.entry_id,
                "published": paper.published.strftime("%Y-%m-%d"),
                "categories": paper.categories
            }
            
            logger.info(f"Found paper: {paper.title}")
        
        # Store papers information in a JSON file
        with open(os.path.join(path, "papers_info.json"), "w") as f:
            json.dump(papers_info, f, indent=2)
        
        logger.info(f"Successfully stored {len(paper_ids)} papers for topic '{topic}'")
        return paper_ids
        
    except Exception as e:
        logger.error(f"Error during paper search: {str(e)}")
        raise e

def extract_info_core(topic: str) -> str:
    """
    Core function to read stored paper information for a topic.
    """
    logger.info(f"Attempting to read papers for topic: '{topic}'")
    
    # Convert topic to directory name
    dir_name = topic.lower().replace(" ", "_")
    path = os.path.join(PAPER_DIR, dir_name, "papers_info.json")
    
    if not os.path.exists(path):
        logger.warning(f"No papers found for topic '{topic}' at path {path}")
        return f"No papers found for topic '{topic}'. Please search for papers first using the search_papers function."
    
    try:
        with open(path, "r") as f:
            papers_info = json.load(f)
        
        logger.info(f"Successfully loaded {len(papers_info)} papers for topic '{topic}'")
        
        # Format the information nicely
        result = f"Research Papers on '{topic}':\n\n"
        
        for paper_id, info in papers_info.items():
            result += f"Paper ID: {paper_id}\n"
            result += f"Title: {info['title']}\n"
            result += f"Authors: {', '.join(info['authors'])}\n"
            result += f"Published: {info['published']}\n"
            result += f"Categories: {', '.join(info['categories'])}\n"
            result += f"Abstract: {info['abstract'][:300]}...\n"
            result += f"URL: {info['url']}\n"
            result += "-" * 80 + "\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error reading papers for topic '{topic}': {str(e)}")
        return f"Error reading papers for topic '{topic}': {str(e)}"

@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Search for papers on arXiv based on a topic and store their information.
    """
    try:
        paper_ids = search_papers_core(request.topic, request.max_results)
        return SearchResponse(
            success=True,
            paper_ids=paper_ids,
            topic=request.topic,
            message=f"Successfully found and stored {len(paper_ids)} papers for topic '{request.topic}'"
        )
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_research(request: AnalyzeRequest):
    """
    Get stored paper information for analysis.
    """
    try:
        analysis = extract_info_core(request.topic)
        return AnalyzeResponse(
            success=True,
            analysis=analysis,
            topic=request.topic
        )
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics")
async def list_topics():
    """
    List all available research topics that have been searched.
    """
    try:
        if not os.path.exists(PAPER_DIR):
            return {"topics": []}
        
        topics = []
        for item in os.listdir(PAPER_DIR):
            topic_path = os.path.join(PAPER_DIR, item)
            if os.path.isdir(topic_path):
                # Convert directory name back to readable topic
                readable_topic = item.replace("_", " ").title()
                topics.append({
                    "topic": readable_topic,
                    "directory": item,
                    "has_papers": os.path.exists(os.path.join(topic_path, "papers_info.json"))
                })
        
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error listing topics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Research Server Web API...")
    
    # Create papers directory if it doesn't exist
    os.makedirs(PAPER_DIR, exist_ok=True)
    
    # Get port from environment variable (Render.com sets this)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Server will run on port {port}")
    logger.info("Available endpoints:")
    logger.info("  GET  / - Web interface (or health check)")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /search - Search for papers")
    logger.info("  POST /analyze - Get paper analysis")
    logger.info("  GET  /topics - List available topics")
    logger.info("  GET  /docs - API documentation")
    
    uvicorn.run(app, host="0.0.0.0", port=port)