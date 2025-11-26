from fastapi import APIRouter
from app.api import search, analysis, panels

api_router = APIRouter()

api_router.include_router(search.router, tags=["search"])
api_router.include_router(analysis.router, tags=["analysis"])
api_router.include_router(panels.router, prefix="/panels", tags=["panels"])