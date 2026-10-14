"""Versioned API router assembly."""

from __future__ import annotations

from fastapi import APIRouter

from intelligence_engine.api import entities, evidence, extraction, intelligence

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(evidence.router)
api_router.include_router(extraction.router)
api_router.include_router(entities.router)
api_router.include_router(entities.resolution_router)
api_router.include_router(entities.correction_router)
api_router.include_router(intelligence.router)
api_router.include_router(intelligence.relation_router)
