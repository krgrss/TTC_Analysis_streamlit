"""
FastAPI application — TTC Analysis API

Purpose:
- Provide API routes for the frontend (TanStack Start app) including network coverage endpoints.

Security:
- Minimal app configuration; consider adding CORS in dev only if required by the frontend.
"""
from fastapi import FastAPI
from .routes import network, qa


def create_app() -> FastAPI:
    app = FastAPI(title="TTC Analysis API")
    # Prefix all routes with /api to match the frontend dev proxy expectation
    app.include_router(network.router, prefix="/api")
    app.include_router(qa.router, prefix="/api")
    return app


app = create_app()
