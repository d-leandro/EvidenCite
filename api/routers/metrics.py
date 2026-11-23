"""Routes for inference time/cost dashboard."""
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.services.metrics import get_runs, make_csv, update_client_time

router = APIRouter(prefix='/api/v1/metrics', tags=['metrics'])


class BrowserTiming(BaseModel):
    run_id: str = Field(min_length=36, max_length=36)
    client_ms: float = Field(ge=0, le=86_400_000)


@router.get('/runs')
def list_runs(limit: int = 500):
    return {'runs': get_runs(limit)}


@router.post('/client')
def browser_timing(request: BrowserTiming):
    from uuid import UUID
    try:
        UUID(request.run_id)
    except ValueError:
        return {'updated': False}
    return {'updated': update_client_time(request.run_id, request.client_ms)}


@router.get('/export.csv')
def download_csv():
    return Response(
        make_csv(), media_type='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename="evidencite_inferencia.csv"'}
    )
