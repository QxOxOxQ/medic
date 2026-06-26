from fastapi import APIRouter, Response, status


router = APIRouter()


@router.get("/healthz", include_in_schema=False, status_code=status.HTTP_204_NO_CONTENT)
def health() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)
