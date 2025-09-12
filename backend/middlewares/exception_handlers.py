from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from backend.logger import logger


async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)

    except Exception as exc:
        logger.exception("Unhandled exception")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(exc)})
