import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()
    # Imported here, not at the top, so tests (which never run start-up) do not need torch.
    from app.embedder import load_embedder

    load_embedder()
    logger.info("embedding model ready in %.1f s (includes importing torch)", time.perf_counter() - started)
    yield


app = FastAPI(title="scifact-research-agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    cited_doc_ids: list[str]
    steps_used: int
    trace_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    return AskResponse(
        answer="stub answer: the agent is not built yet",
        cited_doc_ids=[],
        steps_used=0,
        trace_id=str(uuid.uuid4()),
    )