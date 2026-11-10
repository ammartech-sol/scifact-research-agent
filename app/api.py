import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="scifact-research-agent")


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