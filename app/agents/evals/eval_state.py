from typing import TypedDict, List

class EvaluationState(TypedDict):
    current_query: str
    documents: List[str]
    final_answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int 