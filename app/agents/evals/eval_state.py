# from typing import TypedDict, List

# class EvaluationState(TypedDict):
#     current_query: str
#     documents: List[str]
#     final_answer: str
#     prompt_tokens: int
#     completion_tokens: int
#     total_tokens: int 

from typing import TypedDict, List

class EvaluationState(TypedDict):
    current_query: str
    documents: List[str]
    raw_documents: List[str] 
    final_answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retrieval_time_ms: float       
    rerank_time_ms: float          
    generation_time_ms: float     