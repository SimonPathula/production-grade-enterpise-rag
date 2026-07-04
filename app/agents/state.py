import operator
from typing import Annotated, TypedDict, List

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str