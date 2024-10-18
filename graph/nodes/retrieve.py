from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
       Retrieve documents based on the question in the given state.

       Args:
           state (GraphState): The current state containing the question.

       Returns:
           Dict[str, Any]: A dictionary containing the retrieved documents and the question.
       """

    print("---RETRIEVE---")

    question = state["question"]

    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}

