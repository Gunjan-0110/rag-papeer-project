import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)
from backend.rag_graph import app_graph
from backend.vector_store import search

load_dotenv()

def run_evaluation():
    """Runs automated RAG evaluation metrics using DeepEval and Google Gemini."""
    print("🚀 Initializing Papeer RAG Evaluation Pipeline...")

    # Define test questions and expected ground truths
    test_queries = [
        {
            "input": "What methodology is used for document evaluation?",
            "expected_output": "The system utilizes automated RAG evaluation and retrieval metrics."
        },
        {
            "input": "How does the session isolation work?",
            "expected_output": "Each session uses its own isolated local ChromaDB collection."
        }
    ]

    test_cases = []

    for item in test_queries:
        query = item["input"]
        
        # 1. Retrieve context chunks from local ChromaDB
        retrieved_docs = search(query, session_id="default_session", k=4)
        retrieval_context = [doc.page_content for doc in retrieved_docs] if retrieved_docs else ["Local context fallback."]

        # 2. Invoke our LangGraph pipeline
        initial_state = {"query": query, "documents": [], "generation": "", "route": ""}
        result = app_graph.invoke(initial_state)
        actual_output = result.get("generation", "No response generated.")

        # 3. Create DeepEval test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=item["expected_output"],
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)

    # Define open-source RAG metrics (passing threshold: 0.7)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)

    print("📊 Executing evaluation metrics...")
    
    # Run evaluation
    evaluate(
        test_cases=test_cases,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            contextual_relevancy_metric
        ]
    )
    print("✅ Evaluation complete! Results logged successfully.")

if __name__ == "__main__":
    run_evaluation()