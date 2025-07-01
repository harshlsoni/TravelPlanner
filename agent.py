from typing import TypedDict, Optional, List, Dict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.checkpoint.memory import InMemorySaver
import dateparser
import os
from datetime import datetime
import json 
from node_functions import get_data,normalize_date,reasoning,satisfaction_checker_node,get_exact_train_data_node,should_continue
from langchain_huggingface import HuggingFaceEndpoint


class State(TypedDict):
    input: str # Original user input
    source: Optional[str]
    destination: Optional[str]
    date: Optional[str]
    data: Optional[List[Dict]] # Raw scraped train data
    filtered_train_options: Optional[List[Dict]] # Best options from reasoning node (structured)
    user_original_needs: Optional[str] # Stored original user request for context
    iteration_count: int # Tracks iterations for satisfaction checker
    final_message: Optional[str] # Human-readable message to display at the end
    reasoning_guidance: Optional[Dict] # Guidance from satisfaction_checker for reasoning node
    decision: str # Explicitly added to State to ensure conditional edge can read it

# Initialize LLM
from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("groq"),
    temperature=0.1
)


# Define the graph
graph = StateGraph(State)

# Memory for checkpointing (optional, but good for long-running agents)
checkpoint = InMemorySaver()

# Add nodes to the graph
graph.add_node("Extractor", get_data)
graph.add_edge(START, "Extractor")

graph.add_node("Date_Corrector", normalize_date)
graph.add_edge("Extractor", "Date_Corrector")

graph.add_node("Scraper", get_exact_train_data_node)
graph.add_edge("Date_Corrector", "Scraper")

graph.add_node("Reason", reasoning)
graph.add_edge("Scraper", "Reason")

graph.add_node("Satisfaction_Checker", satisfaction_checker_node)
graph.add_edge("Reason", "Satisfaction_Checker")

# Define a conditional edge from Satisfaction_Checker
# This function will determine the next step based on the output of satisfaction_checker_node


graph.add_conditional_edges(
    "Satisfaction_Checker",
    should_continue, # This function decides the path
    {
        "re_run": "Reason", # Loop back to the Reason node if re-run is needed
        "end_process": END # End the graph execution
    }
)

# Compile the graph
app = graph.compile(checkpointer=checkpoint) # Renamed 'output' to 'app' for clarity

# Example Usage
config = {"configurable": {"thread_id": 1}}

# Test Case 1: Simple request, likely satisfied
print("\n--- Running Test Case 1 ---")
input1 = "Lets plan a trip from ajmer to mumbai a day after tomorrow."
final_state1 = app.invoke({"input": input1}, config=config)
print("\n--- Final State for Test Case 1 ---")
print(f"Final Message: {final_state1.get('final_message', 'No final message.')}")
if final_state1.get('filtered_train_options'):
    print("Best Train Options:")
    print(json.dumps(final_state1['filtered_train_options'], indent=2))
else:
    print("No best train options found.")

