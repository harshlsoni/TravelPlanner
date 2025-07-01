from typing import TypedDict, Optional, List, Dict
from scraper import get_exact_train_options # Assuming 'scraper.py' exists and has this function
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.checkpoint.memory import InMemorySaver
import dateparser
from datetime import datetime
import json
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint
load_dotenv()
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

from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("groq"),
    temperature=0.1
)

def get_data(state: State) -> State:
    """
    Extracts source, destination, and date from user input using an LLM.
    Also stores the original user input as `user_original_needs` and
    initializes `iteration_count`.
    """
    print(f"\n--- Node: get_data ---")
    prompt = ChatPromptTemplate.from_template("Extract the following information from the text: \n {input} \n\n {format_instructions}")

    schema = [
        ResponseSchema(name="source", description="name of source city"),
        ResponseSchema(name="destination", description="name of destination city"),
        ResponseSchema(name="date", description="date of journey (can be natural language like 'tomorrow' or 'next week')")
    ]

    parser = StructuredOutputParser(response_schemas=schema)
    format_instructions = parser.get_format_instructions()

    chain = prompt | llm | parser

    user_input = state["input"]

    try:
        result = chain.invoke({"input": user_input, "format_instructions": format_instructions})
        return {
            **state,
            "source": result.get("source"),
            "destination": result.get("destination"),
            "date": result.get("date"),
            "user_original_needs": user_input, # Store the full original input
            "iteration_count": 0, # Initialize iteration count
            "decision": "continue" # Default decision to proceed
        }
    except Exception as e:
        print(f"Error in get_data: {e}")
        return {**state, "final_message": "Could not extract necessary information. Please try again with a clearer request.", "decision": "FINISH"}


def normalize_date(state: State) -> State:
    """
    Normalizes the extracted date string into a 'YYYY-MM-DD' format.
    """
    print(f"\n--- Node: normalize_date ---")
    raw_date = state.get("date")

    if raw_date:
        try:
            parsed_date = dateparser.parse(raw_date, settings={'PREFER_DATES_FROM': 'future'})
            if parsed_date:
                formatted_date = parsed_date.strftime("%Y-%m-%d")
                print(f"Parsed date: {raw_date} -> {formatted_date}")
                return {**state, "date": formatted_date, "decision": "continue"}
            else:
                print(f"Failed to parse natural language date: '{raw_date}'")
                return {**state, "final_message": "I couldn't understand the date you provided. Please try again.", "decision": "FINISH"}
        except Exception as e:
            print(f"Error parsing date '{raw_date}': {e}")
            return {**state, "final_message": "An error occurred while processing the date. Please ensure it's valid.", "decision": "FINISH"}
    print("No raw date to normalize.")
    return {**state, "final_message": "No date provided. Please specify a date for your journey.", "decision": "FINISH"}


def get_exact_train_data_node(state: State) -> State:
    """
    Calls the external scraper to get train options for the given source, destination, and date.
    """
    print(f"\n--- Node: get_exact_train_data_node ---")
    source = state.get("source")
    dest = state.get("destination")
    date = state.get("date")

    if not all([source, dest, date]):
        print("Missing source, destination, or date for scraping.")
        return {**state, "data": [], "final_message": "Missing information (source, destination, or date) to find trains.", "decision": "FINISH"}

    print(f"Scraping trains from {source} to {dest} on {date}...")
    try:
        train_data = get_exact_train_options(source=source, destination=dest, date=date)
        print(f"Scraped {len(train_data)} train options.")
        return {
            **state,
            "data": train_data,
            "decision": "continue"
        }
    except Exception as e:
        print(f"Error during scraping: {e}")
        with open("data.json" , 'w') as f:
            f.write(train_data)

        return {**state, "data": [], "final_message": "Failed to retrieve train data. The scraping service might be unavailable.", "decision": "FINISH"}


def reasoning(state: State) -> State:
    """
    Analyzes raw train data and user needs to filter and select the best train options.
    It returns structured JSON output.
    """
    print(f"\n--- Node: reasoning ---")
    data = state.get("data", [])
    src = state.get("source", "N/A")
    dest = state.get("destination", "N/A")
    date = state.get("date", "N/A")
    user_original_needs = state.get("user_original_needs", "")
    reasoning_guidance = state.get("reasoning_guidance", {}) # Guidance from satisfaction_checker

    if not data:
        print("No train data was available to reason upon. Setting decision to FINISH.")
        return {
            **state,
            "filtered_train_options": [],
            "final_message": "No train data was available to reason upon.",
            "decision": "FINISH"
        }

    # Define the output schema for the reasoning node to ensure structured output
    reasoning_response_schema = [
        ResponseSchema(
            name="filtered_train_options",
            description="A list of the best train options, adhering to user's needs and priorities. Each item is a dictionary with 'number', 'name', 'departure', 'arrival', 'duration', 'prices', 'availability', 'Date', 'class_types'.",
            type="array",
            items={
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "name": {"type": "string"},
                    "departure": {"type": "string"},
                    "arrival": {"type": "string"},
                    "duration": {"type": "number"}, # Assuming duration in minutes for easier comparison
                    "prices": {"type": "object"},
                    "availability": {"type": "object"},
                    "Date": {"type": "string"},
                    "class_types": {"type": "array", "items": {"type": "string"}}
                }
            }
        ),
        ResponseSchema(
            name="summary_message",
            description="A brief, human-readable summary of the best train option(s) found.",
            type="string"
        )
    ]
    reasoning_parser = StructuredOutputParser(response_schemas=reasoning_response_schema)
    reasoning_format_instructions = reasoning_parser.get_format_instructions()

    # Dynamic prompt based on guidance from previous satisfaction_checker run
    guidance_text = ""
    if reasoning_guidance:
        guidance_text = (
            f"\n\n--- REFINEMENT GUIDANCE FROM PREVIOUS ATTEMPT ---\n"
            f"Reason for re-run: {reasoning_guidance.get('reason_for_re_run', 'N/A')}\n"
            f"Please adjust your filtering based on this guidance: {reasoning_guidance.get('guidance_for_reasoning', 'N/A')}\n"
        )


    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an intelligent AI travel planner.
         Your goal is to help users select the **best train travel option** using structured data about available train routes.

         Your decisions must prioritize:
         1. ✅ **Seat availability** – prefer options with full availability ("AVL"), then RAC (half seat), and avoid "WL"/"Regret".
         2. 🛏️ **Comfort class** – prefer "3A" (AC class) over "SL" (Sleeper).
         3. 🕒 **Efficiency** – avoid routes that take **2+ hours longer** than the average travel time for other options.
         4. 🔁 **Fallback** – if no suitable trains are available for the given date, you can suggest considering trains from ±2 days if the user's needs are flexible.
         5. 🛑 If **no valid options** exist at all for the specified date and preferences, return an empty `filtered_train_options` list and a 'No seats available' summary.
         6. ❓ If data is incomplete or missing, return an empty `filtered_train_options` list and an 'I don't have enough information.' summary.

         Output your response in JSON format as specified by the provided `format_instructions`.
         Ensure the 'class_types' field for each train option reflects all available classes for that train.
         When providing duration, convert it to total minutes (e.g., '5h 30m' becomes 330).
         """ + guidance_text
        ),
        ("human",
         """You are given structured train route data and user input.

         ------------------ CONTEXT (Train Options) ------------------
         {data}

         ------------------ USER INPUT ------------------
         - Source city: {src}
         - Destination city: {dest}
         - Date of journey: {date}
         - Original user request: {user_original_needs}

         ------------------ FIELD DESCRIPTIONS ------------------
         Each train is represented as a dictionary with these fields:
         - "number": Unique train number.
         - "name": Name of the train.
         - "departure": Departure time from source.
         - "arrival": Arrival time at destination.
         - "duration": Travel time in hours and minutes (e.g., "5h 30m" should be converted to minutes for comparison, e.g., 330).
         - "prices": Dictionary with ticket price by class (e.g., "3A": ₹850, "SL": ₹300).
         - "availability": Dictionary with seat availability by class (e.g., "3A": "AVL", "SL": "WL 12").
         - "Date": Date of that train's schedule (YYYY-MM-DD).
         - "class_types": A list of strings, e.g., ["3A", "SL"], indicating all classes available on that train.

         NOTE:
         - "AVL" means full seat available ✅
         - "RAC" means half seat ⚠️
         - "WL"/"Regret" means no seat ❌
         - "3A" = AC economy (more comfortable)
         - "SL" = Sleeper class (less comfortable)

         ------------------ TASK ------------------
         Based on the above, select the **best available train option(s)** according to the priorities and user's original request.
         Provide a list of the chosen train options in the `filtered_train_options` field and a `summary_message` for the user.
         If no suitable options are found based on the strict criteria, return an empty `filtered_train_options` list.

         {format_instructions}
         """
        )
    ])

    chain = prompt | llm | reasoning_parser # Apply the structured parser

    try:
        # Pass raw data as JSON string to the prompt
        llm_response = chain.invoke({
            "data": json.dumps(data),
            "src": src,
            "dest": dest,
            "date": date,
            "user_original_needs": user_original_needs,
            "format_instructions": reasoning_format_instructions
        })

        # The parser should return a dict with 'filtered_train_options' and 'summary_message'
        filtered_options = llm_response.get("filtered_train_options", [])
        summary_message = llm_response.get("summary_message", "I could not find any suitable train options based on your request.")

        # Ensure 'class_types' is present in each option for the satisfaction_checker
        # (Though LLM should ideally populate this based on prompt)
        for option in filtered_options:
            if 'class_types' not in option and 'availability' in option:
                option['class_types'] = list(option['availability'].keys())
            # Convert duration to minutes for easier comparison in satisfaction_checker
            # This is also explicitly asked from the LLM in the prompt
            if isinstance(option.get('duration'), str):
                try:
                    parts = option['duration'].split(' ')
                    hours = int(parts[0].replace('h', '')) if 'h' in parts[0] else 0
                    minutes = int(parts[1].replace('m', '')) if len(parts) > 1 and 'm' in parts[1] else 0
                    option['duration'] = hours * 60 + minutes
                except:
                    option['duration'] = 0 # Default to 0 if parsing fails


    except Exception as e:
        print(f"Error in reasoning node (LLM invocation or parsing): {e}")
        # In case of parsing error, return empty list and an error message
        filtered_options = []
        summary_message = f"An internal error occurred while finding train options. Error: {e}"

    print(f"Reasoning node output - filtered_options count: {len(filtered_options)}")
    print(f"Reasoning node output - summary_message: {summary_message}")

    return {
        **state,
        "filtered_train_options": filtered_options,
        "final_message": summary_message,
        "reasoning_guidance": {}, # Reset guidance after it's used
        "decision": "continue" # Default decision to allow satisfaction checker to run
    }


def satisfaction_checker_node(state: State) -> Dict:
    """
    Langgraph node: Evaluates filtered train options against user's original needs
    using an LLM to determine if the search is complete or if further iteration is required.

    Args:
        state (dict): The current state of the Langgraph, expected to contain:
            - 'filtered_train_options': A list of train options from the reasoning_node.
            - 'user_original_needs': The initial natural language requirements from the user.
            - 'iteration_count': Current iteration count to prevent infinite loops.

    Returns:
        dict: A dictionary indicating the decision ('FINISH' or 'RE-RUN_REASONING_NODE')
              and relevant messages or data. This dictionary will be merged into the state.
    """

    # Extract inputs from the current state
    filtered_train_options = state.get('filtered_train_options', [])
    user_original_needs = state.get('user_original_needs', "")
    iteration_count = state.get('iteration_count', 0)

    print(f"\n--- Node: Satisfaction Checker ---")
    print(f"Received filtered_train_options (count: {len(filtered_train_options)}): {json.dumps(filtered_train_options, indent=2) if filtered_train_options else '[]'}")
    print(f"User original needs: {user_original_needs}")
    print(f"Current iteration: {iteration_count}")

    # --- Configuration for Re-run Logic ---
    MAX_ITERATIONS = 3 # Prevent infinite loops

    # Define the output schema for the satisfaction checker LLM
    satisfaction_response_schema = [
        ResponseSchema(name="is_satisfied", description="True if the train options completely meet user needs based on comfort and availability, False otherwise.", type="boolean"),
        ResponseSchema(name="reason_for_re_run", description="If not satisfied, explain concisely why further iteration is needed.", type="string"),
        ResponseSchema(name="guidance_for_reasoning", description="Specific, actionable guidance for the reasoning node to improve its next selection (e.g., 'relax comfort criteria', 'prioritize AVL seats', 'consider RAC/WL for specific classes').", type="string")
    ]
    satisfaction_parser = StructuredOutputParser(response_schemas=satisfaction_response_schema)
    satisfaction_format_instructions = satisfaction_parser.get_format_instructions()

    # Determine the final decision and messages
    final_decision = ""
    reason_for_stopping = ""
    guidance_for_reasoning_output = ""
    is_llm_satisfied = False # Track LLM's opinion

    # Always increment iteration count at the beginning of the node
    new_iteration_count = iteration_count + 1

    # Scenario A: No options received from Reasoning node
    if not filtered_train_options:
        if new_iteration_count > MAX_ITERATIONS:
            final_decision = "FINISH"
            reason_for_stopping = f"Reached maximum iterations ({MAX_ITERATIONS}). No train options were found at all after multiple attempts."
            guidance_for_reasoning_output = "Inform the user that no suitable options were found within the given constraints after multiple attempts."
            # Update final message in state for clarity
            state["final_message"] = reason_for_stopping
        else:
            final_decision = "RE-RUN_REASONING_NODE"
            reason_for_stopping = "The reasoning node did not find any suitable train options in the previous step."
            guidance_for_reasoning_output = "Broaden filtering criteria significantly, or consider looking for trains on adjacent dates if the current date has no options. Perhaps relax comfort criteria."
            state["final_message"] = "Could not find suitable trains. Re-evaluating with broader criteria."

        print(f"Decision: {final_decision} - {reason_for_stopping}")
        return {
            "decision": final_decision,
            "reasoning_guidance": {
                "reason_for_re_run": reason_for_stopping,
                "guidance_for_reasoning": guidance_for_reasoning_output
            },
            "iteration_count": new_iteration_count
        }

    # Scenario B: Options were received, now use LLM to check satisfaction
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an intelligent evaluator of train travel options.
         Your task is to review a list of 'best' train options (selected by another AI)
         and determine if they truly satisfy the **original user request**, focusing on:
         Your decisions must prioritize:
         1. ✅ **Seat availability** – prefer options with full availability ("AVL"), then RAC (half seat), and avoid "WL"/"Regret".
         2. 🛏️ **Comfort class** – prefer "3A" (AC class) over "SL" (Sleeper).
         3. 🕒 **Efficiency** – avoid routes that take **2+ hours longer** than the average travel time for other options.
        
         ------------------ FIELD DESCRIPTIONS ------------------
         Each train is represented as a dictionary with these fields:
         - "number": Unique train number.
         - "name": Name of the train.
         - "departure": Departure time from source.
         - "arrival": Arrival time at destination.
         - "duration": Travel time in hours and minutes (e.g., "5h 30m" should be converted to minutes for comparison, e.g., 330).
         - "prices": Dictionary with ticket price by class (e.g., "3A": ₹850, "SL": ₹300).
         - "availability": Dictionary with seat availability by class (e.g., "3A": "AVL", "SL": "WL 12").
         - "Date": Date of that train's schedule (YYYY-MM-DD).
         - "class_types": A list of strings, e.g., ["3A", "SL"], indicating all classes available on that train.

         NOTE:
         - "AVL" means full seat available ✅
         - "RAC" means half seat ⚠️
         - "WL"/"Regret" means no seat ❌
         - "3A" = AC economy (more comfortable)
         - "SL" = Sleeper class (less comfortable)
         
         Based on your evaluation, decide if the current options are satisfactory for the user.
         If not, explain *why* and provide specific, actionable `guidance_for_reasoning` for the 'reasoning' AI node
         to improve its selection in the next attempt.

         Prioritize getting a solution, so if an exact match isn't found, suggest relaxing criteria if necessary.
         For example, if the user asks for 'confirmed AC' but only 'RAC 3A' is available, you might suggest relaxing to 'RAC' or a non-AC class to find confirmed seats.

         Output your response in JSON format as specified by the `format_instructions`.
         """
        ),
        ("human",
         """
         ------------------ USER'S ORIGINAL REQUEST ------------------
         {user_original_needs}

         ------------------ BEST TRAIN OPTIONS FOUND SO FAR ------------------
         {filtered_train_options_json}

         ------------------ GUIDANCE FOR YOUR RESPONSE ------------------
         {format_instructions}
         """
        )
    ])

    chain = prompt | llm | satisfaction_parser

    try:
        llm_evaluation_response = chain.invoke({
            "user_original_needs": user_original_needs,
            "filtered_train_options_json": json.dumps(filtered_train_options, indent=2), # Pass as JSON string
            "format_instructions": satisfaction_format_instructions
        })

        is_llm_satisfied = llm_evaluation_response.get("is_satisfied", False)
        reason_for_stopping = llm_evaluation_response.get("reason_for_re_run", "No specific reason provided by LLM.")
        guidance_for_reasoning_output = llm_evaluation_response.get("guidance_for_reasoning", "No specific guidance by LLM.")

    except Exception as e:
        print(f"Error in satisfaction_checker LLM invocation or parsing: {e}")
        # Default to re-run if LLM fails, with a generic error message
        is_llm_satisfied = False
        reason_for_stopping = f"Internal AI evaluation failed: {e}. Attempting re-run with general guidance."
        guidance_for_reasoning_output = "There was an error during AI evaluation. Please try to re-evaluate train options. Be flexible with criteria."

    # Final determination after LLM evaluation and iteration check
    if is_llm_satisfied:
        final_decision = "FINISH"
        # The final_message is already set by the reasoning node's summary_message if satisfied
        print(f"Decision: {final_decision} - LLM found options satisfactory.")
    else:
        # LLM is not satisfied, check iteration count to decide re-run or force finish
        if new_iteration_count > MAX_ITERATIONS:
            final_decision = "FINISH"
            state["final_message"] = f"After multiple attempts, I couldn't find options perfectly matching your needs. Reason: {reason_for_stopping}. Last best attempt: {state.get('final_message', '')}"
            print(f"Decision: {final_decision} (Max Iterations Reached - LLM not satisfied)")
        else:
            final_decision = "RE-RUN_REASONING_NODE"
            state["final_message"] = f"Still refining train options: {reason_for_stopping}"
            print(f"Decision: {final_decision} - {reason_for_stopping}")

    return {
        "decision": final_decision,
        "reasoning_guidance": {
            "reason_for_re_run": reason_for_stopping,
            "guidance_for_reasoning": guidance_for_reasoning_output
        },
        "iteration_count": new_iteration_count
    }

def should_continue(state: State):
    # The `satisfaction_checker_node` now explicitly sets the 'decision' key in the state.
    decision = state.get("decision", "FINISH") # Default to FINISH if somehow not set
    print(f"Conditional edge check - Decision from state: {decision}")
    if decision == "RE-RUN_REASONING_NODE":
        return "re_run"
    return "end_process"