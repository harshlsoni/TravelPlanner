import logging
import requests
from typing import Dict, Any, Optional , TypedDict, List
from flask import Flask, request, jsonify
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.checkpoint.memory import InMemorySaver
import dateparser
import os
from datetime import datetime
from langchain_groq import ChatGroq
import json
from dotenv import load_dotenv
from node_functions import get_data,normalize_date,reasoning,satisfaction_checker_node,get_exact_train_data_node,should_continue
from langchain_huggingface import HuggingFaceEndpoint
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class AIAgent:
    """Simple AI agent using OpenAI GPT for customer service"""
    
    def __init__(self):
        self.api_key = os.getenv('groq')
        if not self.api_key:
            logger.warning("GROQ API key not found. Using mock responses.")
        else:
            api_key = self.api_key
    
    def generate_response(self, message: str, user_id: str) -> str:
        """Generate AI response to user message"""
        try:
            if not self.api_key:
                return self._mock_response(message)
            
            # System prompt for customer service agent
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=self.api_key,
                temperature=0.1
            )

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

            graph.add_conditional_edges(
                "Satisfaction_Checker",
                should_continue, # This function decides the path
                {
                    "re_run": "Reason", # Loop back to the Reason node if re-run is needed
                    "end_process": END # End the graph execution
                }
            )

            output = graph.compile(checkpointer=checkpoint) # Renamed 'output' to 'app' for clarity

            # Example Usage
            config = {"configurable": {"thread_id": 1}}

            final_state = output.invoke({"input": message}, config=config)
            
            return f"Final Message: {final_state.get('final_message', 'No final message.')}"
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Please try again later."
    
    def _mock_response(self, message: str) -> str:
        """Mock responses for testing without OpenAI API"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        elif any(word in message_lower for word in ['order', 'status', 'tracking']):
            return "I can help you check your order status. Please provide your order number."
        elif any(word in message_lower for word in ['product', 'item', 'buy']):
            return "I'd be happy to help you with product information. What are you looking for?"
        elif any(word in message_lower for word in ['support', 'help', 'problem']):
            return "I'm here to help! Can you describe the issue you're experiencing?"
        elif any(word in message_lower for word in ['thank', 'thanks']):
            return "You're welcome! Is there anything else I can help you with?"
        else:
            return "I understand you're asking about: " + message + ". Let me help you with that. Could you provide more details?"

class WhatsAppConnector:
    """Handle WhatsApp integration using WAHA"""
    
    def __init__(self):
        self.waha_url = os.getenv('WAHA_URL', 'http://localhost:3000')
        self.session_name = os.getenv('WAHA_SESSION', 'default')
        self.webhook_url = os.getenv('WEBHOOK_URL', '')
        
    def send_message(self, phone_number: str, message: str) -> bool:
        """Send message via WAHA"""
        try:
            url = f"{self.waha_url}/api/sendText"
            payload = {
                "session": self.session_name,
                "chatId": f"{phone_number}@c.us",
                "text": message
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Message sent successfully to {phone_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {phone_number}: {e}")
            return False
    
    def setup_webhook(self) -> bool:
        """Setup webhook for receiving messages"""
        try:
            if not self.webhook_url:
                logger.warning("Webhook URL not configured")
                return False
                
            url = f"{self.waha_url}/api/{self.session_name}/config"
            payload = {
                "webhooks": [
                    {
                        "url": self.webhook_url,
                        "events": ["message"]
                    }
                ]
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Webhook configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup webhook: {e}")
            return False

# Initialize components
ai_agent = AIAgent()
whatsapp = WhatsAppConnector()

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "WhatsApp AI Agent",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # Log received data for debugging
        logger.info(f"Received webhook data: {json.dumps(data, indent=2)}")
        
        # Extract message information
        event_type = data.get('event')
        if event_type != 'message':
            return jsonify({"status": "ignored", "reason": "not a message event"})
        
        payload = data.get('payload', {})
        message_data = payload.get('body', '')
        from_number = payload.get('from', '').replace('@c.us', '')
        message_type = payload.get('type', '')
        
        # Only process text messages
        if message_type != 'chat':
            return jsonify({"status": "ignored", "reason": "not a text message"})
        
        if not message_data or not from_number:
            return jsonify({"error": "Missing message data or sender"}), 400
        
        # Generate AI response
        ai_response = ai_agent.generate_response(message_data, from_number)
        
        # Send response back to WhatsApp
        success = whatsapp.send_message(from_number, ai_response)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Response sent",
                "from": from_number,
                "original_message": message_data,
                "ai_response": ai_response
            })
        else:
            return jsonify({"error": "Failed to send response"}), 500
            
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/send', methods=['POST'])
def send_message():
    """Manual endpoint to send messages (for testing)"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        message = data.get('message')
        
        if not phone_number or not message:
            return jsonify({"error": "Missing phone_number or message"}), 400
        
        success = whatsapp.send_message(phone_number, message)
        
        if success:
            return jsonify({"status": "success", "message": "Message sent"})
        else:
            return jsonify({"error": "Failed to send message"}), 500
            
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/setup', methods=['POST'])
def setup_webhook():
    """Setup webhook endpoint"""
    try:
        success = whatsapp.setup_webhook()
        if success:
            return jsonify({"status": "success", "message": "Webhook configured"})
        else:
            return jsonify({"error": "Failed to setup webhook"}), 500
    except Exception as e:
        logger.error(f"Error setting up webhook: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')