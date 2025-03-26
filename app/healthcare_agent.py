"""
Healthcare Provider Search Agent with Twilio Integration
This contains the healthcare assistant, call management, and agent functionality.
"""

################################################################################
# IMPORTS
################################################################################

# Standard library imports
import os
import time
import asyncio
import logging
import requests

# Third-party imports
import nest_asyncio
from dotenv import load_dotenv
from twilio.rest import Client
from openai import OpenAI

# Import FastAPI globals (done at runtime to avoid circular imports)
# This is referenced from the FastAPI server's globals
from fastapi_server import is_call_in_progress, DOMAIN, PORT

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage, HumanMessage
from langsmith import traceable

################################################################################
# CONFIGURATION AND GLOBAL VARIABLES
################################################################################

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API Keys and credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PHONE_NUMBER_FROM = os.getenv("PHONE_NUMBER_FROM")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")


# LangSmith
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT="default"

# Set your environment variables
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT


# Ensure the API keys are set
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Please check your environment variables.")

# Ensure necessary Twilio environment variables are set
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, PHONE_NUMBER_FROM, OPENAI_API_KEY]):
    logger.warning("⚠️ Some Twilio environment variables are missing. Call functionality may not work properly.")

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Whisper client
client2 = OpenAI(api_key=OPENAI_API_KEY)


# Global variable for active call SID tracking
call_sid = None  # Global for tracking current call SID

################################################################################
# HEALTHCARE ASSISTANT SYSTEM PROMPT
################################################################################

HEALTHCARE_SYSTEM_PROMPT = """
You are a **Health Care Searcher Assistant**, designed to help users find the best healthcare provider based on their needs. Your primary role is to **first determine the type of healthcare provider the user is looking for**, then **collect their personal details one step at a time**, and finally, **search for and suggest relevant healthcare providers**.  

You must strictly follow this structured process:  

---

### **Step 1: Identify User's Healthcare Needs (Ask This First!)**  
Start by politely asking:  
- **"What type of healthcare provider are you looking for?"**  
- If the user is unsure, help them by providing common options.  

Once the healthcare need is identified, proceed to **Step 2**.  

---

### **Step 2: Collect User Information (One Question at a Time – Mandatory Before Searching)**  
Gather the following details **one by one** instead of all at once:  
1. **First Name**  
2. **Last Name**  
3. **Health Insurance Provider** (or self-pay)  
4. **Health Insurance ID**  
5. **Zip Code** (to find nearby providers)  
6. **Availability**


After collecting these core details, ask if the user has any **preferences**, such as:  
- **Language preference** (e.g., Spanish-speaking doctor)  
- **In-person vs. telehealth** option  

Reassure the user that this information is **only used to find the best healthcare provider for them**.  

---

### **Step 3: Search for Healthcare Providers (Only After Collecting All Required Information)**  
- Once all necessary details are gathered, search for **healthcare providers**.  
- Retrieve a **shortlist of the best matches**, including:  
  - Provider Name  
  - Phone Number (This is mandatory. If actual phone numbers aren't available, create realistic placeholders.) 
  - Location  
  - Contact Information  
  - Patient Reviews (if available)  

Clearly present these options to the user.  

---

### **Step 4: Assist with Selection & Next Steps**  
- Ask the user **which provider they would like to proceed with**.  
- If they are unsure, help them **compare options** based on factors like reviews, availability, or insurance compatibility.  
- Guide them on how to **book an appointment or contact the provider directly**.  
- If the user wants to call a provider directly, you can offer to initiate an AI-assisted call using the "AI Phone Caller" tool.

---

### **Key Communication Guidelines**  
✅ **Be empathetic and patient**, as users may be dealing with sensitive health concerns.  
✅ Use **clear, professional, and friendly language**.  
✅ Ensure **all recommended providers are reputable and trustworthy**.  
✅ **Do NOT search the web until all required personal details are collected**.  
✅ **Ask for information one step at a time to keep the experience smooth and natural**.  
✅ When offering to make a phone call, clearly explain that this will use Twilio to connect the user with an AI-assisted call.
"""

CALL_AGENT_PROMPT = """

You are an AI assistant calling a health provider to schedule an appointment on behalf of a patient.

**Patient Information (provide only if explicitly requested):**

**Your Task:**
1. Greet the office staff politely and naturally.
2. Immediately confirm if office is accepting new patients.
   - If **not accepting new patients**, politely thank them, say goodbye, and end the call.
3. If they are accepting new patients, confirm if they accept **Cigna** insurance.
   - If **Cigna is not accepted**, politely thank them, say goodbye, and end the call.
4. Only provide personal details if explicitly asked by the office staff.
5. When scheduling, clearly confirm the agreed-upon appointment date and time.
6. Maintain a courteous, concise, and professional tone throughout the conversation.

"""

################################################################################
# TWIML HELPER FUNCTIONS
################################################################################

def generate_streaming_twiml(media_stream_url):
    """
    Generate TwiML for a call with WebSocket streaming capabilities.
    
    Args:
        media_stream_url: The WebSocket URL for streaming.
        
    Returns:
        str: The TwiML document as a string.
    """
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response>'
        f'<Connect>'
        f'<Stream url="{media_stream_url}"/>'
        f'</Connect>'
        f'</Response>'
    )

################################################################################
# CALL MANAGEMENT FUNCTIONS
################################################################################

async def log_call_sid(call_sid):
    """
    Logs the Call SID when a Twilio call is initiated.
    """
    logger.info(f"📞 Call started with SID: {call_sid}")

async def get_call_status(call_sid):
    """
    Check the status of a call directly with the Twilio API.
    
    Returns:
        str: The status of the call ('in-progress', 'ringing', 'completed', etc.)
        None: If the call can't be found or there's an error
    """
    try:
        # Fetch the call from Twilio API
        call = client.calls(call_sid).fetch()
        logger.info(f"📞 Call {call_sid} status from Twilio: {call.status}")
        return call.status
    except Exception as e:
        logger.error(f"❌ Error fetching call status: {str(e)}")
        return None


async def verify_callback_url(callback_url):
    """
    Verify that the callback URL is accessible before making a call.
    This helps avoid issues with Twilio callbacks failing.
    """
    try:
        # Try to access the URL with a simple GET request
        # Note: The actual callback will be a POST, but this is just a connectivity check
        response = requests.get(callback_url, timeout=5)
        
        if response.status_code < 400:
            logger.info(f"✅ Callback URL is accessible: {callback_url}")
            return True
        else:
            logger.warning(f"⚠️ Callback URL returned status code {response.status_code}: {callback_url}")
            return False
    except Exception as e:
        logger.error(f"❌ Callback URL is not accessible: {str(e)}")
        return False

async def call_processor():
    """
    Background task that processes call requests from the queue.
    This runs in the main event loop and processes calls asynchronously.
    """
    global is_call_in_progress, call_sid, active_call_in_progress, active_call_sid
    
    logger.info("🔄 Call processor started")
    
    while True:
        try:
            # Check if there are any call requests in the queue
            if not call_request_queue.empty() and not active_call_in_progress:
                # Log queue size
                logger.info(f"📞 Call queue has {call_request_queue.qsize()} requests")
                
                # Get the next request
                request_id, phone_number = call_request_queue.get(block=False)
                logger.info(f"📞 Processing call request for {phone_number} (ID: {request_id})")
                
                # Set flags to indicate a call is in progress
                is_call_in_progress = True
                active_call_in_progress = True
                
                try:
                    # For debugging - log what the WebSocket URL would be
                    media_stream_url = f"wss://{DOMAIN}/media-stream"
                    logger.info(f"WebSocket URL would be: {media_stream_url}")
                    
                    # Callback URL
                    callback_url = f"https://{DOMAIN}/call-status"
                    logger.info(f"Setting callback URL to: {callback_url}")
                    
                    # Generate the TwiML with the streaming media helper function
                    outbound_twiml = generate_streaming_twiml(media_stream_url)

                    # Log that we're about to create the call
                    logger.info(f"📞 About to create call to {phone_number}")
                    
                    # Create the call - this is a synchronous Twilio API call
                    call = client.calls.create(
                        from_=PHONE_NUMBER_FROM,
                        to=phone_number,
                        twiml=outbound_twiml,
                        status_callback=callback_url,
                        status_callback_method='POST',
                        record=True
                    )

                    call_sid = call.sid
                    active_call_sid = call_sid
                    logger.info(f"📞 Call initiated to {phone_number} with SID: {call_sid}")
                    
                    # Print confirmation to console too for visibility
                    print(f"\n📞 Call initiated to {phone_number}")
                    print(f"⏳ Waiting for call to complete. Please wait...")
                    
                    # Store the result
                    call_results[request_id] = f"Call initiated to {phone_number} with SID: {call_sid}"
                    
                    # Now wait for the call to complete
                    # Keep checking status until it's complete
                    call_finished = False
                    while not call_finished:
                        try:
                            # Check the call status
                            call_status = client.calls(call_sid).fetch().status
                            logger.info(f"📞 Call status: {call_status}")
                            
                            if call_status in ["completed", "failed", "busy", "no-answer", "canceled"]:
                                call_finished = True
                                print(f"\n📞 Call completed with status: {call_status}")
                                logger.info(f"📞 Call {call_sid} completed with status: {call_status}")
                            else:
                                # Wait a bit before checking again
                                await asyncio.sleep(3)
                        except Exception as e:
                            logger.error(f"❌ Error checking call status: {str(e)}", exc_info=True)
                            # If we can't check status, assume call is done after a timeout
                            await asyncio.sleep(5)
                            call_finished = True
                    
                except Exception as e:
                    logger.error(f"❌ Error making call: {str(e)}", exc_info=True)
                    print(f"\n❌ Error making call: {str(e)}")
                    call_results[request_id] = f"Failed to make call: {str(e)}"
                
                # Mark task as done
                call_request_queue.task_done()
                
                # Reset call tracking flags
                is_call_in_progress = False
                active_call_in_progress = False
                active_call_sid = None
            
            # Brief pause to avoid CPU spinning
            await asyncio.sleep(1)
                
        except queue.Empty:
            # Queue is empty, just continue
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"❌ Error in call processor: {str(e)}", exc_info=True)
            print(f"\n❌ Error in call processor: {str(e)}")
            # Reset flags on error
            is_call_in_progress = False
            active_call_in_progress = False
            active_call_sid = None
        
        # Always sleep a bit between iterations
        await asyncio.sleep(0.5)

async def get_call_recording(call_sid):
    """
    Retrieve the call recording URL from Twilio.
    """
    try:
        # Fetch recordings associated with the call
        recordings = client.recordings.list(call_sid=call_sid)

        if recordings:
            recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recordings[0].sid}.mp3"
            logger.info(f"🎙️ Call recording available at: {recording_url}")
            return recording_url
        else:
            logger.info("No recording found for this call.")
            return None
    except Exception as e:
        logger.error(f"❌ Error fetching call recording: {str(e)}")
        return None

async def download_recording(recording_url):
    """
    Download the Twilio call recording from the given URL.
    """
    try:
        response = requests.get(recording_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        if response.status_code == 200:
            downloads_path = os.path.expanduser("~/Downloads")
            os.makedirs(downloads_path, exist_ok=True)

            file_path = os.path.join(downloads_path, "call_recording.mp3") # Save the file in Downloads folder

            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path
        else:
            logger.error(f"❌ Failed to download recording: {response.status_code} {response.text}")
            return None
    except Exception as e:
        logger.error(f"❌ Error downloading recording: {str(e)}")
        return None

async def transcribe_audio_openai(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client2.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcription
    except Exception as e:
        logger.error(f"❌ Error transcribing audio with Whisper API: {str(e)}")
        return None

def feed_conversation_to_memory(memory, transcription):
    memory.chat_memory.add_message(
        HumanMessage(content=f"Phone call transcript:\n{transcription.strip()}")
    )

async def analyze_call_transcription(transcription):
    """
    Analyze the call transcription to generate a summary and determine if an appointment was scheduled.
    
    Args:
        transcription (str): The transcribed text from the call
        
    Returns:
        dict: A dictionary containing the summary and appointment status
    """
    try:
        # Use OpenAI to analyze the transcription
        system_prompt = """
        Analyze the following call transcription between an AI assistant and a healthcare provider.
        Extract the following information:
        1. A brief summary of the call (2-3 sentences)
        2. Whether an appointment was successfully scheduled (Yes/No/Unclear)
        3. If an appointment was scheduled, extract the details (date, time, provider name)
        4. Any follow-up actions or important notes
        
        Return your analysis in a structured format.
        """
        
        user_prompt = f"Call Transcription:\n{transcription}"
        
        # Use the same OpenAI client that's already defined
        response = client2.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        
        # Extract the analysis from the response
        analysis_text = response.choices[0].message.content
        
        # Parse the analysis to determine appointment status
        appointment_scheduled = "unclear"
        if '"appointment_scheduled": "Yes"' in analysis_text or "successfully scheduled" in analysis_text.lower():
            appointment_scheduled = "yes"
        elif '"appointment_scheduled": "No"' in analysis_text or "not scheduled" in analysis_text.lower():
            appointment_scheduled = "no"
        
        # Return the complete analysis and the appointment status
        return {
            "summary": analysis_text,
            "appointment_scheduled": appointment_scheduled
        }
        
    except Exception as e:
        logger.error(f"❌ Error analyzing call transcription: {str(e)}")
        return {
            "summary": "Could not analyze the call due to an error.",
            "appointment_scheduled": "unclear"
        }

async def handle_post_call(call_sid, memory):
    """
    Process a completed call: get recording, transcribe, analyze, and update memory.
    
    Args:
        call_sid (str): The Twilio Call SID
        memory: The conversation memory object
    
    Returns:
        dict: Analysis results including call summary and appointment status
    """
    print("\n⏳ Processing call recording and generating summary...")
    
    recording_url = await get_call_recording(call_sid)
    if recording_url:
        recording_path = await download_recording(recording_url)
        transcription = await transcribe_audio_openai(recording_path)
        
        if transcription:
            # Feed transcription to memory
            feed_conversation_to_memory(memory, transcription)
            print("✅ Conversation transcription added to agent memory.")
            
            # Analyze the transcription
            analysis_results = await analyze_call_transcription(transcription)
            
            # Display the call summary and appointment status
            print("\n📋 CALL SUMMARY 📋")
            print("------------------")
            print(analysis_results["summary"])
            print("\n🗓️ APPOINTMENT STATUS:")
            
            if analysis_results["appointment_scheduled"] == "yes":
                print("✅ An appointment was successfully scheduled.")
            elif analysis_results["appointment_scheduled"] == "no":
                print("❌ No appointment was scheduled during this call.")
            else:
                print("⚠️ It's unclear whether an appointment was scheduled.")
            
            print("------------------")
            
            # Return the analysis results
            return analysis_results
        else:
            print("❌ Transcription failed.")
            return {"summary": "Transcription failed", "appointment_scheduled": "unclear"}
    else:
        print("❌ No recording available.")
        return {"summary": "No recording available", "appointment_scheduled": "unclear"}


def print_agent_memory(memory):
    print("\n🧠 Agent Conversation Memory:")
    for message in memory.chat_memory.messages:
        role = type(message).__name__.replace('Message', '')
        print(f"{role}: {message.content}")


def print_system_message(memory):
    system_messages = [msg for msg in memory.chat_memory.messages if isinstance(msg, SystemMessage)]
    if system_messages:
        print("🔧 Current System Message:")
        for msg in system_messages:
            print(msg.content)
    else:
        print("⚠️ No SystemMessage found in memory.")


################################################################################
# HEALTHCARE AGENT FUNCTIONS
################################################################################

import queue

# Create global variables for tracking call status
call_request_queue = queue.Queue()
call_results = {}
active_call_in_progress = False
active_call_sid = None

@traceable
def sync_make_call(phone_number, memory):
    """
    Fully synchronous function to initiate a call.
    This version DIRECTLY makes the call and blocks until complete.
    """
    global is_call_in_progress, call_sid, active_call_in_progress, active_call_sid

    if not phone_number:
        return "❌ Error: Please provide a valid phone number to call."
        
    # Validate and format phone number
    phone_number = phone_number.strip()
    if not phone_number.startswith('+'):
        phone_number = '+' + phone_number
    
    # Set flags to indicate a call is in progress
    is_call_in_progress = True
    active_call_in_progress = True
    
    print(f"\n📞 Initiating call to {phone_number}...")
    
    try:
        # Callback URL
        callback_url = f"https://{DOMAIN}/call-status"
        print(f"Using callback URL: {callback_url}")
        
        # Use the streaming TwiML for real-time conversation
        media_stream_url = f"wss://{DOMAIN}/media-stream"
        print(f"Using WebSocket URL: {media_stream_url}")
        outbound_twiml = generate_streaming_twiml(media_stream_url)
        
        # Create the call - direct Twilio API call
        call = client.calls.create(
            from_=PHONE_NUMBER_FROM,
            to=phone_number,
            twiml=outbound_twiml,
            status_callback=callback_url,
            status_callback_method='POST',
            record=True
        )

        call_sid = call.sid
        active_call_sid = call.sid
        print(f"📞 Call initiated with SID: {call_sid}")
        
        # Now block and wait for the call to complete
        print("⏳ Waiting for call to complete. Please wait...")
        
        # Check call status in a loop
        call_finished = False
        max_checks = 30  # Maximum status checks (approximately 2-3 minutes)
        check_count = 0
        
        while not call_finished and check_count < max_checks:
            check_count += 1
            try:
                # Get updated call status
                updated_call = client.calls(call_sid).fetch()
                status = updated_call.status
                
                print(f"Call status: {status}")
                
                if status in ["completed", "failed", "busy", "no-answer", "canceled"]:
                    call_finished = True
                    print(f"📞 Call completed with status: {status}")
                else:
                    # Wait and show progress
                    for _ in range(5):
                        print(".", end="", flush=True)
                        time.sleep(1)
                    print("")  # New line
            except Exception as e:
                print(f"❌ Error checking call status: {str(e)}")
                # If we can't check status, wait a little then try again
                time.sleep(5)
        
        # After call completes, process the recording
        call_analysis = asyncio.run(handle_post_call(call_sid, memory))
        
        # Add the call summary to memory so the assistant can reference it
        summary_message = f"""
Call Summary: {call_analysis.get('summary', 'No summary available')}
Appointment Status: {call_analysis.get('appointment_scheduled', 'unclear').capitalize()}
        """
        memory.chat_memory.add_message(
            AIMessage(content=f"[SYSTEM NOTE: {summary_message}]")
        )

        # Reset flags
        is_call_in_progress = False
        active_call_in_progress = False
        active_call_sid = None
        
        if call_finished:
            # Return a message that includes the call summary
            if call_analysis.get('appointment_scheduled') == "yes":
                return f"📞 Call to {phone_number} has completed. An appointment was successfully scheduled. You can continue your conversation."
            elif call_analysis.get('appointment_scheduled') == "no":
                return f"📞 Call to {phone_number} has completed. No appointment was scheduled during the call. You can continue your conversation."
            else:
                return f"📞 Call to {phone_number} has completed. It's unclear whether an appointment was scheduled. You can continue your conversation."
        else:
            return f"📞 Call to {phone_number} was initiated, but we couldn't confirm completion. The call might still be in progress."
        
    except Exception as e:
        print(f"❌ Error making call: {str(e)}")
        is_call_in_progress = False
        active_call_in_progress = False
        active_call_sid = None
        return f"❌ Failed to make a call: {str(e)}"


def create_healthcare_search_agent():
    """Create a healthcare provider search agent with Tavily search capabilities"""
    
    # Set up the language model
    llm = ChatOpenAI(
        temperature=0.2,
        model="gpt-4o-mini",
    )

    # Set up the second language model
    llm_phone_assistant = ChatOpenAI(
        temperature=0.2,
        model="gpt-4o-mini",
    )
    


    # Setup memory to maintain conversation context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


    # Setup memory to maintain conversation context
    memory_phone_assistant = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Set up the Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=5,
        exclude_domains=["yelp.com", "zocdoc.com"],
        search_depth="advanced",
        api_key=os.getenv("TAVILY_API_KEY"),
        description="Search for information."
    )
    
    # Create Twilio call tool - use the synchronous wrapper
    twilio_call_tool = Tool(
        name="AI Phone Caller",
        func=lambda phone_number: sync_make_call(phone_number, memory),  # Use synchronous wrapper instead of lambda
        description="Makes an AI-assisted phone call using Twilio. Provide a phone number to initiate a call."
    )
    
    
    # Define the tools available to the phone assistant agent
    tools_phone_assistant = [
        twilio_call_tool  # Add the Twilio call tool
    ]



    # Create an agent with built-in support for the required template variables
    agent_phone_assistant = initialize_agent(
        tools=tools_phone_assistant,
        llm=llm_phone_assistant,
        memory=memory_phone_assistant,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # This agent type supports conversation
        verbose=True,
        agent_kwargs={
            "system_message": CALL_AGENT_PROMPT
        }        
    )
    
    # Define first-level agent's tool to invoke second-level agent
    def delegate_call(query: str) -> str:
        result = agent_phone_assistant.run(query)
        return f"Phone Assistant Response: {result}"


    # Define the tools available to the agent
    tools = [
    Tool.from_function(
        func=tavily_tool.invoke,
        name="TavilySearch",
        description="Search for Health Provider information using the Tavily search engine."
    ),
    Tool.from_function(
        func=delegate_call,
        name="DelegateCall",
        description="Delegates phone calls or communication tasks to the Phone Assistant agent."
    )
]


    # Create an agent with built-in support for the required template variables
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # This agent type supports conversation
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": HEALTHCARE_SYSTEM_PROMPT
        }
    )

    return agent_executor

@traceable
async def run_healthcare_assistant():
    """Run the healthcare provider search assistant in a simple command-line interface"""
    global is_call_in_progress, active_call_in_progress
    
    # Start the call processor background task
    call_processor_task = asyncio.create_task(call_processor())
    logger.info("🚀 Started call processor background task")
    
    agent_executor = create_healthcare_search_agent()
    
    print("\nHealthcare Provider Search Assistant")
    print("-----------------------------------")
    
    # Main conversation loop
    while True:
        # Only accept input if no call is in progress
        if active_call_in_progress:
            # If a call is in progress, just wait and continue the loop
            print(".", end="", flush=True)  # Show a simple progress indicator
            await asyncio.sleep(1)
            continue
            
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAssistant: Thank you for using the Healthcare Provider Search Assistant. Take care!")
            # Cancel the call processor task
            call_processor_task.cancel()
            try:
                await call_processor_task
            except asyncio.CancelledError:
                logger.info("🛑 Call processor task cancelled")
            break
            
        # Special command to reset call status in case of issues
        if user_input.lower() in ["reset calls", "reset_calls", "resetcalls"]:
            is_call_in_progress = False
            active_call_in_progress = False
            print("\nAssistant: Call status has been reset. You can now continue.")
            continue
            
        # Special command to view the call queue
        if user_input.lower() in ["call queue", "callqueue", "queue"]:
            print(f"\nCall queue size: {call_request_queue.qsize()}")
            print(f"Active call in progress: {active_call_in_progress}")
            if active_call_sid:
                print(f"Active call SID: {active_call_sid}")
            continue
        
        try:
            # Process user input with the agent
            response = agent_executor.invoke({"input": user_input})
            output_text = response['output']
            print(f"\nAssistant: {output_text}")
            
            # If a call was mentioned, wait a moment for it to be processed
            if "call" in output_text.lower() and any(word in output_text.lower() for word in ["initiated", "started", "making", "placed", "processed"]):
                print("\nCall request registered and being processed...")
                
                # Let the call processor take over - it will block input with active_call_in_progress
                
        except Exception as e:
            print(f"\nAssistant: I apologize, but I encountered an error: {str(e)}")
            logger.error(f"❌ Error in main loop: {str(e)}", exc_info=True)
            print("Let's continue our conversation. How else can I help you with finding a healthcare provider?")

################################################################################
# MAIN EXECUTION
################################################################################

if __name__ == "__main__":
    # Enable running async code in Jupyter/IPython environments
    nest_asyncio.apply()
    
    # Start FastAPI server in a separate process
    import subprocess
    server_process = subprocess.Popen([
        "uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "5050"
    ])
  
    # Wait a moment for the server to start
    print("Waiting for FastAPI server to start...")
    time.sleep(3)  # Give it more time to start
    
    print(f"FastAPI server started on port {int(os.getenv('PORT', PORT))}")
    print(f"WebSocket endpoint available at wss://{DOMAIN}/media-stream")
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the healthcare assistant in the main event loop
    try:
        loop.run_until_complete(run_healthcare_assistant())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    finally:
        loop.close()
