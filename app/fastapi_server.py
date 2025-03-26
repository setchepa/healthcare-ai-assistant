"""
FastAPI server for Healthcare Provider Search with Twilio Integration and OpenAI Realtime API
This handles all FastAPI routes, endpoint definitions, and websocket handling for real-time AI conversations.
"""

################################################################################
# IMPORTS FOR FASTAPI SERVER
################################################################################

# Standard library imports
import os
import re
import json
import time
import logging
import asyncio
import requests
from typing import Dict, Any
from datetime import datetime, timedelta

# Third-party imports
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import websockets

# OpenAI imports
from openai import AsyncOpenAI
import httpx

# Twilio helper library
from twilio.twiml.voice_response import VoiceResponse, Connect

# OpenAI AI Voice settings
VOICE = "alloy"

SYSTEM_MESSAGE = """


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


# Call event types to log
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

################################################################################
# CONFIGURATION AND ENVIRONMENT VARIABLES
################################################################################

# Getting forwarding element
def remove_http_https(url):
    return re.sub(r'^https?://', '', url)

def get_ngrok_url():
    try:
        # Ngrok API to fetch active tunnels
        response = requests.get("http://127.0.0.1:4040/api/tunnels")
        response.raise_for_status()
        data = response.json()

        # Extract forwarding URL (HTTPS preferred)
        for tunnel in data.get("tunnels", []):
            if tunnel["proto"] == "https":
                return tunnel["public_url"]

        # If no HTTPS tunnel is found, fallback to HTTP
        for tunnel in data.get("tunnels", []):
            if tunnel["proto"] == "http":
                return tunnel["public_url"]

        return "No active tunnels found."

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Process and clean environment variables
PORT = int(os.getenv("PORT", 5050))
raw_domain = os.getenv("DOMAIN", remove_http_https(get_ngrok_url()))
DOMAIN = re.sub(r"(^\w+:|^)\/\/|\/+$", "", raw_domain)  # Strips protocols and trailing slashes

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY not set in environment variables")

# OpenAI model settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Global variables to track active calls
is_call_in_progress: bool = False
active_calls: Dict[str, Dict[str, Any]] = {}

################################################################################
# DATA MODELS
################################################################################

class CallState:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        self.transcript = []
        self.audio_buffer = bytearray()
        self.openai_messages = [
            {"role": "system", "content": "You are a Pirate. Greet like one"}
        ]
        self.is_speaking = False
        self.is_waiting_for_response = False
        self.speech_queue = []

################################################################################
# FASTAPI INITIALIZATION
################################################################################

# Create FastAPI application with CORS support
app = FastAPI(title="OpenAI Integration")


# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

################################################################################
# HELPER FUNCTIONS
################################################################################


async def cleanup_stale_calls():
    """
    Periodic task to clean up stale calls
    """
    while True:
        try:
            now = datetime.now()
            stale_sids = []
            
            for call_sid, call_state in active_calls.items():
                # If call is inactive for more than 3 minutes, mark for cleanup
                if now - call_state.last_activity > timedelta(minutes=3):
                    stale_sids.append(call_sid)
            
            # Remove stale calls
            for call_sid in stale_sids:
                logger.info(f"🧹 Cleaning up stale call: {call_sid}")
                del active_calls[call_sid]
                
            # Wait for next cleanup cycle
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"❌ Error in cleanup task: {str(e)}", exc_info=True)
            await asyncio.sleep(60)  # Wait before retrying

################################################################################
# STARTUP EVENT
################################################################################

# Register startup event
@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks when FastAPI app starts"""
    asyncio.create_task(cleanup_stale_calls())
    logger.info("🚀 Started background task for cleaning up stale calls")
    
    # Verify server configuration
    asyncio.create_task(verify_server_availability())

################################################################################
# FASTAPI ENDPOINTS
################################################################################

@app.get('/', response_class=HTMLResponse)
async def index_page():
    """Root endpoint to verify the server is running"""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Healthcare Provider Assistant</title>
        </head>
        <body>
            <h1>Healthcare Provider Assistant</h1>
            <p>Twilio + OpenAI Realtime API integration is running!</p>
        </body>
    </html>
    """)


@app.get('/health')
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "active_calls": len(active_calls),
        "openai_configured": bool(OPENAI_API_KEY)
    }

@app.post('/incoming-call')
async def handle_incoming_call(request: Request):
    """
    Handle incoming Twilio voice calls and connect them to our websocket
    """
    global is_call_in_progress
    
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        
        # Check if we are already handling a call
        if is_call_in_progress:
            logger.warning(f"⚠️ Call already in progress. Rejecting call {call_sid}")
            
            response = VoiceResponse()
            response.say("We're sorry, but we're currently handling another call. Please try again later.")
            response.hangup()
            
            return Response(
                content=str(response),
                media_type="text/xml"
            )
        
        # Mark that we are now handling a call
        is_call_in_progress = True
        
        # Initialize call state
        call_state = CallState(call_sid)
        active_calls[call_sid] = call_state
        
        logger.info(f"📞 Incoming call: {call_sid}")
        
        # Build TwiML response with Media Streams
        response = VoiceResponse()
        
        # Connect to WebSocket for real-time audio processing
        connect = Connect()
        connect.stream(url=f"wss://{DOMAIN}/media-stream")
        
        response.append(connect)
        
        # Set a reasonable timeout for the call
        response.say("Thank you for your call. Goodbye.", loop=0)
        
        return Response(
            content=str(response),
            media_type="text/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ Error handling incoming call: {str(e)}", exc_info=True)
        
        # Return a basic response even in case of error
        response = VoiceResponse()
        response.say("We're sorry, but we encountered an error. Please try again later.")
        response.hangup()
        
        return Response(
            content=str(response),
            media_type="text/xml"
        )

@app.post('/call-status')
async def call_status_callback(request: Request):
    """
    Handle Twilio call status callbacks according to Twilio's expectations.
    """
    global is_call_in_progress
    
    try:
        # Log that we received a callback
        logger.info("📞 Received call status callback")
        
        # Try to parse as form data (which is how Twilio sends callbacks)
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "Unknown")
        call_status = form_data.get("CallStatus", "Unknown")
        
        logger.info(f"📞 Call status update: SID {call_sid}, Status {call_status}")
        
        # Reset the call in progress flag if the call is complete
        if call_status in ["completed", "failed", "busy", "no-answer", "canceled"]:
            is_call_in_progress = False
            
            # Clean up any resources associated with this call
            if call_sid in active_calls:
                del active_calls[call_sid]
                
            logger.info(f"🔄 Reset call in progress flag due to status: {call_status}")
            
        # Return an empty TwiML response as required by Twilio
        twiml_response = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
        
        # Create a Response object with the correct content type
        return Response(
            content=twiml_response,
            media_type="text/xml",
            status_code=200
        )
            
    except Exception as e:
        logger.error(f"❌ Error in call status callback: {str(e)}", exc_info=True)
        
        # Even in case of error, return an empty TwiML response
        twiml_response = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
        return Response(
            content=twiml_response,
            media_type="text/xml",
            status_code=200
        )

@app.get('/reset-calls')
async def reset_calls():
    """
    Reset the call tracking system in case of issues.
    """
    global is_call_in_progress
    
    old_status = is_call_in_progress
    is_call_in_progress = False
    
    # Clear active calls
    active_calls.clear()
    
    logger.info(f"🔄 Reset call tracking system. Old state: in_progress={old_status}")
    return {"success": True, "message": f"Reset call in progress flag from {old_status} to False"}

@app.websocket('/media-stream')
async def handle_media_stream(websocket: WebSocket):
    """
    Handle WebSocket connections between Twilio and OpenAI.
    This creates a direct bridge between Twilio's Media Streams and OpenAI's Realtime API.
    """
    logger.info("📡 Client connected to WebSocket")
    await websocket.accept()
    
    # Track the Twilio stream SID
    stream_sid = None
    
    try:
        # Connect to OpenAI's Realtime API
        async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            logger.info("🔗 Connected to OpenAI Realtime API")
            
            async def receive_from_twilio():
                """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
                nonlocal stream_sid
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        logger.debug(f"Received from Twilio: {data['event'] if 'event' in data else data}")
                        
                        # Handle the 'start' event to get the stream SID
                        if 'start' in data:
                            stream_sid = data['start']['streamSid']
                            logger.info(f"📞 Incoming stream started: {stream_sid}")
                            
                            # Send connection acknowledgment back to Twilio - THIS IS CRITICAL
                            # Make sure the format exactly matches what Twilio expects
                            await websocket.send_text(json.dumps({
                                "event": "connected",
                                "streamSid": stream_sid
                            }))
                        
                        # Forward audio data to OpenAI
                        elif 'media' in data and data.get('event') == 'media' and openai_ws.open:
                            if stream_sid != data.get('streamSid'):
                                logger.warning(f"⚠️ Stream SID mismatch: expected {stream_sid}, got {data.get('streamSid')}")
                                continue
                                
                            # Format the audio data according to OpenAI's expected format
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data['media']['payload']
                            }
                            await openai_ws.send(json.dumps(audio_append))
                        
                        # Echo back mark events for synchronization
                        elif 'mark' in data and stream_sid:
                            # Ensure the event format matches exactly what Twilio expects
                            await websocket.send_text(json.dumps({
                                "event": "mark",
                                "streamSid": stream_sid,
                                "mark": {
                                    "name": data['mark']['name']
                                }
                            }))
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.info("📡 Twilio WebSocket connection closed")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error from Twilio message: {e}")
                except Exception as e:
                    logger.error(f"❌ Error in receive_from_twilio: {str(e)}", exc_info=True)
            
            async def send_to_twilio():
                """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
                nonlocal stream_sid
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        
                        # Log important events
                        if response['type'] in LOG_EVENT_TYPES:
                            logger.info(f"🤖 OpenAI event: {response['type']}")
                            logger.debug(f"OpenAI response: {response}")
                        
                        # Handle session updated confirmation
                        if response['type'] == 'session.updated':
                            logger.info("✅ OpenAI session updated successfully")
                        
                        # Handle audio responses and forward to Twilio
                        if response['type'] == 'response.audio.delta' and response.get('delta') and stream_sid:
                            try:
                                # Format audio data for Twilio's expected format
                                # This is the exact format Twilio expects
                                audio_delta = {
                                    "event": "media",  # event type must be "media"
                                    "streamSid": stream_sid,  # must use the correct stream SID
                                    "media": {
                                        "payload": response['delta']  # base64 encoded audio
                                    }
                                }
                                
                                # Validate the message structure
                                if not isinstance(audio_delta, dict) or \
                                   not audio_delta.get("event") or \
                                   not audio_delta.get("streamSid") or \
                                   not isinstance(audio_delta.get("media"), dict) or \
                                   not audio_delta["media"].get("payload"):
                                    logger.error("❌ Invalid message structure for Twilio")
                                    continue
                                    
                                # Validate data types
                                if not isinstance(audio_delta["event"], str) or \
                                   not isinstance(audio_delta["streamSid"], str) or \
                                   not isinstance(audio_delta["media"]["payload"], str):
                                    logger.error("❌ Invalid data types in message for Twilio")
                                    continue
                                
                                # Send the audio back to Twilio
                                await websocket.send_text(json.dumps(audio_delta))
                            except Exception as e:
                                logger.error(f"❌ Error processing audio data: {str(e)}", exc_info=True)
                        
                        # If there's no stream SID yet but we're getting responses, log a warning
                        elif not stream_sid and response['type'] == 'response.audio.delta':
                            logger.warning("⚠️ Received audio from OpenAI but no stream SID from Twilio yet")
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.info("🤖 OpenAI WebSocket connection closed")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error from OpenAI message: {e}")
                except Exception as e:
                    logger.error(f"❌ Error in send_to_twilio: {str(e)}", exc_info=True)
            
            # Initialize OpenAI session
            await initialize_openai_session(openai_ws)
            
            # Run both communication directions concurrently
            await asyncio.gather(receive_from_twilio(), send_to_twilio())
    except Exception as e:
        logger.error(f"❌ Error in WebSocket handler: {str(e)}", exc_info=True)
    finally:
        # Log disconnection
        logger.info("📡 WebSocket connection closed")

async def initialize_openai_session(openai_ws):
    """Initialize the OpenAI session with voice and settings."""
    try:
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "g711_ulaw",  # Match Twilio's format
                "output_audio_format": "g711_ulaw", # Match Twilio's format
                "voice": VOICE,
                "instructions": SYSTEM_MESSAGE,
                "modalities": ["text", "audio"],
                "temperature": 0.8,
            }
        }
        logger.info('Sending session update to OpenAI')
        await openai_ws.send(json.dumps(session_update))
        
        # Optional: Send initial greeting
        #await send_initial_conversation_item(openai_ws)
    except Exception as e:
        logger.error(f"❌ Error initializing OpenAI session: {str(e)}", exc_info=True)

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation so AI talks first."""
    try:
        initial_conversation_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Hello, this "
                        )
                    }
                ]
            }
        }
        await openai_ws.send(json.dumps(initial_conversation_item))
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("✅ Sent initial conversation item to OpenAI")
    except Exception as e:
        logger.error(f"❌ Error sending initial conversation: {str(e)}", exc_info=True)

################################################################################
# SERVER STARTUP FUNCTION
################################################################################

async def verify_server_availability():
    """
    Check if the server is available at the specified domain.
    This helps detect potential ngrok tunnel or port issues.
    """
    try:
        # First, check if we can connect to localhost at the specified port
        try:
            response = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ Local server is running on port {PORT}")
            else:
                logger.warning(f"⚠️ Local server returned unexpected status code: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to local server: {str(e)}")
        
        # Then check the public URL (if it includes http/https)
        if "http" in raw_domain:
            try:
                response = requests.get(f"{raw_domain}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Server is accessible via public URL: {raw_domain}")
                else:
                    logger.warning(f"⚠️ Public URL returned unexpected status code: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to public URL: {str(e)}")
                logger.warning(f"⚠️ This may cause issues with Twilio callbacks. Check your ngrok tunnel.")
                
        # Verify OpenAI API key is working
        try:
            if OPENAI_API_KEY:
                # Just do a simple test completion
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 5
                        },
                        timeout=5
                    )
                    if response.status_code == 200:
                        logger.info("✅ OpenAI API key is valid")
                    else:
                        logger.warning(f"⚠️ OpenAI API key validation failed: {response.status_code}")
            else:
                logger.warning("⚠️ OpenAI API key not set")
        except Exception as e:
            logger.warning(f"⚠️ Could not validate OpenAI API key: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error verifying server availability: {str(e)}")

def start_fastapi_server():
    """Start the FastAPI server for handling Twilio media streams and OpenAI integration"""
    logger.info(f"Starting FastAPI server on port {PORT}")
    # Use timeout settings to avoid hanging requests
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        timeout_keep_alive=30,  # Shorter keep-alive timeout
        log_level="info"
    )

if __name__ == "__main__":
    start_fastapi_server()
