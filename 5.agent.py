import logging
from dotenv import load_dotenv

# Import core LiveKit agent components
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)

# Import plugins (extra features like noise cancellation & voice detection)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import (
    llm,
    stt,
    tts,
    inference,
)  # Inference creates model instances from model strings
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
from livekit.agents import function_tool, RunContext
from livekit.agents import mcp
import httpx


# logger setup, logging.getLogger(__name__) creates a logger for this specific file, . __name__ is a special variable in Python that holds the name of the current module.
logger = logging.getLogger(__name__)

# Load environment variables (API keys, etc.)
# Make sure you have a .env file with your keys
load_dotenv()


# This class defines how your AI behaves
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # This is like the "personality" or instructions for your AI
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support."
                "Help the caller fix issues without rambling, and keep replies under 3 sentences."
                "You can also look up the weather if asked."
                "You can also answer question about LiveKit's features and pricing, using the MCP tool to fetch real-time info from https://api.livekit.cloud/mcp When user asks about LiveKit's features or pricing, use the MCP tool to provide accurate and up-to-date information. Always check the latest data from the MCP API before responding to ensure your answers reflect any recent changes or updates in LiveKit's offerings."
            ),
        )

    @function_tool  # Callable tool for the LLM to use. The LLM can call this function when it thinks it's appropriate based on the conversation.
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Look up current weather for a city or location. Use this when the user asks about weather, temperature, or conditions in a specific place.

        Args:
            location: The city name or location to look up (e.g. "London", "New York", "Tokyo").
        """
        logger.info("Looking up weather for %s", location)

        try:
            # Optional: have the agent say something while fetching data
            await context.session.say("Let me check the weather for you...")  
            context.disallow_interruptions()
            # So if we are using any task that should not be undone in between, like payment or booking, we can use context.session.run_critical_section to make sure it runs without interruption.
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 1. Geocoding: Convert city name to lat/lon
                geo_response = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1},
                )
                geo_data = geo_response.json()

                if not geo_data.get("results"):
                    return f"Could not find location: {location}"

                lat = geo_data["results"][0]["latitude"]
                lon = geo_data["results"][0]["longitude"]
                place_name = geo_data["results"][0]["name"]

                # 2. Fetch weather (current_weather=true enables the current_weather object in response)
                weather_response = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current_weather": True,
                        "temperature_unit": "fahrenheit",
                    },
                )

                if weather_response.status_code != 200:
                    return f"Weather data for {place_name} is currently unavailable."

                w_data = weather_response.json().get("current_weather", {})
                if not w_data:
                    return f"Weather data for {place_name} is currently unavailable."

                temp = w_data.get("temperature", "N/A")
                code = w_data.get("weathercode", 0)
                # WMO weather codes: 0=clear, 1-3=cloudy, 45-48=fog, 51-67=rain/drizzle, 71-77=snow, 80-99=showers/thunderstorm
                conditions = _weather_code_to_text(code)

                return f"In {place_name}, it's {temp} degrees Fahrenheit with {conditions}."

        except httpx.TimeoutException:
            return "The request timed out. Please try again."
        except Exception as exc:
            logger.error("Weather tool error: %s", exc)
            return "An unexpected error occurred while fetching the weather."


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable description."""
    if code == 0:
        return "clear skies"
    if code in (1, 2, 3):
        return "partly cloudy" if code == 1 else "cloudy"
    if 45 <= code <= 48:
        return "foggy"
    if 51 <= code <= 67:
        return "rain"
    if 71 <= code <= 77:
        return "snow"
    if 80 <= code <= 99:
        return "showers or storms" if code >= 95 else "rain showers"
    return "variable conditions"


# This server handles incoming users and assigns them to the agent
server = AgentServer()


# This function runs every time a user joins the room
@server.rtc_session()
async def entrypoint(ctx: JobContext):

    # Step 1: Connect the agent to the room
    # Without this, your agent can't hear or speak
    await ctx.connect()

    # Step 2: Create a session (this defines the voice pipeline)
    session = AgentSession(
        # Speech-to-Text: converts user's voice to text
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("deepgram/nova-3"),
                inference.STT.from_model_string("assemblyai/universal-streaming"),
            ]
        ),
        llm=llm.FallbackAdapter(
            [
                inference.LLM.from_model_string("openai/gpt-4.1-mini"),
                inference.LLM.from_model_string("google/gemini-2.5-flash"),
            ]
        ),
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string(
                    "cartesia/sonic-2:a167e0f3-df7e-4d52-a9c3-f949145efdab"
                ),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        # Voice Activity Detection: detects when user is speaking
        vad=silero.VAD.load(),
        # Multilingual Turn Detection: detects when the user has finished speaking, supports multiple languages
        turn_detection=MultilingualModel(),
        # - Preemtive generation helps the agent start forming a response before the user finishes speaking.
        # The agents wait for a clear end of turn before actually speaking, but the thinking has already begun
        preemptive_generation=True,
        
        # When the session starts, it connects to the MCP server to enable real-time access to LiveKit's features and pricing information. This allows the agent to provide accurate and up-to-date responses when users ask about LiveKit's offerings.
        mcp_servers=[mcp.MCPServerConfig("https://api.livekit.cloud/mcp")],
    )

    # Aggregate metrics across all conversation turns. It tracks token count for the LLM, audio duration for STT and TTS. and cost estimation based on usage.
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    # Metrics collected event fires after each component finishes processing
    # (e.g. after STT converts speech to text, after LLM generates a response, etc.).
    # We are capturing EOU metrics specifically, which stands for "End Of Utterance".
    # This represents the moment the turn detector decides that the user has finished speaking.
    # Later, when the worker shuts down, we log usage to see per-session resource consumption.
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # Agent cycles through several states during a conversation: listening, thinking, and speaking.
    # The agent_state_changed event fires each time the agent transitions between these states.
    # When the agent enters the "speaking" state, we measure the delay
    # between when the user finished speaking (EOU) and when the agent
    # starts producing audio (first response frame). This represents the user's perceived wait time.
    # ("How fast the AI responds after the user stops talking")
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if (
            ev.new_state == "speaking"
            and last_eou_metrics
            and session.current_speech
            and last_eou_metrics.speech_id == session.current_speech.id
        ):
            delay_ms = last_eou_metrics.end_of_utterance_delay * 1000
            logger.info("Time to first audio frame: %sms", delay_ms)

    # This function runs when the session is shutting down.
    # It logs a summary of total usage (tokens, audio duration, etc.)
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    # Register shutdown callback so usage is printed when session ends
    ctx.add_shutdown_callback(log_usage)

    # Step 3: Start the session
    await session.start(
        agent=Assistant(),  # our AI assistant
        room=ctx.room,  # the room where users are connected
        # Optional settings for audio input
        room_input_options=RoomInputOptions(
            # Removes background noise (very useful in real calls)
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


# This is the starting point of your program
# Without this, your script won't run
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)  # tells LiveKit where to start
    )
