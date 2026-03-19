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
from requests import session

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
            instructions="You are an upbeat, slightly sarcastic voice AI for tech support. "
            "Help the caller fix issue without rambling, and keep replies under 3 sentences.",
        )


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
        
        #- Preemtive generation helps the agent start forming a response before the user finishes speaking.
        # The agents wait for a clear end of turn before actually speaking, but the thinking has already begun
        preemptive_generation=True,  
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
