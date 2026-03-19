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
from livekit.agents import llm, stt, tts, inference # Inference creates model instances from model strings 

# Load environment variables (API keys, etc.)
# Make sure you have a .env file with your keys
load_dotenv()


# This class defines how your AI behaves
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # This is like the "personality" or instructions for your AI
            instructions="You are an upbeat, slightly sarcastic voice AI for tech support. " \
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
        inference.TTS.from_model_string("cartesia/sonic-2:a167e0f3-df7e-4d52-a9c3-f949145efdab"),
        inference.TTS.from_model_string("inworld/inworld-tts-1"),
        ]
    ),

        # Voice Activity Detection: detects when user is speaking
        vad=silero.VAD.load(),
        # Multilingual Turn Detection: detects when the user has finished speaking, supports multiple languages
        turn_detection=MultilingualModel(),
    )

    # Step 3: Start the session
    await session.start(
        agent=Assistant(),   # our AI assistant
        room=ctx.room,       # the room where users are connected

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
        WorkerOptions(
            entrypoint_fnc=entrypoint  # tells LiveKit where to start
        )
    )