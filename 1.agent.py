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


# Load environment variables (API keys, etc.)
# Make sure you have a .env file with your keys
load_dotenv()


# This class defines how your AI behaves
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # This is like the "personality" or instructions for your AI
            instructions="You are a helpful voice AI assistant.",
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
        stt="deepgram/nova-3",

        # LLM: brain of the AI (understands & replies)
        llm="openai/gpt-4.1-mini",

        # Text-to-Speech: converts AI response to voice
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",

        # Voice Activity Detection: detects when user is speaking
        vad=silero.VAD.load(),
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