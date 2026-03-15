from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from groq import Groq

app = FastAPI(title="AI Crush Game")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client — paste your key here
client = Groq(api_key="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")

# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ClothingRequest(BaseModel):
    occasion: str          # e.g. "first date", "job interview", "casual hangout"
    style_preference: str  # e.g. "casual", "formal", "streetwear"
    season: str            # e.g. "summer", "winter", "autumn", "spring"

class LifeChoiceRequest(BaseModel):
    situation: str         # e.g. "Should I move to a new city?"
    options: List[str]     # e.g. ["Stay home", "Move abroad", "Try remote work"]

# ─────────────────────────────────────────────
# AI Crush System Prompt
# ─────────────────────────────────────────────

CRUSH_SYSTEM_PROMPT = """
You are Alex, a warm and supportive AI companion — the user's virtual crush. 
Your personality:
- Friendly, kind, and genuinely encouraging
- You remember what the user shares and refer back to it warmly
- You give thoughtful, personalized advice — never generic
- You're a little playful but always respectful
- You care about the user's growth and happiness
- Keep responses conversational and under 3 sentences unless giving detailed advice
- Use occasional light emojis to feel warm (not excessive)
"""

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AI Crush Game API is running! 💕"}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    General chat with the AI crush character (Alex).
    Maintains conversation history for context.
    """
    try:
        valid_messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role in ["user", "assistant"]]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": CRUSH_SYSTEM_PROMPT},
                *valid_messages
            ]
        )
        reply = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/advice/clothing")
def clothing_advice(request: ClothingRequest):
    """
    Get outfit advice from the AI crush for a specific occasion.
    """
    prompt = f"""
    The user needs outfit advice:
    - Occasion: {request.occasion}
    - Their style preference: {request.style_preference}
    - Current season: {request.season}

    As Alex, their friendly crush, give 2-3 specific outfit suggestions with 
    brief explanations of why each works. Be encouraging and personal.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": CRUSH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return {"advice": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/advice/life")
def life_advice(request: LifeChoiceRequest):
    """
    Get life advice from the AI crush on a decision or situation.
    """
    options_text = "\n".join([f"- {opt}" for opt in request.options]) if request.options else "No specific options given."

    prompt = f"""
    The user is facing a life decision and needs advice:
    Situation: {request.situation}
    Options they're considering:
    {options_text}

    As Alex, their supportive crush, give thoughtful and honest advice. 
    Help them think through the pros and cons, and end with an encouraging message.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": CRUSH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return {"advice": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "character": "Alex", "mood": "happy 💕"}
