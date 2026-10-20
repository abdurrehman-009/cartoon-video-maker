"""
ToonGenius AI — Your Cartoon Creation Studio
============================================

A beginner-friendly Streamlit app that uses Groq (text generation only) to create
cartoon stories, characters, scenes, and ready-to-use image / video prompts.

* Story, character and scene generation  -> Groq (configured with GROQ_API_KEY)
* Image generation                       -> OPTIONAL separate image provider (IMAGE_* secrets)
* Video generation                       -> OPTIONAL separate video provider (VIDEO_* secrets)

Groq is NOT used to generate images or videos. If no provider is configured, the app
says so clearly and never shows fake results.

Secrets (Streamlit Cloud -> App -> Settings -> Secrets) or environment variables:
    GROQ_API_KEY, GROQ_MODEL (optional),
    IMAGE_API_KEY, IMAGE_API_URL, IMAGE_MODEL (optional),
    VIDEO_API_KEY, VIDEO_API_URL, VIDEO_MODEL (optional)
"""

import base64
import copy
import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import groq
import requests
import streamlit as st
from groq import Groq
from PIL import Image

# The page config must be the first Streamlit command.
st.set_page_config(
    page_title="ToonGenius AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CONSTANTS
# =============================================================================

APP_TITLE = "ToonGenius AI — Your Cartoon Creation Studio"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

PAGES = [
    "Dashboard",
    "Story Generator",
    "Character Studio",
    "Scene Builder",
    "Image Studio",
    "Video Studio",
    "My Projects",
    "Settings",
]
PAGE_ICONS = {
    "Dashboard": "🏠",
    "Story Generator": "📖",
    "Character Studio": "🎭",
    "Scene Builder": "🎬",
    "Image Studio": "🖼️",
    "Video Studio": "🎞️",
    "My Projects": "📁",
    "Settings": "⚙️",
}

GENRES = ["Adventure", "Comedy", "Fantasy", "Educational", "Friendship"]
AUDIENCES = ["Children", "General"]
DURATIONS = [15, 30, 60]
LANGUAGES = [
    "English", "Urdu", "Hindi", "Arabic", "Spanish", "French",
    "German", "Portuguese", "Turkish", "Chinese", "Japanese", "Korean",
]
STYLES = ["3D cartoon", "2D cartoon", "Anime-inspired", "Storybook"]
TONES = ["Funny", "Emotional", "Adventurous", "Educational"]
AGE_CATEGORIES = ["Child", "Teen", "Adult", "Elder", "Ageless / magical"]

# How many scenes to ask for, depending on the story duration.
SCENES_FOR_DURATION = {15: 3, 30: 5, 60: 8}

# Input limits (keeps prompts sensible and protects your API quota).
MAX_IDEA = 1500
MAX_PROMPT = 2000
MAX_SHORT = 300
MAX_MEDIUM = 1000

# Max characters stored per scene field.
FIELD_LIMITS = {
    "location": 300, "characters": 300, "camera": 300, "lighting": 400,
    "action": 1000, "narration": 1000, "dialogue": 1500, "visual": 1000,
    "image_prompt": MAX_PROMPT, "video_prompt": MAX_PROMPT,
}

# Image aspect ratios and the pixel sizes sent to the image provider.
# ADAPTER NOTE: different providers support different sizes. If your provider
# rejects these values, edit this dictionary to match its documentation.
IMAGE_ASPECTS = {
    "Square (1:1)": "1:1",
    "Landscape (16:9)": "16:9",
    "Vertical (9:16)": "9:16",
    "Portrait (3:4)": "3:4",
}
IMAGE_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    "3:4": "1024x1365",
}

VIDEO_ASPECTS = ["16:9", "9:16", "1:1"]
VIDEO_LENGTHS = [3, 5, 8, 10]
VIDEO_POLL_SECONDS = 5
VIDEO_TIMEOUT_SECONDS = 300
MAX_DOWNLOAD_BYTES = 150 * 1024 * 1024  # 150 MB safety cap for downloaded videos

# Default values for every input widget whose value we want to keep when the
# user moves between pages.
WIDGET_DEFAULTS = {
    "dash_idea": "",
    "story_idea": "",
    "sg_genre": GENRES[0],
    "sg_audience": AUDIENCES[0],
    "sg_duration": 30,
    "sg_language": "English",
    "sg_style": STYLES[0],
    "sg_tone": TONES[0],
    "ch_name": "",
    "ch_species": "",
    "ch_age": AGE_CATEGORIES[0],
    "ch_personality": "",
    "ch_appearance": "",
    "ch_clothing": "",
    "ch_colors": "",
    "ch_abilities": "",
    "ch_role": "",
    "ch_style": STYLES[0],
    "img_prompt": "",
    "img_aspect": list(IMAGE_ASPECTS.keys())[0],
    "img_style": STYLES[0],
    "vid_prompt": "",
    "vid_aspect": VIDEO_ASPECTS[0],
    "vid_length": 5,
    "vid_mode": "Text-to-video",
}

# =============================================================================
# 2. STYLING (custom CSS)
# =============================================================================

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #0b1030 0%, #141a4a 50%, #231457 100%);
    color: #eef0ff;
}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"],
.stApp .stCaption, .stApp small { color: #eef0ff; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #ffffff; letter-spacing: 0.2px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090d29 0%, #1a1052 100%);
    border-right: 1px solid rgba(167, 139, 250, 0.25);
}
.brand {
    font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #fbbf24);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero {
    background: linear-gradient(120deg, rgba(124,58,237,0.55), rgba(236,72,153,0.40), rgba(251,191,36,0.25));
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 0.3rem 0; font-size: 2rem; line-height: 1.2; }
.hero p { margin: 0; opacity: 0.92; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 16px; padding: 0.8rem 1rem;
}
div[data-testid="stMetricValue"] { color: #f9a8d4; }
.stButton > button, .stDownloadButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7c3aed, #ec4899);
    color: #ffffff; border: 0; border-radius: 12px; font-weight: 600;
    padding: 0.5rem 0.9rem;
    transition: transform 0.08s ease, box-shadow 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 6px 18px rgba(236,72,153,0.35); color: #fff;
}
.stButton > button:disabled { opacity: 0.45; }
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: rgba(255,255,255,0.08) !important; color: #ffffff !important;
    border: 1px solid rgba(167,139,250,0.35) !important; border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important; color: #ffffff !important;
    border: 1px solid rgba(167,139,250,0.35) !important; border-radius: 10px !important;
}
div[data-baseweb="select"] span { color: #ffffff !important; }
.streamlit-expanderHeader, details summary { color: #ffffff !important; }
@media (max-width: 640px) {
    .hero h1 { font-size: 1.45rem; }
    .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
"""

# =============================================================================
# 3. SMALL UTILITIES
# =============================================================================


class ToonError(Exception):
    """An error with a friendly, user-readable message."""


def now_str():
    """Current UTC time as a readable string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def new_id():
    """Short random identifier."""
    return uuid.uuid4().hex[:8]


def get_secret(name, default=""):
    """
    Read a setting from Streamlit Secrets first, then from environment variables.
    Returns `default` when neither is set. Never logs the value.
    """
    value = None
    try:
        if name in st.secrets:
            value = st.secrets[name]
    except Exception:
        value = None
    if value is None or str(value).strip() == "":
        value = os.environ.get(name, default)
    if value is None:
        return default
    return str(value).strip()


def mask_secret(value):
    """Show only the last 4 characters of a secret (never the full key)."""
    if not value:
        return "Not set"
    if len(value) < 12:
        return "•" * 8
    return "•" * 8 + value[-4:]


def host_only(url):
    """Show only the host part of a URL (query strings may contain tokens)."""
    if not url:
        return "Not set"
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/…" if parsed.netloc else "Invalid URL"


def scrub(text, secret_values):
    """Remove any secret values from text before showing it to the user."""
    for secret in secret_values:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


def clean_str(value, limit=None, default=""):
    """Convert any value to a trimmed string (optionally clipped to `limit`)."""
    if value is None:
        text = ""
    elif isinstance(value, (list, tuple)):
        text = ", ".join(clean_str(v) for v in value if v not in (None, ""))
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.strip()
    if not text:
        text = default
    if limit and len(text) > limit:
        text = text[:limit].rstrip()
    return text


def check_text(label, value, max_len, required=True):
    """Validate a user text field. Raises ToonError with a friendly message."""
    value = (value or "").strip()
    if required and not value:
        raise ToonError(f"Please fill in: {label}.")
    if len(value) > max_len:
        raise ToonError(f"{label} is too long ({len(value)} characters). Maximum is {max_len}.")
    return value


def flash(kind, message):
    """Queue a message that is shown at the top of the next page render."""
    st.session_state["flash"] = (kind, message)


def show_flash():
    """Display and clear the queued message (if any)."""
    item = st.session_state.get("flash")
    if item:
        kind, message = item
        st.session_state["flash"] = None
        {"success": st.success, "error": st.error, "warning": st.warning}.get(kind, st.info)(message)


def copy_box(label, text, key):
    """
    Streamlit has no native clipboard API, so 'Copy' buttons reveal the text in a
    code box. The box has a built-in copy icon in its top-right corner.
    """
    flag = f"show_{key}"
    if st.button(f"📋 {label}", key=f"btn_{key}"):
        st.session_state[flag] = not st.session_state.get(flag, False)
    if st.session_state.get(flag):
        st.code(text or "(empty)", language=None, wrap_lines=True)
        st.caption("Click the copy icon in the top-right corner of the box above.")


def go_to(page, **state_updates):
    """Button callback: set some session values and jump to another page."""
    for key, value in state_updates.items():
        st.session_state[key] = value
    st.session_state["nav"] = page


# =============================================================================
# 4. PROJECT DATA MODEL
# =============================================================================


def new_project(name="Untitled Project"):
    """Create an empty project dictionary."""
    stamp = now_str()
    return {
        "id": new_id(),
        "name": name,
        "created": stamp,
        "updated": stamp,
        "idea": "",
        "story": None,
        "characters": [],
        "scenes": [],
        "assets": {"images": [], "videos": []},
    }


def init_state():
    """Create all session-state values the first time the app runs."""
    ss = st.session_state
    ss.setdefault("work", new_project())          # the project being edited
    ss.setdefault("projects", {})                 # saved projects (this session only)
    ss.setdefault("image_cache", {})              # generated image bytes (this session only)
    ss.setdefault("video_cache", {})              # generated video bytes (this session only)
    ss.setdefault("flash", None)
    ss.setdefault("import_counter", 0)
    ss.setdefault("nav", "Dashboard")
    for key, value in WIDGET_DEFAULTS.items():
        ss.setdefault(key, value)
    # Streamlit forgets widget values for widgets that are not shown on the current
    # page. Re-assigning them every run keeps what the user typed.
    for key in WIDGET_DEFAULTS:
        ss[key] = ss[key]


def save_current_project():
    """Store a copy of the working project in the saved-projects list."""
    work = st.session_state.work
    work["updated"] = now_str()
    st.session_state.projects[work["id"]] = copy.deepcopy(work)


def cb_save_project():
    save_current_project()
    flash("success", "Project saved for this session. Download it as JSON in My Projects to keep it permanently.")


def cb_new_project():
    st.session_state.work = new_project()
    flash("info", "Started a new empty project. Projects you saved earlier are still in My Projects.")


def cb_load_project(project_id):
    project = st.session_state.projects.get(project_id)
    if project:
        st.session_state.work = copy.deepcopy(project)
        flash("success", f"Loaded project “{project['name']}”.")


def story_to_text(story, scenes):
    """Turn the story and scenes into plain text (for copy / reading)."""
    lines = [story["title"], "", story["summary"], "", "CHARACTERS"]
    for ch in story["characters"]:
        lines.append(f"- {ch['name']}: {ch['description']}")
    lines += ["", "BEGINNING", story["beginning"], "", "MIDDLE", story["middle"], "", "ENDING", story["ending"]]
    for sc in scenes:
        lines += [
            "",
            f"SCENE {sc['number']} ({sc['duration']}s) — {sc['location']}",
            f"Characters: {sc['characters']}",
            f"Action / expressions: {sc['action']}",
            f"Camera: {sc['camera']}",
            f"Lighting / background: {sc['lighting']}",
            f"Narration: {sc['narration']}",
            f"Dialogue: {sc['dialogue']}",
            f"Visual description: {sc['visual']}",
            f"Image prompt: {sc['image_prompt']}",
            f"Video prompt: {sc['video_prompt']}",
        ]
    return "\n".join(lines)


# =============================================================================
# 5. GROQ (TEXT GENERATION)
# =============================================================================


def parse_json_text(text):
    """Parse JSON from a model reply, tolerating code fences and extra text."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in the reply.")
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("The reply was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("The JSON reply must be an object.")
    return data


def groq_chat_json(system_prompt, user_prompt, temperature=0.8, max_tokens=5000):
    """
    Ask Groq for a JSON answer and return it as a Python dict.
    Tries JSON mode first; if the model rejects it, retries without JSON mode.
    Raises ToonError with a friendly message on any problem.
    """
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise ToonError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud → your app → Settings → Secrets "
            "(or set it as an environment variable for local testing). See the Settings page for help."
        )
    model = get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL
    client = Groq(api_key=api_key, timeout=60.0, max_retries=1)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(2):
        kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if attempt == 0:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
            return parse_json_text(response.choices[0].message.content or "")
        except groq.AuthenticationError:
            raise ToonError("Groq rejected your API key. Check GROQ_API_KEY in your Secrets.") from None
        except groq.RateLimitError:
            raise ToonError("Groq rate limit reached. Please wait a minute and try again.") from None
        except groq.APITimeoutError:
            raise ToonError("Groq took too long to answer. Please try again.") from None
        except groq.APIConnectionError:
            raise ToonError("Could not reach Groq. Check your internet connection and try again.") from None
        except groq.NotFoundError:
            raise ToonError(
                f"The Groq model “{model}” is not available to your account. "
                "Set GROQ_MODEL in your Secrets to a model you can use."
            ) from None
        except groq.BadRequestError:
            if attempt == 0:
                continue  # retry once without JSON mode
            raise ToonError("Groq could not process this request. Try shorter or simpler input.") from None
        except groq.APIStatusError as exc:
            raise ToonError(f"Groq returned an error (HTTP {exc.status_code}). Please try again later.") from None
        except ValueError:
            continue  # malformed JSON -> try once more
    raise ToonError("The AI replied in a format the app could not read. Please click Generate again.")


# ---- Normalizers: turn any model reply (or imported file) into safe, known data ----


def normalize_dialogue(value):
    """Dialogue may be a string or a list of {character, line}; return plain text."""
    if isinstance(value, str):
        return value.strip()[:FIELD_LIMITS["dialogue"]]
    lines = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                who = clean_str(item.get("character") or item.get("name") or item.get("speaker"))
                line = clean_str(item.get("line") or item.get("text") or item.get("dialogue"))
                if line:
                    lines.append(f"{who}: {line}" if who else line)
            elif item:
                lines.append(clean_str(item))
    return "\n".join(lines)[:FIELD_LIMITS["dialogue"]]


def normalize_scene(raw, number, default_duration):
    """Validate one scene dictionary and return it in the app's format."""
    if not isinstance(raw, dict):
        raise ValueError("A scene was not an object.")
    chars = raw.get("characters")
    if isinstance(chars, list):
        chars_text = ", ".join(
            clean_str(c.get("name") if isinstance(c, dict) else c) for c in chars if c
        )
    else:
        chars_text = clean_str(chars)
    try:
        duration = int(float(raw.get("duration_seconds") or raw.get("duration") or default_duration))
    except (TypeError, ValueError):
        duration = default_duration
    scene = {
        "sid": new_id(),
        "ver": 0,
        "number": number,
        "duration": min(max(duration, 1), 120),
        "location": clean_str(raw.get("location"), FIELD_LIMITS["location"]),
        "characters": clean_str(chars_text, FIELD_LIMITS["characters"]),
        "action": clean_str(raw.get("action_expression") or raw.get("action"), FIELD_LIMITS["action"]),
        "camera": clean_str(raw.get("camera"), FIELD_LIMITS["camera"]),
        "lighting": clean_str(raw.get("lighting_background") or raw.get("lighting"), FIELD_LIMITS["lighting"]),
        "narration": clean_str(raw.get("narration"), FIELD_LIMITS["narration"]),
        "dialogue": normalize_dialogue(raw.get("dialogue")),
        "visual": clean_str(raw.get("visual_description") or raw.get("visual"), FIELD_LIMITS["visual"]),
        "image_prompt": clean_str(raw.get("image_prompt"), FIELD_LIMITS["image_prompt"]),
        "video_prompt": clean_str(raw.get("video_prompt"), FIELD_LIMITS["video_prompt"]),
    }
    if not (scene["location"] or scene["action"] or scene["visual"] or scene["narration"]):
        raise ValueError("A scene had no usable content.")
    if not scene["image_prompt"]:
        scene["image_prompt"] = clean_str(scene["visual"] or scene["action"], FIELD_LIMITS["image_prompt"])
    if not scene["video_prompt"]:
        scene["video_prompt"] = clean_str(scene["action"] or scene["visual"], FIELD_LIMITS["video_prompt"])
    return scene


def normalize_scenes(raw_scenes, total_duration):
    """Validate a list of scenes (max 12) and number them 1..N."""
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("The reply had no scenes.")
    raw_scenes = raw_scenes[:12]
    default_duration = max(1, round(total_duration / len(raw_scenes)))
    return [normalize_scene(raw, i + 1, default_duration) for i, raw in enumerate(raw_scenes)]


def normalize_story(data, params):
    """Validate the story JSON. Returns (story_dict, scenes_list)."""
    title = clean_str(data.get("title"), 150)
    if not title:
        raise ValueError("The reply had no title.")
    characters = []
    raw_chars = data.get("characters")
    if isinstance(raw_chars, list):
        for item in raw_chars[:10]:
            if isinstance(item, dict):
                name = clean_str(item.get("name"), 80)
                if name:
                    characters.append({"name": name, "description": clean_str(item.get("description"), 600)})
            elif item:
                characters.append({"name": clean_str(item, 80), "description": ""})
    structure = data.get("structure") if isinstance(data.get("structure"), dict) else data
    story = {
        "title": title,
        "summary": clean_str(data.get("summary"), 800),
        "characters": characters,
        "beginning": clean_str(structure.get("beginning"), 1200),
        "middle": clean_str(structure.get("middle"), 1200),
        "ending": clean_str(structure.get("ending"), 1200),
        "genre": params["genre"],
        "audience": params["audience"],
        "duration": params["duration"],
        "language": params["language"],
        "style": params["style"],
        "tone": params["tone"],
    }
    scenes = normalize_scenes(data.get("scenes"), params["duration"])
    return story, scenes


def sanitize_story(raw):
    """Light validation for a story loaded from an imported project file."""
    if not isinstance(raw, dict) or not clean_str(raw.get("title")):
        return None
    characters = []
    for item in raw.get("characters") or []:
        if isinstance(item, dict) and clean_str(item.get("name")):
            characters.append({
                "name": clean_str(item.get("name"), 80),
                "description": clean_str(item.get("description"), 600),
            })
    return {
        "title": clean_str(raw.get("title"), 150),
        "summary": clean_str(raw.get("summary"), 800),
        "characters": characters,
        "beginning": clean_str(raw.get("beginning"), 1200),
        "middle": clean_str(raw.get("middle"), 1200),
        "ending": clean_str(raw.get("ending"), 1200),
        "genre": clean_str(raw.get("genre"), 40, GENRES[0]),
        "audience": clean_str(raw.get("audience"), 40, AUDIENCES[0]),
        "duration": raw.get("duration") if raw.get("duration") in DURATIONS else 30,
        "language": clean_str(raw.get("language"), 40, "English"),
        "style": clean_str(raw.get("style"), 40, STYLES[0]),
        "tone": clean_str(raw.get("tone"), 40, TONES[0]),
    }


# ---- Prompts ----

SCENE_SCHEMA = """{
  "scene_number": 1,
  "duration_seconds": 5,
  "location": "where the scene happens",
  "characters": ["Character name", "Character name"],
  "action_expression": "what the characters do and their facial expressions",
  "camera": "camera angle or movement",
  "lighting_background": "lighting, mood and background details",
  "narration": "narrator voice-over text (may be empty)",
  "dialogue": [{"character": "Character name", "line": "what they say"}],
  "visual_description": "what the viewer sees",
  "image_prompt": "detailed English prompt for an image generator",
  "video_prompt": "detailed English prompt describing motion for a video generator"
}"""

STORY_SCHEMA = (
    '{\n'
    '  "title": "story title",\n'
    '  "summary": "2-3 sentence summary",\n'
    '  "characters": [{"name": "...", "description": "look and personality"}],\n'
    '  "structure": {"beginning": "...", "middle": "...", "ending": "..."},\n'
    '  "scenes": [ ' + SCENE_SCHEMA.replace("\n", "\n    ") + ' ]\n'
    '}'
)

SYSTEM_STORY = (
    "You are an award-winning cartoon writer and storyboard artist. "
    "Reply with ONE valid JSON object only. No markdown, no code fences, no commentary. "
    "Follow exactly this JSON structure:\n" + STORY_SCHEMA
)


def style_rules(params):
    """Shared writing rules that depend on the user's choices."""
    rules = [
        f"Write narration and dialogue in {params['language']}.",
        "ALWAYS write image_prompt and video_prompt in English, even if the story language is different.",
        f"Every image_prompt and video_prompt must mention the {params['style']} look and describe the "
        "characters' appearance consistently from scene to scene.",
        "Keep content safe and friendly: no graphic violence, hate, or adult content.",
    ]
    if params["audience"] == "Children":
        rules.append("The audience is young children: simple words, gentle conflict, positive message.")
    return "\n".join(f"- {r}" for r in rules)


def generate_story(params, fresh_take=False):
    """Generate a complete story + scenes with Groq. Returns (story, scenes)."""
    scene_count = SCENES_FOR_DURATION[params["duration"]]
    user_prompt = (
        f"Create a {params['duration']}-second {params['genre']} cartoon story.\n"
        f"Idea: {params['idea']}\n"
        f"Audience: {params['audience']}. Tone: {params['tone']}. Animation style: {params['style']}.\n"
        f"Use exactly {scene_count} scenes. The scene durations must add up to about {params['duration']} seconds.\n"
        f"Rules:\n{style_rules(params)}\n"
    )
    if fresh_take:
        user_prompt += "Give a noticeably different take from any previous version of this idea.\n"
    data = groq_chat_json(SYSTEM_STORY, user_prompt, temperature=0.95 if fresh_take else 0.8)
    try:
        return normalize_story(data, params)
    except ValueError:
        # One more attempt, because small formatting problems are common.
        data = groq_chat_json(SYSTEM_STORY, user_prompt, temperature=0.7)
        try:
            return normalize_story(data, params)
        except ValueError:
            raise ToonError("The AI's story was incomplete. Please click Generate again.") from None


def story_context(story, scenes=None):
    """Short text summary of the story, used as context for later AI calls."""
    chars = "; ".join(f"{c['name']}: {c['description']}" for c in story["characters"])
    text = (
        f"Title: {story['title']}\nSummary: {story['summary']}\nCharacters: {chars}\n"
        f"Beginning: {story['beginning']}\nMiddle: {story['middle']}\nEnding: {story['ending']}\n"
        f"Style: {story['style']}. Tone: {story['tone']}. Language: {story['language']}. "
        f"Audience: {story['audience']}. Total length: {story['duration']} seconds."
    )
    if scenes:
        text += "\nCurrent scenes:\n" + "\n".join(
            f"{s['number']}. {s['location']} — {s['narration'] or s['action']}" for s in scenes
        )
    return text


def generate_scenes(story, count):
    """Rebuild the whole scene list from the story."""
    system = (
        "You are a cartoon storyboard artist. Reply with ONE valid JSON object only: "
        '{"scenes": [ ' + SCENE_SCHEMA + " ]}. No markdown or commentary."
    )
    params = {k: story[k] for k in ("language", "style", "audience")}
    user = (
        f"Story:\n{story_context(story)}\n\nCreate exactly {count} scenes covering the whole story. "
        f"Scene durations should add up to about {story['duration']} seconds.\nRules:\n{style_rules(params)}"
    )
    data = groq_chat_json(system, user, temperature=0.8)
    try:
        return normalize_scenes(data.get("scenes"), story["duration"])
    except ValueError:
        raise ToonError("The AI's scenes were incomplete. Please try again.") from None


def regenerate_scene(story, scenes, index, instruction):
    """Regenerate one scene while keeping it consistent with the rest of the story."""
    current = scenes[index]
    system = (
        "You are a cartoon storyboard artist. Reply with ONE valid JSON object only: "
        '{"scene": ' + SCENE_SCHEMA + "}. No markdown or commentary."
    )
    params = {k: story[k] for k in ("language", "style", "audience")}
    user = (
        f"Story:\n{story_context(story, scenes)}\n\n"
        f"Rewrite scene number {current['number']} (about {current['duration']} seconds). "
        f"Keep it consistent with the other scenes and the story flow.\n"
        f"Current version: location={current['location']}; action={current['action']}; "
        f"narration={current['narration']}\n"
        f"Extra instruction from the user: {instruction or 'Make it fresh and more visual.'}\n"
        f"Rules:\n{style_rules(params)}"
    )
    data = groq_chat_json(system, user, temperature=0.9)
    raw = data.get("scene") if isinstance(data.get("scene"), dict) else data
    try:
        scene = normalize_scene(raw, current["number"], current["duration"])
    except ValueError:
        raise ToonError("The AI's scene was incomplete. Please try again.") from None
    scene["sid"] = current["sid"]
    scene["ver"] = current["ver"] + 1  # new widget keys so the editor shows the new text
    return scene


def generate_character_profile(inputs):
    """Create a detailed character profile and an image prompt with Groq."""
    system = (
        "You are a professional cartoon character designer. Reply with ONE valid JSON object only: "
        '{"profile": "detailed character profile as plain text with short paragraphs: personality, '
        'backstory, habits, strengths, weaknesses, role in the story", '
        '"visual_description": "precise visual description of the character", '
        '"image_prompt": "professional English prompt for an image generator: character, pose, outfit, '
        'colors, art style, lighting, plain or simple background"}. No markdown or commentary.'
    )
    details = "\n".join(f"- {label}: {value}" for label, value in inputs["details"] if value)
    user = (
        f"Design this cartoon character in a {inputs['style']} style:\n{details}\n"
        "Keep it friendly and suitable for all audiences."
    )
    data = groq_chat_json(system, user, temperature=0.85, max_tokens=2500)
    profile = clean_str(data.get("profile"), 3000)
    image_prompt = clean_str(data.get("image_prompt"), MAX_PROMPT)
    if not profile or not image_prompt:
        raise ToonError("The AI's character profile was incomplete. Please try again.")
    return {
        "profile": profile,
        "visual_description": clean_str(data.get("visual_description"), 1500),
        "image_prompt": image_prompt,
    }


# =============================================================================
# 6. IMAGE AND VIDEO PROVIDERS (OPTIONAL, SEPARATE FROM GROQ)
# =============================================================================


def image_provider_ready():
    """Image generation needs at least IMAGE_API_KEY and IMAGE_API_URL."""
    return bool(get_secret("IMAGE_API_KEY") and get_secret("IMAGE_API_URL"))


def video_provider_ready():
    """Video generation needs at least VIDEO_API_KEY and VIDEO_API_URL."""
    return bool(get_secret("VIDEO_API_KEY") and get_secret("VIDEO_API_URL"))


def safe_request(method, url, secret_values=(), what="provider", **kwargs):
    """
    Make an HTTP request and convert every failure into a friendly ToonError.
    Secret values are removed from any error text shown to the user.
    """
    try:
        response = requests.request(method, url, **kwargs)
    except requests.exceptions.Timeout:
        raise ToonError(f"The {what} took too long to respond (timeout). Please try again.") from None
    except requests.exceptions.ConnectionError:
        raise ToonError(f"Could not connect to the {what}. Check the URL in your Secrets and your network.") from None
    except requests.exceptions.RequestException as exc:
        raise ToonError(f"The request to the {what} failed ({type(exc).__name__}).") from None
    if response.status_code in (401, 403):
        raise ToonError(f"The {what} rejected your API key (HTTP {response.status_code}). Check your Secrets.")
    if response.status_code == 429:
        raise ToonError(f"The {what} says you hit a rate limit or ran out of quota (HTTP 429).")
    if response.status_code >= 400:
        snippet = scrub(response.text[:300], secret_values)
        raise ToonError(f"The {what} returned HTTP {response.status_code}: {snippet}")
    return response


def validate_image_bytes(data):
    """Confirm the bytes are a real image. Returns (format, width, height)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            return img.format or "PNG", img.width, img.height
    except Exception:
        raise ToonError("The provider did not return a valid image, so nothing was saved.") from None


def generate_image_with_provider(prompt, aspect_label, style):
    """
    ADAPTER: image provider integration.

    This adapter sends an OpenAI-style "images/generations" request:
        POST IMAGE_API_URL  {"model", "prompt", "n": 1, "size"}
        Authorization: Bearer IMAGE_API_KEY
    and expects a reply like {"data": [{"b64_json": "..."} or {"url": "..."}]}.

    Not every provider uses this format. If yours differs, edit THIS function
    (the request body, headers and the response parsing) following your provider's docs.
    Returns (image_bytes, format_name, width, height).
    """
    api_key = get_secret("IMAGE_API_KEY")
    api_url = get_secret("IMAGE_API_URL")
    model = get_secret("IMAGE_MODEL")
    if not (api_key and api_url):
        raise ToonError("No image provider is configured. See the Settings page.")
    if not api_url.lower().startswith(("https://", "http://")):
        raise ToonError("IMAGE_API_URL must start with https://")

    aspect = IMAGE_ASPECTS[aspect_label]
    full_prompt = f"{style} style cartoon illustration. {prompt}\nAspect ratio {aspect}."[:4000]
    payload = {"prompt": full_prompt, "n": 1, "size": IMAGE_SIZES[aspect]}
    if model:
        payload["model"] = model
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = safe_request("POST", api_url, secret_values=[api_key], what="image provider",
                            headers=headers, json=payload, timeout=120)
    try:
        data = response.json()
    except ValueError:
        raise ToonError("The image provider did not return JSON. Its format may need a custom adapter.") from None

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        raise ToonError(
            "The image provider's reply was not in the expected format "
            "(edit generate_image_with_provider in app.py to match your provider)."
        )
    item = items[0]
    if item.get("b64_json"):
        try:
            image_bytes = base64.b64decode(item["b64_json"])
        except Exception:
            raise ToonError("The provider returned unreadable image data.") from None
    elif item.get("url"):
        download = safe_request("GET", item["url"], secret_values=[api_key], what="image download", timeout=60)
        image_bytes = download.content
    else:
        raise ToonError("The provider's reply contained no image. Nothing was generated.")

    if len(image_bytes) > 25 * 1024 * 1024:
        raise ToonError("The returned image is too large (over 25 MB).")
    fmt, width, height = validate_image_bytes(image_bytes)
    return image_bytes, fmt, width, height


def find_first(data, keys):
    """Search nested JSON for the first value stored under any of `keys`."""
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if value not in (None, "", [], {}):
                return value
        for value in data.values():
            found = find_first(value, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_first(value, keys)
            if found not in (None, "", [], {}):
                return found
    return None


def find_video_url(data):
    """Search nested JSON for a URL that looks like a finished video."""
    if isinstance(data, dict):
        direct = data.get("video_url")
        if isinstance(direct, str) and direct.startswith("http"):
            return direct
        for value in data.values():
            found = find_video_url(value)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_video_url(value)
            if found:
                return found
    elif isinstance(data, str) and data.startswith("http"):
        if urlparse(data).path.lower().endswith((".mp4", ".webm", ".mov")):
            return data
    return None


def looks_like_video(data):
    """Check the file header: MP4/MOV ('ftyp') or WebM/Matroska (EBML)."""
    return len(data) > 1000 and (data[4:8] == b"ftyp" or data[:4] == b"\x1aE\xdf\xa3")


def generate_video_with_provider(prompt, aspect, length, image_bytes=None, image_mime="image/png", on_status=None):
    """
    ADAPTER: video provider integration (submit a job, poll, download).

    This TEMPLATE adapter does the following:
      1. POST VIDEO_API_URL with JSON {"model", "prompt", "aspect_ratio", "duration", optional "image"}
         and the header Authorization: Bearer VIDEO_API_KEY.
      2. If the reply already contains a video URL, download it.
         Otherwise it looks for a job id / status URL and polls until the job finishes.
      3. Downloads the video and checks that the file really is a video.

    Video providers differ a lot (field names, polling endpoints, auth headers). Verify this
    against your provider's documentation and edit this function where it differs.
    Returns the video bytes. Raises ToonError if no valid video was produced.
    """
    api_key = get_secret("VIDEO_API_KEY")
    api_url = get_secret("VIDEO_API_URL")
    model = get_secret("VIDEO_MODEL")
    if not (api_key and api_url):
        raise ToonError("No video provider is configured. See the Settings page.")
    if not api_url.lower().startswith(("https://", "http://")):
        raise ToonError("VIDEO_API_URL must start with https://")

    def status(message):
        if on_status:
            on_status(message)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt[:MAX_PROMPT], "aspect_ratio": aspect, "duration": length}
    if model:
        payload["model"] = model
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload["image"] = f"data:{image_mime};base64,{encoded}"

    status("Sending your request to the video provider…")
    response = safe_request("POST", api_url, secret_values=[api_key], what="video provider",
                            headers=headers, json=payload, timeout=90)
    try:
        data = response.json()
    except ValueError:
        raise ToonError("The video provider did not return JSON. Its format may need a custom adapter.") from None

    video_url = find_video_url(data)
    if not video_url:
        job_id = find_first(data, ["id", "job_id", "task_id", "request_id"])
        status_url = find_first(data, ["status_url", "polling_url", "poll_url"])
        if not status_url and job_id:
            status_url = api_url.rstrip("/") + "/" + str(job_id)
        if not status_url:
            raise ToonError(
                "The provider's reply had neither a video URL nor a job id. "
                "Edit generate_video_with_provider in app.py to match your provider. "
                "Reply started with: " + scrub(json.dumps(data)[:200], [api_key])
            )
        started = time.time()
        while True:
            if time.time() - started > VIDEO_TIMEOUT_SECONDS:
                raise ToonError("The video job did not finish in time (5 minutes). Try again later.")
            time.sleep(VIDEO_POLL_SECONDS)
            poll = safe_request("GET", status_url, secret_values=[api_key], what="video provider",
                                headers=headers, timeout=30)
            try:
                poll_data = poll.json()
            except ValueError:
                raise ToonError("The provider's status reply was not JSON.") from None
            video_url = find_video_url(poll_data)
            state = str(find_first(poll_data, ["status", "state"]) or "").lower()
            if video_url:
                break
            if state in ("failed", "error", "cancelled", "canceled", "rejected"):
                raise ToonError(f"The video provider reported that the job {state}.")
            status(f"Still working… provider status: {state or 'unknown'} "
                   f"({int(time.time() - started)}s)")

    status("Downloading the finished video…")
    download = safe_request("GET", video_url, secret_values=[api_key], what="video download", timeout=180)
    video_bytes = download.content
    if len(video_bytes) > MAX_DOWNLOAD_BYTES:
        raise ToonError("The returned video file is too large.")
    if not looks_like_video(video_bytes):
        raise ToonError("The provider returned a file that is not a valid video, so nothing was saved.")
    return video_bytes


# =============================================================================
# 7. PAGES
# =============================================================================


def render_hero(title, text):
    """Gradient banner at the top of a page (static text only)."""
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{text}</p></div>", unsafe_allow_html=True)


def groq_warning():
    """Reusable warning when the Groq key is missing."""
    if not get_secret("GROQ_API_KEY"):
        st.warning(
            "🔑 **GROQ_API_KEY is not set yet.** Add it under Streamlit Cloud → your app → Settings → "
            "Secrets, then restart the app. The Settings page shows exactly how."
        )


# ------------------------------- Dashboard ----------------------------------


def start_story_from_idea():
    """Button callback: copy the dashboard idea to the Story Generator and open it."""
    idea = st.session_state.get("dash_idea", "").strip()
    if idea:
        st.session_state["story_idea"] = idea[:MAX_IDEA]
    st.session_state["nav"] = "Story Generator"


def page_dashboard():
    work = st.session_state.work
    projects = st.session_state.projects
    render_hero(APP_TITLE, "Turn one idea into a story, characters, scenes, and prompts for image and video tools.")
    groq_warning()

    st.markdown("### 💡 Start with a cartoon idea")
    st.text_area("Your cartoon idea", key="dash_idea", max_chars=MAX_IDEA, height=100,
                 placeholder="A shy robot who wants to learn how to dance…")
    st.button("✨ Create a story from this idea", on_click=start_story_from_idea, key="dash_go_story")

    st.markdown("### ⚡ Quick actions")
    actions = [
        ("🎭 New character", "Character Studio"),
        ("🎬 Open scenes", "Scene Builder"),
        ("🖼️ Image Studio", "Image Studio"),
        ("🎞️ Video Studio", "Video Studio"),
        ("📁 My Projects", "My Projects"),
        ("⚙️ Settings", "Settings"),
    ]
    cols = st.columns(3)
    for i, (label, page) in enumerate(actions):
        cols[i % 3].button(label, on_click=go_to, args=(page,), key=f"qa_{page}")

    st.markdown("### 📊 Your studio")
    saved_stories = sum(1 for p in projects.values() if p.get("story"))
    if work.get("story") and work["id"] not in projects:
        saved_stories += 1
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stories", saved_stories)
    m2.metric("Characters (current project)", len(work["characters"]))
    m3.metric("Scenes (current project)", len(work["scenes"]))
    m4.metric("Saved projects", len(projects))

    st.markdown("### 🕘 Recent projects")
    if not projects:
        st.info("No saved projects yet. Create a story, then click **Save project** in the sidebar.")
    else:
        recent = sorted(projects.values(), key=lambda p: p.get("updated", ""), reverse=True)[:3]
        for project in recent:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                title = project["story"]["title"] if project.get("story") else "No story yet"
                c1.write(f"**{project['name']}**  \n{title} · updated {project.get('updated', '')}")
                c2.button("Open", on_click=cb_load_project, args=(project["id"],), key=f"recent_{project['id']}")

    with st.expander("🚀 First time here? Start in 4 steps", expanded=not work.get("story")):
        st.markdown(
            "1. **Story Generator** — type an idea, choose the options, and click *Generate story*.\n"
            "2. **Character Studio** — design your heroes and get image prompts for them.\n"
            "3. **Scene Builder** — edit each scene's prompts, or regenerate a single scene.\n"
            "4. **Image / Video Studio** — generate pictures and clips **if** you configured a provider; "
            "otherwise copy the prompts into any image or video tool you like.\n\n"
            "Save your work in **My Projects** and download the JSON file — session data disappears when the app restarts."
        )


# ---------------------------- Story Generator -------------------------------


def current_story_params():
    """Collect and validate the story form values."""
    ss = st.session_state
    idea = check_text("Story idea", ss["story_idea"], MAX_IDEA)
    if len(idea) < 10:
        raise ToonError("Please describe your idea in a little more detail (at least 10 characters).")
    return {
        "idea": idea,
        "genre": ss["sg_genre"],
        "audience": ss["sg_audience"],
        "duration": ss["sg_duration"],
        "language": ss["sg_language"],
        "style": ss["sg_style"],
        "tone": ss["sg_tone"],
    }


def run_story_generation(fresh_take):
    """Validate, call Groq, and store the new story and scenes in the project."""
    try:
        params = current_story_params()
        with st.spinner("Writing your cartoon story… this can take 10–30 seconds."):
            story, scenes = generate_story(params, fresh_take=fresh_take)
    except ToonError as exc:
        st.error(str(exc))
        return
    work = st.session_state.work
    work["story"] = story
    work["scenes"] = scenes
    work["idea"] = params["idea"]
    if work["name"] == "Untitled Project":
        work["name"] = story["title"][:60]
    work["updated"] = now_str()
    st.success("Story generated! Scroll down to read it, or open Scene Builder to edit the scenes.")


def render_story(story, scenes):
    """Show the story in a readable layout."""
    st.markdown(f"## {story['title']}")
    st.write(story["summary"])
    st.caption(
        f"{story['genre']} · {story['audience']} · {story['duration']}s · {story['language']} · "
        f"{story['style']} · {story['tone']}"
    )
    st.markdown("#### 🎭 Characters")
    cols = st.columns(2)
    for i, ch in enumerate(story["characters"]):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{ch['name']}**")
            st.write(ch["description"])
    st.markdown("#### 🧩 Story structure")
    b, m, e = st.columns(3)
    b.markdown("**Beginning**")
    b.write(story["beginning"])
    m.markdown("**Middle**")
    m.write(story["middle"])
    e.markdown("**Ending**")
    e.write(story["ending"])
    st.markdown("#### 🎬 Scene-by-scene breakdown")
    for sc in scenes:
        with st.expander(f"Scene {sc['number']} · {sc['duration']}s · {sc['location']}"):
            st.write(f"**Characters:** {sc['characters']}")
            st.write(f"**Action & expressions:** {sc['action']}")
            st.write(f"**Camera:** {sc['camera']}")
            st.write(f"**Lighting & background:** {sc['lighting']}")
            st.write(f"**Narration:** {sc['narration'] or '—'}")
            st.write("**Dialogue:**")
            st.text(sc["dialogue"] or "—")
            st.write(f"**Visual description:** {sc['visual']}")
            st.write("**Image prompt:**")
            st.code(sc["image_prompt"], language=None, wrap_lines=True)
            st.write("**Video-animation prompt:**")
            st.code(sc["video_prompt"], language=None, wrap_lines=True)


def page_story():
    work = st.session_state.work
    render_hero("📖 Story Generator", "Describe your idea and let the AI write the story, characters, scenes and prompts.")
    groq_warning()

    st.text_area("Your story idea", key="story_idea", max_chars=MAX_IDEA, height=120,
                 placeholder="A little dragon who is afraid of fire learns to be brave…")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Genre", GENRES, key="sg_genre")
    c2.selectbox("Audience", AUDIENCES, key="sg_audience")
    c3.selectbox("Duration", DURATIONS, key="sg_duration", format_func=lambda d: f"{d} seconds")
    c4, c5, c6 = st.columns(3)
    c4.selectbox("Language", LANGUAGES, key="sg_language")
    c5.selectbox("Animation style", STYLES, key="sg_style")
    c6.selectbox("Tone", TONES, key="sg_tone")

    b1, b2, b3 = st.columns(3)
    generate = b1.button("✨ Generate story", key="sg_generate")
    regenerate = b2.button("🔄 Regenerate", key="sg_regen", disabled=not work.get("story"),
                           help="Create a different version using the same settings.")
    if generate:
        run_story_generation(fresh_take=False)
    elif regenerate:
        run_story_generation(fresh_take=True)
    st.caption("Generating or regenerating replaces the current story and its scenes (including edits).")

    work = st.session_state.work
    if work.get("story"):
        with b3:
            copy_box("Copy story", story_to_text(work["story"], work["scenes"]), "story_all")
        st.divider()
        render_story(work["story"], work["scenes"])
    else:
        st.info("Your story will appear here.")


# ---------------------------- Character Studio ------------------------------


def delete_character(char_id):
    work = st.session_state.work
    work["characters"] = [c for c in work["characters"] if c["id"] != char_id]


def import_story_characters():
    """Copy the characters from the story into the character list (as drafts)."""
    work = st.session_state.work
    existing = {c["name"].lower() for c in work["characters"]}
    added = 0
    for ch in (work.get("story") or {}).get("characters", []):
        if ch["name"].lower() in existing:
            continue
        work["characters"].append({
            "id": new_id(), "name": ch["name"], "species": "", "age": "", "personality": "",
            "appearance": ch["description"], "clothing": "", "colors": "", "abilities": "", "role": "",
            "style": (work["story"] or {}).get("style", STYLES[0]),
            "profile": ch["description"], "visual_description": ch["description"],
            "image_prompt": "", "created": now_str(),
        })
        added += 1
    flash("success" if added else "info", f"Added {added} character(s) from the story." if added else "No new story characters to add.")


def page_characters():
    work = st.session_state.work
    render_hero("🎭 Character Studio", "Design a character. The AI writes the profile and an image prompt you can use in any image tool.")
    groq_warning()
    show_flash()

    c1, c2, c3 = st.columns(3)
    c1.text_input("Character name", key="ch_name", max_chars=80)
    c2.text_input("Species or type", key="ch_species", max_chars=80, placeholder="Fox, robot, dragon…")
    c3.selectbox("Age category", AGE_CATEGORIES, key="ch_age")
    d1, d2 = st.columns(2)
    d1.text_area("Personality", key="ch_personality", max_chars=MAX_SHORT, height=90)
    d2.text_area("Appearance", key="ch_appearance", max_chars=MAX_SHORT, height=90)
    e1, e2, e3 = st.columns(3)
    e1.text_input("Clothing", key="ch_clothing", max_chars=MAX_SHORT)
    e2.text_input("Colors", key="ch_colors", max_chars=MAX_SHORT)
    e3.text_input("Special abilities", key="ch_abilities", max_chars=MAX_SHORT)
    f1, f2 = st.columns(2)
    f1.text_input("Role in the story", key="ch_role", max_chars=MAX_SHORT)
    f2.selectbox("Art style", STYLES, key="ch_style")

    b1, b2 = st.columns(2)
    if b1.button("✨ Generate character profile", key="ch_generate"):
        ss = st.session_state
        try:
            name = check_text("Character name", ss["ch_name"], 80)
            species = check_text("Species or type", ss["ch_species"], 80)
            fields = {
                "personality": check_text("Personality", ss["ch_personality"], MAX_SHORT, required=False),
                "appearance": check_text("Appearance", ss["ch_appearance"], MAX_SHORT, required=False),
                "clothing": check_text("Clothing", ss["ch_clothing"], MAX_SHORT, required=False),
                "colors": check_text("Colors", ss["ch_colors"], MAX_SHORT, required=False),
                "abilities": check_text("Special abilities", ss["ch_abilities"], MAX_SHORT, required=False),
                "role": check_text("Role in the story", ss["ch_role"], MAX_SHORT, required=False),
            }
            inputs = {
                "style": ss["ch_style"],
                "details": [
                    ("Name", name), ("Species / type", species), ("Age category", ss["ch_age"]),
                    ("Personality", fields["personality"]), ("Appearance", fields["appearance"]),
                    ("Clothing", fields["clothing"]), ("Colors", fields["colors"]),
                    ("Special abilities", fields["abilities"]), ("Role in the story", fields["role"]),
                ],
            }
            with st.spinner("Designing your character…"):
                result = generate_character_profile(inputs)
            work["characters"].append({
                "id": new_id(), "name": name, "species": species, "age": ss["ch_age"], **fields,
                "style": ss["ch_style"], **result, "created": now_str(),
            })
            st.success(f"{name} was added to your characters below.")
        except ToonError as exc:
            st.error(str(exc))
    b2.button("📥 Import characters from story", on_click=import_story_characters, key="ch_import",
              disabled=not work.get("story"))

    st.caption("Image prompts are for an external image generator. Groq only writes text — it does not create images.")
    st.divider()
    st.markdown(f"### Your characters ({len(work['characters'])})")
    if not work["characters"]:
        st.info("No characters yet. Generate one above.")
    for ch in work["characters"]:
        title = f"{ch['name']} — {ch['species']}" if ch.get("species") else ch["name"]
        with st.expander(title):
            st.write(ch["profile"])
            if ch.get("visual_description"):
                st.markdown("**Visual description**")
                st.write(ch["visual_description"])
            if ch.get("image_prompt"):
                st.markdown("**Image prompt**")
                st.code(ch["image_prompt"], language=None, wrap_lines=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                copy_box("Copy description", ch["profile"], f"chd_{ch['id']}")
            with k2:
                copy_box("Copy image prompt", ch.get("image_prompt", ""), f"chp_{ch['id']}")
            k3.button("🖼️ Use in Image Studio", on_click=go_to, args=("Image Studio",),
                      kwargs={"img_prompt": ch.get("image_prompt", "")[:MAX_PROMPT]},
                      key=f"chi_{ch['id']}", disabled=not ch.get("image_prompt"))
            k4.button("🗑️ Delete", on_click=delete_character, args=(ch["id"],), key=f"chx_{ch['id']}")


# ------------------------------ Scene Builder -------------------------------


def sync_scene_field(sid, field, key):
    """on_change callback: copy an edited widget value back into the scene data."""
    for sc in st.session_state.work["scenes"]:
        if sc["sid"] == sid:
            value = st.session_state.get(key, sc.get(field))
            sc[field] = int(value) if field == "duration" else value
            break


def scene_input(sc, field, label, area=False, height=90):
    """Draw an editable scene field that saves its value automatically."""
    key = f"sc_{sc['sid']}_{sc['ver']}_{field}"
    kwargs = {"label": label, "value": sc.get(field, ""), "key": key,
              "max_chars": FIELD_LIMITS[field], "on_change": sync_scene_field, "args": (sc["sid"], field, key)}
    if area:
        st.text_area(height=height, **kwargs)
    else:
        st.text_input(**kwargs)


def renumber_scenes(scenes):
    for i, sc in enumerate(scenes):
        sc["number"] = i + 1


def blank_scene(number):
    scene = normalize_scene({"location": "New location", "action_expression": "Describe what happens"}, number, 5)
    scene["image_prompt"] = ""
    scene["video_prompt"] = ""
    return scene


def page_scenes():
    work = st.session_state.work
    scenes = work["scenes"]
    render_hero("🎬 Scene Builder", "Edit every scene's details and prompts, copy them, or regenerate a single scene.")
    groq_warning()
    show_flash()

    if not work.get("story") and not scenes:
        st.info("You need a story first. Generate one, then come back to edit its scenes.")
        st.button("📖 Go to Story Generator", on_click=go_to, args=("Story Generator",), key="sc_goto_story")
        return

    story = work.get("story")
    m1, m2 = st.columns(2)
    m1.metric("Scenes", len(scenes))
    m2.metric("Total duration", f"{sum(s['duration'] for s in scenes)} s")

    t1, t2, t3 = st.columns([1, 1, 1])
    if t1.button("➕ Add blank scene", key="sc_add"):
        scenes.append(blank_scene(len(scenes) + 1))
        st.rerun()
    count = t2.selectbox("Scenes for a full rebuild", list(range(3, 11)), index=2, key="sc_rebuild_count")
    if t3.button("🔁 Rebuild all scenes with AI", key="sc_rebuild", disabled=not story,
                 help="Replaces all scenes (and your edits) with a fresh set based on the story."):
        try:
            with st.spinner("Rebuilding all scenes…"):
                work["scenes"] = generate_scenes(story, count)
            st.rerun()
        except ToonError as exc:
            st.error(str(exc))

    st.markdown("### 🎞️ Storyboard")
    for idx, sc in enumerate(scenes):
        with st.expander(f"Scene {sc['number']} · {sc['duration']}s · {sc['location']}", expanded=(idx == 0)):
            a1, a2 = st.columns([1, 3])
            key = f"sc_{sc['sid']}_{sc['ver']}_duration"
            a1.number_input("Duration (s)", min_value=1, max_value=120, value=int(sc["duration"]), step=1,
                            key=key, on_change=sync_scene_field, args=(sc["sid"], "duration", key))
            with a2:
                scene_input(sc, "location", "Location")
            scene_input(sc, "characters", "Characters in this scene")
            scene_input(sc, "action", "Actions and expressions", area=True)
            b1, b2 = st.columns(2)
            with b1:
                scene_input(sc, "camera", "Camera angle / movement")
            with b2:
                scene_input(sc, "lighting", "Lighting and background")
            scene_input(sc, "narration", "Narration", area=True, height=70)
            scene_input(sc, "dialogue", "Dialogue", area=True, height=70)
            scene_input(sc, "visual", "Visual description", area=True, height=70)
            scene_input(sc, "image_prompt", "Image prompt", area=True, height=110)
            scene_input(sc, "video_prompt", "Video-animation prompt", area=True, height=110)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                copy_box("Copy image prompt", sc["image_prompt"], f"sci_{sc['sid']}")
            with c2:
                copy_box("Copy video prompt", sc["video_prompt"], f"scv_{sc['sid']}")
            c3.button("🖼️ Send to Image Studio", on_click=go_to, args=("Image Studio",),
                      kwargs={"img_prompt": sc["image_prompt"][:MAX_PROMPT]}, key=f"scti_{sc['sid']}",
                      disabled=not sc["image_prompt"])
            c4.button("🎞️ Send to Video Studio", on_click=go_to, args=("Video Studio",),
                      kwargs={"vid_prompt": sc["video_prompt"][:MAX_PROMPT]}, key=f"sctv_{sc['sid']}",
                      disabled=not sc["video_prompt"])

            instruction = st.text_input("Optional: how should this scene change?", max_chars=MAX_SHORT,
                                        key=f"sc_{sc['sid']}_{sc['ver']}_instr",
                                        placeholder="Make it funnier, move it to the beach…")
            r1, r2 = st.columns(2)
            if r1.button("🔄 Regenerate this scene", key=f"scr_{sc['sid']}", disabled=not story):
                try:
                    with st.spinner(f"Regenerating scene {sc['number']}…"):
                        scenes[idx] = regenerate_scene(story, scenes, idx, instruction.strip())
                    st.rerun()
                except ToonError as exc:
                    st.error(str(exc))
            if r2.button("🗑️ Delete this scene", key=f"scd_{sc['sid']}"):
                scenes.pop(idx)
                renumber_scenes(scenes)
                st.rerun()


# ------------------------------ Image Studio --------------------------------


def image_prompt_sources():
    """Prompts available to load into the Image Studio."""
    work = st.session_state.work
    sources = {"— Write my own prompt —": None}
    for sc in work["scenes"]:
        sources[f"Scene {sc['number']}: {sc['location'][:40]}"] = sc["image_prompt"]
    for ch in work["characters"]:
        if ch.get("image_prompt"):
            sources[f"Character: {ch['name']}"] = ch["image_prompt"]
    return sources


def cb_load_image_prompt():
    prompt = image_prompt_sources().get(st.session_state.get("img_source"))
    if prompt:
        st.session_state["img_prompt"] = prompt[:MAX_PROMPT]


def page_images():
    work = st.session_state.work
    render_hero("🖼️ Image Studio", "Generate pictures with an image provider you configure. Groq only writes the prompts.")

    sources = list(image_prompt_sources().keys())
    if st.session_state.get("img_source") not in sources:
        st.session_state["img_source"] = sources[0]
    st.selectbox("Load a prompt from your project", sources, key="img_source", on_change=cb_load_image_prompt)
    st.text_area("Image prompt", key="img_prompt", max_chars=MAX_PROMPT, height=140,
                 placeholder="A cheerful fox wearing a red scarf, standing in a snowy forest…")
    c1, c2 = st.columns(2)
    c1.selectbox("Aspect ratio", list(IMAGE_ASPECTS.keys()), key="img_aspect")
    c2.selectbox("Animation style", STYLES, key="img_style")

    ready = image_provider_ready()
    if not ready:
        st.warning(
            "**No image provider is configured, so no image can be generated.**\n\n"
            "Add `IMAGE_API_KEY` and `IMAGE_API_URL` (and usually `IMAGE_MODEL`) to your Streamlit Secrets. "
            "Open **Settings** for instructions. Meanwhile, copy your prompt below and use it in any image tool."
        )
    copy_box("Copy image prompt", st.session_state["img_prompt"], "img_main")

    if st.button("🎨 Generate image", key="img_generate", disabled=not ready):
        try:
            prompt = check_text("Image prompt", st.session_state["img_prompt"], MAX_PROMPT)
            with st.spinner("Contacting your image provider…"):
                data, fmt, width, height = generate_image_with_provider(
                    prompt, st.session_state["img_aspect"], st.session_state["img_style"])
            image_id = new_id()
            st.session_state.image_cache[image_id] = {
                "bytes": data, "format": fmt, "width": width, "height": height, "prompt": prompt}
            work["assets"]["images"].append({
                "id": image_id, "prompt": prompt, "aspect": IMAGE_ASPECTS[st.session_state["img_aspect"]],
                "style": st.session_state["img_style"], "model": get_secret("IMAGE_MODEL") or "unspecified",
                "created": now_str(), "note": "Image data lives in this session only. Download it to keep it.",
            })
            st.success("Image generated by your image provider.")
        except ToonError as exc:
            st.error(str(exc))

    cache = st.session_state.image_cache
    if cache:
        st.divider()
        st.markdown("### 🖼️ Generated images (this session)")
        for image_id, item in list(reversed(list(cache.items())))[:6]:
            with st.container(border=True):
                st.image(item["bytes"], caption=f"{item['width']}×{item['height']} · {item['prompt'][:90]}")
                ext = item["format"].lower().replace("jpeg", "jpg")
                st.download_button("⬇️ Download image", data=item["bytes"], file_name=f"toongenius_{image_id}.{ext}",
                                   mime=f"image/{'jpeg' if ext == 'jpg' else ext}", key=f"dl_img_{image_id}")


# ------------------------------ Video Studio --------------------------------


def video_prompt_sources():
    work = st.session_state.work
    sources = {"— Write my own prompt —": None}
    for sc in work["scenes"]:
        sources[f"Scene {sc['number']}: {sc['location'][:40]}"] = sc["video_prompt"]
    return sources


def cb_load_video_prompt():
    prompt = video_prompt_sources().get(st.session_state.get("vid_source"))
    if prompt:
        st.session_state["vid_prompt"] = prompt[:MAX_PROMPT]


def page_video():
    work = st.session_state.work
    render_hero("🎞️ Video Studio", "Request a short animated clip from a video provider you configure.")

    sources = list(video_prompt_sources().keys())
    if st.session_state.get("vid_source") not in sources:
        st.session_state["vid_source"] = sources[0]
    st.selectbox("Select a story scene", sources, key="vid_source", on_change=cb_load_video_prompt)
    st.text_area("Video prompt", key="vid_prompt", max_chars=MAX_PROMPT, height=140,
                 placeholder="The little fox hops through falling snow while the camera slowly follows…")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Aspect ratio", VIDEO_ASPECTS, key="vid_aspect")
    c2.selectbox("Clip length", VIDEO_LENGTHS, key="vid_length", format_func=lambda s: f"{s} seconds")
    c3.radio("Mode", ["Text-to-video", "Image-to-video"], key="vid_mode")

    image_bytes, image_mime = None, "image/png"
    if st.session_state["vid_mode"] == "Image-to-video":
        st.caption("Not every provider supports image-to-video. Your adapter must accept an image.")
        cache = st.session_state.image_cache
        options = ["— None —"] + [f"{iid}: {item['prompt'][:50]}" for iid, item in cache.items()]
        if st.session_state.get("vid_image_choice") not in options:
            st.session_state["vid_image_choice"] = options[0]
        choice = st.selectbox("Use a generated image", options, key="vid_image_choice")
        upload = st.file_uploader("…or upload a starting image", type=["png", "jpg", "jpeg", "webp"], key="vid_upload")
        if upload is not None:
            raw = upload.getvalue()
            if len(raw) > 5 * 1024 * 1024:
                st.error("Please upload an image smaller than 5 MB.")
            else:
                try:
                    fmt, _, _ = validate_image_bytes(raw)
                    image_bytes, image_mime = raw, f"image/{fmt.lower()}"
                except ToonError as exc:
                    st.error(str(exc))
        elif choice != "— None —":
            item = cache[choice.split(":")[0]]
            image_bytes, image_mime = item["bytes"], f"image/{item['format'].lower()}"

    ready = video_provider_ready()
    if not ready:
        st.warning(
            "**No video provider is configured, so no video can be generated.**\n\n"
            "Add `VIDEO_API_KEY` and `VIDEO_API_URL` (and usually `VIDEO_MODEL`) to your Streamlit Secrets. "
            "Open **Settings** for instructions. Meanwhile, copy your prompt below and use it in any video tool."
        )
    copy_box("Copy video prompt", st.session_state["vid_prompt"], "vid_main")

    if st.button("🎬 Generate video", key="vid_generate", disabled=not ready):
        try:
            prompt = check_text("Video prompt", st.session_state["vid_prompt"], MAX_PROMPT)
            if st.session_state["vid_mode"] == "Image-to-video" and not image_bytes:
                raise ToonError("Choose or upload a starting image for image-to-video.")
            with st.status("Generating video…", expanded=True) as status_box:
                try:
                    video = generate_video_with_provider(
                        prompt, st.session_state["vid_aspect"], st.session_state["vid_length"],
                        image_bytes if st.session_state["vid_mode"] == "Image-to-video" else None,
                        image_mime, on_status=status_box.write)
                except ToonError:
                    status_box.update(label="Video generation failed", state="error")
                    raise
                status_box.update(label="Video ready", state="complete")
            video_id = new_id()
            st.session_state.video_cache[video_id] = {"bytes": video, "prompt": prompt}
            work["assets"]["videos"].append({
                "id": video_id, "prompt": prompt, "aspect": st.session_state["vid_aspect"],
                "length": st.session_state["vid_length"], "model": get_secret("VIDEO_MODEL") or "unspecified",
                "created": now_str(), "note": "Video data lives in this session only. Download it to keep it.",
            })
        except ToonError as exc:
            st.error(str(exc))

    cache = st.session_state.video_cache
    if cache:
        st.divider()
        st.markdown("### 🎞️ Generated videos (this session)")
        for video_id, item in list(reversed(list(cache.items())))[:3]:
            with st.container(border=True):
                st.video(item["bytes"])
                st.caption(item["prompt"][:120])
                st.download_button("⬇️ Download video", data=item["bytes"], file_name=f"toongenius_{video_id}.mp4",
                                   mime="video/mp4", key=f"dl_vid_{video_id}")


# ------------------------------- My Projects --------------------------------


def export_project_json(project):
    """Serialize a project as pretty JSON text."""
    return json.dumps({"format": "toongenius-project", "version": 1, "project": project},
                      ensure_ascii=False, indent=2)


def project_from_import(data):
    """
    Validate an imported JSON object and build a clean project from it.
    Raises ToonError if the file is not a usable ToonGenius project.
    """
    if not isinstance(data, dict):
        raise ToonError("This file is not a ToonGenius project (expected a JSON object).")
    raw = data.get("project") if isinstance(data.get("project"), dict) else data
    if not any(k in raw for k in ("story", "characters", "scenes", "name")):
        raise ToonError("This JSON file does not look like a ToonGenius project.")
    project = new_project(clean_str(raw.get("name"), 80, "Imported Project"))
    project["idea"] = clean_str(raw.get("idea"), MAX_IDEA)
    project["created"] = clean_str(raw.get("created"), 40, project["created"])
    project["story"] = sanitize_story(raw.get("story"))
    for item in raw.get("characters") or []:
        if isinstance(item, dict) and clean_str(item.get("name")):
            project["characters"].append({
                "id": new_id(),
                "name": clean_str(item.get("name"), 80),
                "species": clean_str(item.get("species"), 80),
                "age": clean_str(item.get("age"), 40),
                "personality": clean_str(item.get("personality"), MAX_SHORT),
                "appearance": clean_str(item.get("appearance"), MAX_SHORT),
                "clothing": clean_str(item.get("clothing"), MAX_SHORT),
                "colors": clean_str(item.get("colors"), MAX_SHORT),
                "abilities": clean_str(item.get("abilities"), MAX_SHORT),
                "role": clean_str(item.get("role"), MAX_SHORT),
                "style": clean_str(item.get("style"), 40, STYLES[0]),
                "profile": clean_str(item.get("profile"), 3000),
                "visual_description": clean_str(item.get("visual_description"), 1500),
                "image_prompt": clean_str(item.get("image_prompt"), MAX_PROMPT),
                "created": clean_str(item.get("created"), 40, now_str()),
            })
    raw_scenes = raw.get("scenes") or []
    if isinstance(raw_scenes, list) and raw_scenes:
        try:
            project["scenes"] = normalize_scenes(raw_scenes, 30)
        except ValueError:
            raise ToonError("The scenes in this file are not valid.") from None
    assets = raw.get("assets") if isinstance(raw.get("assets"), dict) else {}
    for kind in ("images", "videos"):
        refs = assets.get(kind) if isinstance(assets.get(kind), list) else []
        project["assets"][kind] = [
            {k: clean_str(v, 500) for k, v in ref.items() if isinstance(k, str)}
            for ref in refs[:50] if isinstance(ref, dict)
        ]
    return project


def rename_project(project_id, key):
    new_name = st.session_state.get(key, "").strip()
    if not new_name:
        flash("error", "A project name cannot be empty.")
        return
    new_name = new_name[:80]
    project = st.session_state.projects.get(project_id)
    if project:
        project["name"] = new_name
        project["updated"] = now_str()
    if st.session_state.work["id"] == project_id:
        st.session_state.work["name"] = new_name
    flash("success", "Project renamed.")


def delete_project(project_id):
    project = st.session_state.projects.pop(project_id, None)
    if project:
        flash("success", f"Deleted project “{project['name']}”.")


def page_projects():
    work = st.session_state.work
    projects = st.session_state.projects
    render_hero("📁 My Projects", "Save, rename, load, delete, export and import your cartoon projects.")
    st.warning(
        "**Important:** projects are kept in this browser session only. They can disappear when the session ends "
        "or the app restarts. There is no permanent cloud storage — **download the JSON file** to keep your work safe."
    )
    show_flash()

    st.markdown("### Current project")
    with st.container(border=True):
        name_key = f"cur_name_{work['id']}"
        st.text_input("Project name", value=work["name"], key=name_key, max_chars=80)
        st.caption(
            f"Story: {'yes' if work.get('story') else 'no'} · Characters: {len(work['characters'])} · "
            f"Scenes: {len(work['scenes'])} · Images: {len(work['assets']['images'])} · "
            f"Videos: {len(work['assets']['videos'])}"
        )
        a1, a2, a3 = st.columns(3)
        if a1.button("💾 Save project", key="pj_save"):
            new_name = st.session_state.get(name_key, "").strip()
            if not new_name:
                st.error("A project name cannot be empty.")
            else:
                work["name"] = new_name[:80]
                save_current_project()
                st.success("Saved for this session. Download the JSON to keep it permanently.")
        a2.download_button("⬇️ Download JSON", data=export_project_json(work),
                           file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', work['name']) or 'project'}.json",
                           mime="application/json", key="pj_download_current")
        a3.button("🆕 New project", on_click=cb_new_project, key="pj_new")

    st.markdown("### 📥 Import a project")
    uploaded = st.file_uploader("Upload a ToonGenius project JSON", type=["json"],
                                key=f"pj_upload_{st.session_state.import_counter}")
    if uploaded is not None and st.button("Import this file", key="pj_import"):
        try:
            if uploaded.size > 5 * 1024 * 1024:
                raise ToonError("The file is too large (over 5 MB).")
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ToonError("That file is not valid JSON.") from None
            project = project_from_import(data)
            projects[project["id"]] = project
            st.session_state.import_counter += 1
            flash("success", f"Imported “{project['name']}”. Click Load to open it.")
            st.rerun()
        except ToonError as exc:
            st.error(str(exc))

    st.markdown(f"### 🗂️ Saved projects ({len(projects)})")
    if not projects:
        st.info("Nothing saved yet. Use **Save project** above.")
    for project in sorted(projects.values(), key=lambda p: p.get("updated", ""), reverse=True):
        pid = project["id"]
        with st.expander(f"{project['name']} · updated {project.get('updated', '')}"):
            story = project.get("story")
            st.write(f"**Story:** {story['title'] if story else 'none'}")
            st.caption(f"Characters: {len(project['characters'])} · Scenes: {len(project['scenes'])}")
            rn_key = f"rn_{pid}"
            st.text_input("Rename", value=project["name"], key=rn_key, max_chars=80)
            b1, b2, b3, b4 = st.columns(4)
            b1.button("✏️ Rename", on_click=rename_project, args=(pid, rn_key), key=f"pj_rn_{pid}")
            b2.button("📂 Load", on_click=cb_load_project, args=(pid,), key=f"pj_load_{pid}")
            b3.download_button("⬇️ JSON", data=export_project_json(project), file_name=f"toongenius_{pid}.json",
                               mime="application/json", key=f"pj_dl_{pid}")
            confirm = st.checkbox("Confirm delete", key=f"pj_cf_{pid}")
            b4.button("🗑️ Delete", on_click=delete_project, args=(pid,), key=f"pj_del_{pid}", disabled=not confirm)


# --------------------------------- Settings ---------------------------------

SECRETS_EXAMPLE = """# Required for stories, characters and scenes
GROQ_API_KEY = "paste-your-groq-key-here"

# Optional: change the Groq model (default: llama-3.3-70b-versatile)
GROQ_MODEL = "llama-3.3-70b-versatile"

# Optional: image generation provider (placeholders - use values from YOUR provider's docs)
IMAGE_API_KEY = "your-image-provider-key"
IMAGE_API_URL = "https://your-image-provider.example.com/v1/images/generations"
IMAGE_MODEL = "your-image-model-name"

# Optional: video generation provider (placeholders - use values from YOUR provider's docs)
VIDEO_API_KEY = "your-video-provider-key"
VIDEO_API_URL = "https://your-video-provider.example.com/your-endpoint"
VIDEO_MODEL = "your-video-model-name"
"""


def page_settings():
    render_hero("⚙️ Settings", "Check which services are configured and learn how to set them up.")

    st.markdown("### 🔐 Configuration status")
    st.caption("Values are read from Streamlit Secrets (or environment variables). Full keys are never shown.")
    rows = [
        ("GROQ_API_KEY", "key", "Required — story, character and scene text generation."),
        ("GROQ_MODEL", "plain", f"Optional — default is {DEFAULT_GROQ_MODEL}."),
        ("IMAGE_API_KEY", "key", "Optional — image-generation provider key."),
        ("IMAGE_API_URL", "url", "Optional — image provider endpoint."),
        ("IMAGE_MODEL", "plain", "Optional — image model identifier."),
        ("VIDEO_API_KEY", "key", "Optional — video-generation provider key."),
        ("VIDEO_API_URL", "url", "Optional — video provider endpoint."),
        ("VIDEO_MODEL", "plain", "Optional — video model identifier."),
    ]
    for name, kind, description in rows:
        value = get_secret(name)
        shown = mask_secret(value) if kind == "key" else host_only(value) if kind == "url" else (value or "Not set")
        icon = "✅" if value else ("⚠️" if name == "GROQ_API_KEY" else "➖")
        c1, c2, c3 = st.columns([2, 2, 4])
        c1.markdown(f"{icon} `{name}`")
        c2.write(shown)
        c3.caption(description)

    st.markdown("### 🧪 Test Groq")
    if st.button("Run a quick Groq test", key="st_test"):
        try:
            with st.spinner("Contacting Groq…"):
                result = groq_chat_json("Reply with JSON only.", 'Return exactly {"ok": true} as JSON.',
                                        temperature=0, max_tokens=30)
            model = get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL
            st.success(f"Groq responded successfully using model “{model}”." if result else "Groq responded.")
        except ToonError as exc:
            st.error(str(exc))

    st.markdown("### 📝 How to add your secrets")
    st.markdown(
        "On Streamlit Community Cloud open your app → **⋮ / Manage app → Settings → Secrets**, paste the lines you "
        "need (TOML format), save, then reboot the app. For local testing you can use a "
        "`.streamlit/secrets.toml` file or environment variables. **Never upload keys to GitHub.**"
    )
    st.code(SECRETS_EXAMPLE, language="toml")

    st.markdown("### 🖼️ Image and 🎞️ video providers")
    st.markdown(
        "- **Groq is used only for text.** It does not generate images or videos.\n"
        "- Image and video generation need a **separate provider account**, which may cost money. "
        "The app never assumes they are free.\n"
        "- Providers use different request formats. `app.py` contains two clearly marked adapter functions: "
        "`generate_image_with_provider()` (OpenAI-style *images/generations* request) and "
        "`generate_video_with_provider()` (submit → poll → download template). "
        "**Check them against your provider's documentation and edit them if the format differs.**\n"
        "- If a provider is missing, fails, or returns an invalid file, the app shows an error and "
        "never displays a fake result."
    )
    st.markdown("### 🔒 Notes")
    st.markdown(
        "- Projects live in session state and disappear when the app restarts — export JSON to keep them.\n"
        "- If the default Groq model is unavailable to your account, set `GROQ_MODEL` to another model."
    )


# =============================================================================
# 8. SIDEBAR AND MAIN
# =============================================================================


def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='brand'>🎨 ToonGenius AI</div>", unsafe_allow_html=True)
        st.caption("Your Cartoon Creation Studio")
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed",
                 format_func=lambda p: f"{PAGE_ICONS[p]}  {p}")
        st.divider()
        st.caption("Current project")
        st.write(f"**{st.session_state.work['name']}**")
        st.button("💾 Save project", on_click=cb_save_project, key="sb_save")
        st.button("🆕 New project", on_click=cb_new_project, key="sb_new")
        st.divider()
        st.caption("Services")
        st.write(("✅" if get_secret("GROQ_API_KEY") else "⚠️") + " Groq (text)")
        st.write(("✅" if image_provider_ready() else "➖") + " Image provider")
        st.write(("✅" if video_provider_ready() else "➖") + " Video provider")


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()
    show_flash()
    pages = {
        "Dashboard": page_dashboard,
        "Story Generator": page_story,
        "Character Studio": page_characters,
        "Scene Builder": page_scenes,
        "Image Studio": page_images,
        "Video Studio": page_video,
        "My Projects": page_projects,
        "Settings": page_settings,
    }
    pages[st.session_state["nav"]]()


main()
