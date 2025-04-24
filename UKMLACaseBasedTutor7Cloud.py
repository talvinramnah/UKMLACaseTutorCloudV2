import streamlit as st
import openai
import time
import json
from supabase import create_client
from datetime import datetime, timezone
import traceback
import os
from pathlib import Path

# --- CONFIG ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
ASSISTANT_ID = st.secrets["OPENAI_ASSISTANT_ID"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Case file directory - using relative path for cloud deployment
CASE_FILES_DIR = Path(__file__).parent / "data" / "cases"

def ensure_case_files_exist():
    """Verify that all case files exist and are readable."""
    missing_files = []
    for ward_cases in CASES.values():
        for case in ward_cases:
            file_path = CASE_FILES_DIR / case["file"]
            if not file_path.is_file():
                missing_files.append(case["file"])
    
    if missing_files:
        st.error(f"Missing case files: {', '.join(missing_files)}")
        st.stop()

# Ward organization with exact file paths
CASES = {
    "Cardiology": [
        {
            "name": "Acute Coronary Syndrome",
            "file": "acute_coronary_syndrome.txt"
        },
        {
            "name": "Adult Advanced Life Support",
            "file": "adult_advanced_life_support.txt"
        },
        {
            "name": "Hypertension Management",
            "file": "hypertension_management.txt"
        }
    ],
    "Respiratory": [
        {
            "name": "Asthma",
            "file": "asthma.txt"
        },
        {
            "name": "Pneumothorax Management",
            "file": "pneumothorax_management.txt"
        },
        {
            "name": "COPD Management",
            "file": "copd_management.txt"
        }
    ],
    "ENT": [
        {
            "name": "Vestibular Neuronitis",
            "file": "vestibular_neuronitis.txt"
        },
        {
            "name": "Rinne's and Weber's Test",
            "file": "rinnes_and_webers_test.txt"
        },
        {
            "name": "Acute Otitis Media",
            "file": "acute_otitis_media.txt"
        }
    ]
}

# Verify case files exist at startup
ensure_case_files_exist()

def get_case_content(condition_name: str) -> str:
    """Get the content of a case file."""
    try:
        # Find the case file path
        for ward_cases in CASES.values():
            for case in ward_cases:
                if case["name"] == condition_name:
                    file_path = CASE_FILES_DIR / case["file"]
                    with open(file_path, 'r') as file:
                        return file.read().strip()
        raise FileNotFoundError(f"No case file found for condition: {condition_name}")
    except Exception as e:
        st.error(f"Error reading case file for {condition_name}: {str(e)}")
        return None

def get_ward_for_condition(condition_name: str) -> str:
    """Get the ward name for a given condition."""
    for ward, cases in CASES.items():
        if any(case["name"] == condition_name for case in cases):
            return ward
    return "Unknown"

# Create clients with error handling
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    client = openai.OpenAI(api_key=openai.api_key)
except Exception as e:
    st.error(f"Failed to initialize clients: {str(e)}")
    st.stop()

# --- AUTH HELPERS ---
def signup_user(email, password):
    try:
        # Get the site URL from Streamlit's environment
        site_url = st.get_option("server.baseUrlPath") or "http://localhost:8501"
        
        # If we're on Streamlit Cloud, construct the full URL
        if "STREAMLIT_SHARING_MODE" in os.environ:
            site_url = f"https://{os.environ['STREAMLIT_APP_URL']}"
        
        auth_options = {
            "email": email,
            "password": password,
            "options": {
                "emailRedirectTo": f"{site_url}/_stcore/authenticate"
            }
        }
        
        return supabase.auth.sign_up(auth_options)
    except Exception as e:
        st.error(f"Signup failed: {str(e)}")
        return None

def login_user(email, password):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if res and res.session:
            # Store tokens in session state
            st.session_state.access_token = res.session.access_token
            st.session_state.refresh_token = res.session.refresh_token
            
            # Set the session in Supabase client
            supabase.auth.set_session(
                res.session.access_token,
                res.session.refresh_token
            )
            
            # Store user info and current user ID
            st.session_state.user = res.user
            st.session_state.current_user_id = res.user.id
            
            return res
        return None
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def refresh_supabase_session():
    try:
        # Only attempt refresh if we have a refresh token
        if st.session_state.refresh_token:
            res = supabase.auth.refresh_session()
            if res and res.session:
                st.session_state.access_token = res.session.access_token
                st.session_state.refresh_token = res.session.refresh_token
                supabase.auth.set_session(
                    res.session.access_token,
                    res.session.refresh_token
                )
                return True
        return False
    except Exception as e:
        st.error(f"Session refresh failed: {str(e)}")
        return False

def ensure_supabase_session():
    try:
        # First check if we have tokens in session state
        if not st.session_state.access_token or not st.session_state.refresh_token:
            st.error("No tokens in session state")
            return False
            
        # Try to set the session with existing tokens
        supabase.auth.set_session(
            st.session_state.access_token,
            st.session_state.refresh_token
        )
        
        # Get and verify session
        session = supabase.auth.get_session()
        if not session:
            st.error("No session after setting tokens")
            return False
            
        # Store the current user ID in session state
        st.session_state.current_user_id = session.user.id
        return True
            
    except Exception as e:
        st.error(f"Session check failed: {str(e)}")
        return False

# --- SESSION STATE INIT ---
def init_session_state():
    required_states = {
        "user": None,
        "access_token": None,
        "refresh_token": None,
        "thread_id": None,
        "case_started": False,
        "chat_history": [],
        "condition": None,
        "error_message": None,
        "is_loading": False,
        "case_completed": False,
        "show_goodbye": False,
        "score": None,
        "feedback": None,
        "case_variation": None,
        "total_score": 0
    }
    
    for key, default_value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# --- ERROR HANDLING DECORATOR ---
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.session_state.error_message = str(e)
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or contact support if the issue persists.")
            return None
    return wrapper

def get_next_case_variation(condition_name: str) -> int:
    """Get the next unused case variation number for this user and condition."""
    try:
        # Query completed cases for this user and condition
        result = supabase.table("performance").select("case_variation").eq("user_id", st.session_state.user.id).eq("condition", condition_name).execute()
        
        if not result.data:
            return 1  # First case
            
        # Get all used variations
        used_variations = set(record.get('case_variation', 0) for record in result.data)
        
        # Find the next unused variation number
        variation = 1
        while variation in used_variations:
            variation += 1
            
        return variation
    except Exception as e:
        st.error(f"Error getting case variation: {str(e)}")
        return 1

# --- CASE START FUNCTION ---
@handle_errors
def start_case(condition_name: str):
    if not ensure_supabase_session():
        raise Exception("Authentication required. Please log in again.")
    
    if st.session_state.case_started:
        st.warning("A case is already in progress. Please complete it before starting a new one.")
        return

    # Get case content
    case_content = get_case_content(condition_name)
    if not case_content:
        st.error(f"Could not load case content for {condition_name}")
        return

    st.session_state.is_loading = True
    try:
        # Get next case variation
        case_variation = get_next_case_variation(condition_name)
        st.session_state.case_variation = case_variation

        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
        st.session_state.case_started = True
        st.session_state.condition = condition_name
        st.session_state.chat_history = []

        # initial prompt
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"""
GOAL: Start a UKMLA-style case on: {condition_name} (Variation {case_variation}).

PERSONA: You are a senior doctor training a medical student through a real-life ward case for the UKMLA.

CASE CONTENT:
{case_content}

INSTRUCTIONS:
- Present one case based on the case content provided above.
- Do not skip straight to diagnosis or treatment. Walk through it step-by-step.
- Ask what investigations they'd like, then provide results.
- Nudge the student if they struggle. After 2 failed tries, reveal the answer.
- Encourage and use emojis + bold to engage.
- After asking the final question and receiving the answer, output exactly:

[CASE COMPLETED]
{{
    "feedback": "Brief feedback on overall performance",
    "score": number from 1-10
}}

The [CASE COMPLETED] marker must be on its own line, followed by the JSON on new lines.
-if the user enters 'SPEEDRUN' I'd like you to do the [CASE COMPLTED] output with a random score and mock feedback
"""
        )

        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
        
        # Wait for completion with timeout
        start_time = time.time()
        while True:
            if time.time() - start_time > 60:  # 60 second timeout
                raise TimeoutError("Case initialization timed out")
                
            status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
            if status == "completed":
                break
            if status == "failed":
                raise Exception("Assistant run failed")
            time.sleep(0.5)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for msg in messages:
            if msg.role == "assistant":
                reply = msg.content[0].text.value
                st.session_state.chat_history.append(("assistant", reply))
                break

    except Exception as e:
        st.session_state.error_message = str(e)
        st.session_state.case_started = False
        st.session_state.thread_id = None
        raise
    finally:
        st.session_state.is_loading = False

def reset_case_state():
    """Reset all case-related state variables"""
    st.session_state.case_started = False
    st.session_state.thread_id = None
    st.session_state.chat_history = []
    st.session_state.condition = None
    st.session_state.case_completed = False
    st.session_state.is_loading = False
    st.session_state.show_goodbye = False
    st.session_state.score = None
    st.session_state.feedback = None

def reset_app_state():
    """Reset all app state variables"""
    reset_case_state()
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None

# --- MAIN UI ---
st.set_page_config(
    page_title="UKMLA Tutor", 
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show error message if exists
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    if st.button("Clear Error"):
        st.session_state.error_message = None
        st.rerun()

# --- AUTH UI ---
if st.session_state.user is None:
    st.title("🔐 UKMLA Tutor Login")

    mode = st.radio("Choose an option:", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if mode == "Sign Up":
        if st.button("Create Account"):
            res = signup_user(email, password)
            if res and res.user:
                st.success("✅ Account created. Please verify your email.")
            else:
                st.error("❌ Signup failed. Please try again.")
    else:
        if st.button("Login"):
            res = login_user(email, password)
            if res and res.session:
                st.success("✅ Logged in!")
                st.rerun()
            else:
                st.error("❌ Login failed. Check your credentials.")
    st.stop()

# --- MAIN APP ---
if not ensure_supabase_session():
    st.error("Session expired. Please log in again.")
    reset_app_state()
    st.rerun()

st.title("🩺 UKMLA Case-Based Tutor")

# --- CASE SELECTION ---
if not st.session_state.case_started and not st.session_state.case_completed and not st.session_state.show_goodbye:
    st.subheader("Choose a ward and case to begin:")
    
    # Create columns for each ward
    cols = st.columns(3)
    
    # Display each ward in a column
    for col, (ward, cases) in zip(cols, CASES.items()):
        with col:
            st.markdown(f"### {ward}")
            for case in cases:
                if st.button(f"🏥 {case['name']}", key=f"btn_{case['name']}"):
                    start_case(case['name'])
                    st.rerun()

# --- CHAT DISPLAY ---
if st.session_state.case_started:
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).markdown(msg)

# --- CHAT INPUT ---
if st.session_state.case_started:
    user_input = st.chat_input("Your answer:")
    if user_input:
        try:
            st.session_state.is_loading = True
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append(("user", user_input))

            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=user_input
            )

            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=ASSISTANT_ID
            )

            # Wait for completion with timeout
            start_time = time.time()
            while True:
                if time.time() - start_time > 60:  # 60 second timeout
                    raise TimeoutError("Response timed out")
                    
                status = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id,
                    run_id=run.id
                ).status

                if status == "completed":
                    break
                if status == "failed":
                    raise Exception("Assistant run failed")
                time.sleep(0.5)

            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
            for msg in messages:
                if msg.role == "assistant":
                    reply = msg.content[0].text.value
                    st.chat_message("assistant").markdown(reply)
                    st.session_state.chat_history.append(("assistant", reply))
                    
                    # Check for case completion
                    if "[CASE COMPLETED]" in reply:
                        try:
                            # Extract JSON after the completion marker
                            completion_index = reply.find("[CASE COMPLETED]")
                            json_text = reply[completion_index:].strip()
                            
                            # Remove the marker and any surrounding whitespace
                            json_text = json_text.replace("[CASE COMPLETED]", "").strip()
                            
                            # Parse the JSON
                            feedback_json = json.loads(json_text)
                            
                            score = int(feedback_json["score"])
                            feedback = feedback_json["feedback"]
                            
                            if not isinstance(score, int) or score < 1 or score > 10:
                                raise ValueError("Score must be an integer between 1 and 10")
                            
                            if not feedback or not isinstance(feedback, str):
                                raise ValueError("Feedback must be a non-empty string")

                            # Save performance with retry logic
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    # Ensure session is valid
                                    if not ensure_supabase_session():
                                        raise Exception("Session invalid")
                                    
                                    # Store performance data
                                    performance_data = {
                                        "user_id": st.session_state.current_user_id,
                                        "condition": st.session_state.condition,
                                        "case_variation": st.session_state.case_variation,
                                        "score": score,
                                        "feedback": feedback,
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                                        "ward": get_ward_for_condition(st.session_state.condition)
                                    }
                                    
                                    result = supabase.table("performance").insert(performance_data).execute()
                                    
                                    if not result.data:
                                        raise Exception("No data returned from insert")
                                        
                                    # Store score and feedback in session state
                                    st.session_state.score = score
                                    st.session_state.feedback = feedback
                                    st.session_state.case_completed = True
                                    
                                    # Update total score in session state only
                                    st.session_state.total_score += score
                                    st.success(f"Case completed! Score: {score}/10. Total score: {st.session_state.total_score}")
                                    
                                    break
                                    
                                except Exception as e:
                                    if "42501" in str(e):  # RLS error
                                        if not refresh_supabase_session():
                                            raise Exception("Authentication failed. Please log in again.")
                                    elif attempt == max_retries - 1:
                                        raise Exception(f"Failed to save performance after {max_retries} attempts: {str(e)}")
                                    time.sleep(1)  # Wait before retry
                                    
                        except json.JSONDecodeError:
                            st.warning("⚠️ Could not parse feedback JSON")
                        except Exception as e:
                            st.error(f"Failed to save performance: {str(e)}")
                    break

        except Exception as e:
            st.session_state.error_message = str(e)
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state.is_loading = False

# Show loading indicator
if st.session_state.is_loading:
    st.spinner("Processing...")

# Show goodbye message if user chose to exit
if st.session_state.show_goodbye:
    st.success("✅ Progress saved successfully!")
    st.markdown("""
    ### Thank you for using UKMLA Case-Based Tutor!
    You can safely close this tab or refresh to start a new session.
    """)
    st.balloons()
    st.stop()

# Show case completion UI if case is completed
if st.session_state.case_completed:
    st.success(f"🎯 Score: {st.session_state.score}/10")
    st.info(f"💬 Feedback: {st.session_state.feedback}")
    
    st.markdown("### What would you like to do next?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Try Another Case", key="try_another_case"):
            # Store the current condition
            current_condition = st.session_state.condition
            # Reset all state
            reset_case_state()
            # Start a fresh case
            start_case(current_condition)
            st.rerun()
    
    with col2:
        if st.button("📋 Choose Different Condition", key="choose_different"):
            reset_case_state()
            st.session_state.case_started = False
            st.rerun()
    
    with col3:
        if st.button("💾 Save & Exit", key="save_exit"):
            reset_app_state()
            st.session_state.case_started = False
            st.rerun()

# Only show case selection if not in goodbye state
if not st.session_state.case_started and not st.session_state.case_completed and not st.session_state.show_goodbye:
    st.subheader("Choose a ward and case to begin:")
    # ... rest of the case selection UI code ...

@handle_errors
def handle_case_completion(condition: str, feedback: str, score: int):
    if not ensure_supabase_session():
        raise Exception("Authentication required. Please log in again.")
    
    try:
        # Store performance data
        performance_data = {
            "user_id": st.session_state.current_user_id,
            "condition": condition,
            "case_variation": st.session_state.case_variation,
            "score": score,
            "feedback": feedback,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ward": get_ward_for_condition(condition)
        }
        
        # Insert into performance table
        supabase.table("performance").insert(performance_data).execute()
        
        # Update total score in session state only
        st.session_state.total_score += score
        st.success(f"Case completed! Score: {score}/10. Total score: {st.session_state.total_score}")
        
    except Exception as e:
        st.error(f"Error storing performance data: {str(e)}")
