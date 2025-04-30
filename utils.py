import time
import streamlit as st
from functools import wraps
from typing import Optional, Callable, Any
import uuid

def get_user_session_id() -> str:
    """
    Get or create a unique session ID for the current user.
    This ensures each user has their own isolated session.
    """
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    return st.session_state.user_session_id

def get_user_specific_key(base_key: str) -> str:
    """
    Create a user-specific key for session state storage.
    This prevents state collision between concurrent users.
    """
    user_id = st.session_state.get('current_user_id', 'anonymous')
    session_id = get_user_session_id()
    return f"{user_id}_{session_id}_{base_key}"

def get_user_state(key: str, default: Any = None) -> Any:
    """
    Get a user-specific value from session state.
    """
    user_key = get_user_specific_key(key)
    return st.session_state.get(user_key, default)

def set_user_state(key: str, value: Any):
    """
    Set a user-specific value in session state.
    """
    user_key = get_user_specific_key(key)
    st.session_state[user_key] = value

def rate_limit(seconds: int = 1):
    """
    Decorator to prevent function from being called more frequently than specified.
    Uses session_state to track last call time per user.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            # Get unique key for this function and user
            func_key = get_user_specific_key(f"last_call_{func.__name__}")
            
            current_time = time.time()
            last_call = st.session_state.get(func_key, 0)
            
            # Check if enough time has passed
            if current_time - last_call < seconds:
                st.warning(f"Please wait before trying again.")
                return None
                
            # Update last call time
            st.session_state[func_key] = current_time
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_session_expiry() -> bool:
    """
    Check if the current session has expired due to inactivity.
    Returns True if session is valid, False if expired.
    """
    activity_key = get_user_specific_key('last_activity')
    
    if activity_key not in st.session_state:
        st.session_state[activity_key] = time.time()
        return True
        
    idle_time = time.time() - st.session_state[activity_key]
    if idle_time > 3600:  # 1 hour timeout
        st.warning("Session expired due to inactivity. Please log in again.")
        return False
        
    # Update last activity time
    st.session_state[activity_key] = time.time()
    return True

def is_chat_ready() -> bool:
    """
    Check if chat is ready to accept input.
    Returns True if chat is ready, False otherwise.
    """
    # Get user-specific state
    case_started = get_user_state('case_started', False)
    thread_id = get_user_state('thread_id')
    is_loading = get_user_state('is_loading', False)
    
    if not case_started:
        st.warning("Please start a case first.")
        return False
        
    if not thread_id:
        st.warning("Please wait for the case to load...")
        return False
        
    if is_loading:
        st.warning("Please wait for the previous response...")
        return False
        
    return True

def init_user_session():
    """
    Initialize or reset user-specific session state.
    Call this when user logs in or starts a new session.
    """
    if not st.session_state.get('current_user_id'):
        return
        
    # Initialize user-specific state if not exists
    defaults = {
        'case_started': False,
        'thread_id': None,
        'chat_history': [],
        'condition': None,
        'is_loading': False,
        'case_completed': False,
        'score': None,
        'feedback': None,
        'case_variation': None,
        'total_score': 0
    }
    
    for key, default_value in defaults.items():
        if get_user_state(key) is None:
            set_user_state(key, default_value)

def clear_user_session():
    """
    Clear all user-specific session state.
    Call this on logout or session expiry.
    """
    if not st.session_state.get('current_user_id'):
        return
        
    # Clear all user-specific keys
    user_prefix = f"{st.session_state.current_user_id}_{get_user_session_id()}"
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith(user_prefix)]
    for key in keys_to_clear:
        del st.session_state[key] 