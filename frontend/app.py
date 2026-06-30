"""Streamlit frontend for the Domain Knowledge Co-Pilot."""

from __future__ import annotations

import requests
import streamlit as st
from streamlit_local_storage import LocalStorage

BASE_URL = "http://localhost:8000"
_ls = LocalStorage()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _check_auth(r) -> bool:
    """Return False and trigger logout if the JWT has expired."""
    if r is not None and r.status_code == 401:
        st.warning("Session expired. Please log in again.")
        logout()
        return False
    return True


def api_post(path: str, *, json: dict | None = None, files=None, auth: bool = True):
    headers = _headers() if auth else {}
    try:
        r = requests.post(f"{BASE_URL}{path}", json=json, files=files, headers=headers, timeout=60)
        if auth and not _check_auth(r):
            return None
        return r
    except requests.ConnectionError:
        st.error("Cannot reach the backend. Make sure the FastAPI server is running on port 8000.")
        return None


def api_get(path: str, *, params: dict | None = None):
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=30)
        if not _check_auth(r):
            return None
        return r
    except requests.ConnectionError:
        st.error("Cannot reach the backend. Make sure the FastAPI server is running on port 8000.")
        return None


def api_delete(path: str):
    try:
        r = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=30)
        if not _check_auth(r):
            return None
        return r
    except requests.ConnectionError:
        st.error("Cannot reach the backend.")
        return None


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def restore_session() -> None:
    """Re-populate session state from localStorage after a page refresh."""
    if st.session_state.get("token"):
        return
    token = _ls.getItem("dk_token")
    if token:
        st.session_state["token"] = token
        st.session_state["username"] = _ls.getItem("dk_username") or ""
        load_corpora()


def is_logged_in() -> bool:
    return bool(st.session_state.get("token"))


def logout():
    for key in ("token", "username", "corpora", "active_corpus_id", "chat_history"):
        st.session_state.pop(key, None)
    st.session_state["_pending_ls_delete"] = True
    st.rerun()


def load_corpora():
    r = api_get("/corpora")
    if r and r.status_code == 200:
        st.session_state["corpora"] = r.json()
    else:
        st.session_state["corpora"] = []


def load_chat_history(corpus_id: int):
    r = api_get(f"/corpora/{corpus_id}/chat-history", params={"limit": 100})
    if r and r.status_code == 200:
        msgs = r.json()
        msgs.reverse()
        st.session_state["chat_history"] = msgs
    else:
        st.session_state["chat_history"] = []


# ---------------------------------------------------------------------------
# Auth page
# ---------------------------------------------------------------------------

def render_auth():
    st.title("Domain Knowledge Co-Pilot")
    st.caption("Ask questions over your own document library")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please fill in all fields.")
            else:
                r = api_post("/login", json={"email": email, "password": password}, auth=False)
                if r is None:
                    pass
                elif r.status_code == 200:
                    data = r.json()
                    username = email.split("@")[0]
                    st.session_state["token"] = data["access_token"]
                    st.session_state["username"] = username
                    st.session_state["_pending_ls_save"] = True
                    load_corpora()
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Login failed."))

    with tab_signup:
        with st.form("signup_form"):
            username = st.text_input("Username")
            email_s = st.text_input("Email", key="signup_email")
            password_s = st.text_input("Password", type="password", key="signup_pw")
            submitted_s = st.form_submit_button("Create Account", use_container_width=True)

        if submitted_s:
            if not username or not email_s or not password_s:
                st.error("Please fill in all fields.")
            else:
                r = api_post(
                    "/signup",
                    json={"username": username, "email": email_s, "password": password_s},
                    auth=False,
                )
                if r is None:
                    pass
                elif r.status_code == 201:
                    st.success("Account created! Please log in.")
                else:
                    st.error(r.json().get("detail", "Sign up failed."))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown(f"**{st.session_state.get('username', 'User')}**")
        if st.button("Logout", use_container_width=True):
            logout()

        st.divider()

        # --- Corpus picker ---
        st.subheader("Corpora")
        corpora: list[dict] = st.session_state.get("corpora", [])

        if corpora:
            corpus_names = [c["name"] for c in corpora]
            active_idx = 0
            active_id = st.session_state.get("active_corpus_id")
            if active_id:
                ids = [c["id"] for c in corpora]
                if active_id in ids:
                    active_idx = ids.index(active_id)

            chosen = st.selectbox(
                "Active corpus",
                options=corpus_names,
                index=active_idx,
                label_visibility="collapsed",
            )
            new_active = corpora[corpus_names.index(chosen)]

            if new_active["id"] != st.session_state.get("active_corpus_id"):
                st.session_state["active_corpus_id"] = new_active["id"]
                load_chat_history(new_active["id"])
                st.rerun()
        else:
            st.caption("No corpora yet.")

        # --- Create corpus ---
        with st.expander("New corpus"):
            with st.form("new_corpus_form"):
                new_name = st.text_input("Name", placeholder="e.g. Thesis papers")
                create_btn = st.form_submit_button("Create", use_container_width=True)
            if create_btn:
                if not new_name.strip():
                    st.error("Name cannot be empty.")
                else:
                    r = api_post("/corpora", json={"name": new_name.strip()})
                    if r and r.status_code == 201:
                        load_corpora()
                        corpus_data = r.json()
                        st.session_state["active_corpus_id"] = corpus_data["id"]
                        load_chat_history(corpus_data["id"])
                        st.rerun()
                    elif r:
                        st.error(r.json().get("detail", "Failed to create corpus."))

        # --- Delete corpus ---
        if corpora and st.session_state.get("active_corpus_id"):
            with st.expander("Delete active corpus"):
                st.warning("This permanently removes the corpus and all its data.")
                if st.button("Delete", type="primary", use_container_width=True):
                    cid = st.session_state["active_corpus_id"]
                    r = api_delete(f"/corpora/{cid}")
                    if r and r.status_code in (200, 204):
                        st.session_state.pop("active_corpus_id", None)
                        st.session_state.pop("chat_history", None)
                        load_corpora()
                        st.rerun()
                    elif r:
                        st.error(r.json().get("detail", "Delete failed."))

        st.divider()

        # --- Upload PDF ---
        st.subheader("Upload PDF")
        active_corpus_id = st.session_state.get("active_corpus_id")
        if not active_corpus_id:
            st.caption("Select or create a corpus first.")
        else:
            uploaded = st.file_uploader(
                "Choose a PDF",
                type=["pdf"],
                label_visibility="collapsed",
            )
            if uploaded:
                with st.spinner("Uploading and indexing..."):
                    r = api_post(
                        f"/corpora/{active_corpus_id}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    )
                if r and r.status_code == 201:
                    data = r.json()
                    chunks = data.get("chunks_stored", data.get("id", ""))
                    st.success(
                        f"Indexed **{uploaded.name}**"
                        + (f" ({chunks} chunks)" if chunks else "")
                    )
                elif r:
                    st.error(r.json().get("detail", "Upload failed."))


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

def render_chat():
    active_corpus_id = st.session_state.get("active_corpus_id")
    corpora: list[dict] = st.session_state.get("corpora", [])

    if not active_corpus_id:
        st.info("Create or select a corpus from the sidebar to get started.")
        return

    corpus_name = next(
        (c["name"] for c in corpora if c["id"] == active_corpus_id),
        f"Corpus {active_corpus_id}",
    )
    st.title(corpus_name)

    # Load history on first render for this corpus
    if "chat_history" not in st.session_state:
        load_chat_history(active_corpus_id)

    chat_history: list[dict] = st.session_state.get("chat_history", [])

    # Render past messages
    for msg in chat_history:
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])

    # Chat input
    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                r = api_post(
                    f"/corpora/{active_corpus_id}/query",
                    json={"question": question},
                )

            if r is None:
                st.error("No response from backend.")
            elif r.status_code == 200:
                answer = r.json().get("answer", "")
                st.markdown(answer)
                st.session_state.setdefault("chat_history", []).append(
                    {"question": question, "answer": answer}
                )
            else:
                detail = r.json().get("detail", "Query failed.")
                st.error(detail)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Domain Knowledge Co-Pilot",
    page_icon="📚",
    layout="wide",
)

restore_session()

# Flush deferred localStorage writes — must happen in a render without an
# immediate st.rerun() following, so the browser JS actually executes.
if st.session_state.pop("_pending_ls_save", False):
    _ls.setItem("dk_token", st.session_state.get("token", ""), key="ls_set_token")
    _ls.setItem("dk_username", st.session_state.get("username", ""), key="ls_set_username")

if st.session_state.pop("_pending_ls_delete", False):
    _ls.deleteItem("dk_token", key="ls_del_token")
    _ls.deleteItem("dk_username", key="ls_del_username")

if not is_logged_in():
    render_auth()
else:
    render_sidebar()
    render_chat()
