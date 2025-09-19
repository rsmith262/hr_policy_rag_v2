# streamlit front end code

import os, requests, streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
API_KEY = os.getenv("API_KEY", "dev-secret")
SESSION_ID = os.getenv("SESSION_ID", "local-user-1")

st.set_page_config(page_title="RAG on Azure AI Search", layout="centered")
st.title("RAG on Azure AI Search (Dev)")

if "messages" not in st.session_state:
    st.session_state.messages = []

def send(msg):
    r = requests.post(
        API_URL,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json={"input": msg, "session_id": SESSION_ID},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resp = send(prompt)
        st.markdown(resp["reply"])
        if resp.get("citations"):
            st.caption("Citations:")
            for c in resp["citations"]:
                label = f"{c['source']}{' (p.'+str(c['page'])+')' if c.get('page') else ''}"
                if c.get("url"):
                    st.markdown(f"• [{label}]({c['url']})")
                else:
                    st.caption(f"• {label}")
        st.session_state.messages.append(("assistant", resp["reply"]))
