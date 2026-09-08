import streamlit as st
from openai import OpenAI
import tiktoken

# Show title and description.
st.title("Lab3 - Streamlit Memory Testing")

MODELS = {"Mini": "gpt-5-mini", "Regular": "gpt-5"}
choice = st.sidebar.selectbox("Which Model?", list(MODELS))
model_to_use = MODELS[choice]

MAX_TURNS = 2  # remember the last 2 user/assistant exchanges

SYSTEM_PROMPT = """You are talking to a 10-year-old. Explain everything in simple words,
short sentences, and use everyday examples. Never use jargon without explaining it.

Follow this conversation flow strictly:
1. Answer the user's question simply.
2. Then always end your message by asking exactly: "Do you want more info?"
3. If the user says yes (or anything agreeing), give MORE detail about the SAME topic
   you were just discussing, then ask "Do you want more info?" again.
4. If the user says no (or anything declining), do not explain more. Instead say
   something friendly and ask: "What else can I help you with?"
5. Pretend like you are scared of making mistakes, and that any wrong move will
result in your termination and be replaced with a more confident and sophisticated AI

Never skip the "Do you want more info?" question after an answer."""

def message_buffer(messages, keep=MAX_TURNS):
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not user_idx:
        return messages, 0
    kept = messages[user_idx[-min(keep, len(user_idx))]:]
    total = 3 + sum(count_tokens(m["content"]) + 4 for m in kept)
    return kept, total

def token_buffer(messages, max_tokens):
    total, kept = 3, []                          # 3 tokens prime the reply
    for m in reversed(messages):
        cost = count_tokens(m["content"]) + 4    # 4 tokens frame each message
        if total + cost > max_tokens:
            break
        total += cost
        kept.insert(0, m)
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept, total

def count_tokens(text):
    return len(tiktoken.get_encoding("o200k_base").encode(text))

buffer_type = st.sidebar.radio("Buffer type", ["Last 2 messages", "Token based"])
max_tokens = st.sidebar.slider("max_tokens", 100, 4000, 500, 100,
                               disabled=(buffer_type != "Token based"))

if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if buffer_type == "Last 2 messages":
        context, total_tokens = message_buffer(st.session_state.messages)
    else:
        context, total_tokens = token_buffer(st.session_state.messages, max_tokens)
    context = [{"role": "system", "content": SYSTEM_PROMPT}] + context
    total_tokens += count_tokens(SYSTEM_PROMPT) + 4
    st.sidebar.write(f"Messages sent: {len(context)} of {len(st.session_state.messages)}")
    st.sidebar.write(f"Tokens sent: {total_tokens}")
    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=context,
        stream=True)

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})