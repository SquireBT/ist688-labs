import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("Lab2 - Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.OPENAI_API_KEY
 
# Get the API key from Streamlit secrets (./.streamlit/secrets.toml).
# See https://docs.streamlit.io/develop/concepts/connections/secrets-management
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    # Raised when no secrets.toml exists (e.g. running locally in Codespaces)
    # or when the key is not defined.
    openai_api_key = ""
 
if not openai_api_key:
    st.info(
        "No OpenAI API key detected. Add your key to `.streamlit/secrets.toml` "
        "as `OPENAI_API_KEY = \"sk-...\"` and reload this page.",
        icon="\N{OLD KEY}",
    )
    st.stop()
 
# The three summary styles. The key is the on-screen label; the value is the
# instruction sent to the LLM as part of the prompt.
SUMMARY_CHOICES = {
    "Summarize in 100 words": "Summarize the document in about 100 words.",
    "Summarize in 2 connecting paragraphs": (
        "Summarize the document in exactly two paragraphs, where the second "
        "paragraph clearly connects to and builds on the first."
    ),
    "Summarize in 5 bullet points": (
        "Summarize the document in exactly 5 bullet points. "
        "Output nothing except the bullet points."
    ),
}
 
# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)
 
# --- Options, in the sidebar below the lab navigation -----------------------
# st.navigation renders the Lab 1 / Lab 2 links at the top of the sidebar.
# The divider and header below separate these Lab 2 options from that nav.
 
st.sidebar.divider()
st.sidebar.header("Lab 2 options")
 
summary_choice = st.sidebar.radio(
    "How should the document be summarized?",
    options=list(SUMMARY_CHOICES.keys()),
)
 
use_advanced = st.sidebar.checkbox("Use advanced model")
 
# nano is the cheap default; mini is the more capable model.
model = "gpt-5-mini" if use_advanced else "gpt-5-nano"
st.sidebar.caption(f"Model in use: `{model}`")
 
# Nothing is generated until this button is pressed.
generate = st.button("Generate summary", type="primary", disabled=not uploaded_file)
 
if not uploaded_file:
    st.info("Upload a .txt or .md file to get started.")
 
# --- Generate --------------------------------------------------------------
 
if generate and uploaded_file:
 
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)
 
    # Read the uploaded file and build the instruction from the selected option.
    document = uploaded_file.read().decode()
    instruction = SUMMARY_CHOICES[summary_choice]
 
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
        }
    ]
 
    st.subheader(summary_choice)
 
    # Generate the summary using the OpenAI API.
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
 
    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)