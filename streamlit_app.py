import streamlit as st

# Define one page per lab. Lab 2 is the default page.
lab1_page = st.Page("Lab1.py", title="Lab 1", icon="1️⃣")
lab2_page = st.Page("Lab2.py", title="Lab 2", icon="2️⃣", default=True)

# Build the navigation (sidebar links) and get the selected page back.
pg = st.navigation([lab1_page, lab2_page])

st.set_page_config(page_title="HW Manager", page_icon="📄")

# Run the selected page.
pg.run()