import os

import requests
import streamlit as st
from langchain.agents import create_agent
from langchain_core.tools import tool

DEFAULT_CITY = "Syracuse, NY"
FORECAST_BLOCKS = 4

# Keep the OpenAI key in the environment so LangChain can find it.
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

SYSTEM_PROMPT = (
    "You are a 'what to wear' assistant. Call get_current_weather for the city the user "
    "names; if no city is given, use Syracuse, NY. Then reply with three short sections: "
    "a one-line weather summary (include how it changes later in the day), clothing "
    "recommendations with reasoning tied to the actual numbers, and two or three outdoor "
    "activity suggestions that suit the conditions. Be concise and practical. "
    "Always begin by stating the location value returned by the tool, since it may differ "
    "from what the student typed."
)


@tool
def get_weather_data(location: str = "Syracuse, NY") -> dict:
    """Return the current weather and the next ~12 hours of forecast for a city so the assistant can recommend what to wear."""
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)

    if response.status_code == 404:
        raise ValueError(f'Could not find a location named "{location}".')
    if response.status_code != 200:
        raise ValueError(f"wttr.in error: status {response.status_code}")

    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError(f'Could not find a location named "{location}".') from exc

    if "current_condition" not in data or "weather" not in data or "nearest_area" not in data:
        raise ValueError(f'Could not find a location named "{location}".')

    current = data["current_condition"][0]
    today = data["weather"][0]
    area = data["nearest_area"][0]

    hourly = [
        {
            "time": f"{int(hour['time']) // 100:02d}:00",
            "description": hour["weatherDesc"][0]["value"],
            "chance_of_rain": int(hour.get("chanceofrain", 0)),
            "feels_like": float(hour["FeelsLikeF"]),
        }
        for hour in today["hourly"][:FORECAST_BLOCKS]
    ]

    weather = {
        "location": (
            f"{area['areaName'][0]['value']}, {area['region'][0]['value']}, "
            f"{area['country'][0]['value']}"
        ),
        "description": current["weatherDesc"][0]["value"],
        "chance_of_rain": int(current["chanceofrain"]) if "chanceofrain" in current else None,
        "feels_like": float(current["FeelsLikeF"]),
        "high": float(today["maxtempF"]),
        "low": float(today["mintempF"]),
        "units": "Fahrenheit",
        "rest_of_day": hourly,
    }

    return weather


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather_data],
    system_prompt=SYSTEM_PROMPT,
)

# ------------------------------------------------------------ ui
st.title("What to Wear Bot")
location = st.text_input("City", placeholder=DEFAULT_CITY)

if "location_error" in st.session_state:
    st.warning(st.session_state["location_error"])

if st.button("Get advice"):
    st.session_state.pop("location_error", None)

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"What should I wear in {location or DEFAULT_CITY}?"}]}
        )

        with st.chat_message("assistant"):
            st.write(result["messages"][-1].content)

        # Part D answers:
        # a. There are usually 4 messages: system, user, AI tool call request, tool result, and final AI answer.
        #    The exact count depends on whether the model calls the tool and how many tool rounds occur.
        # b. The model asks for the weather on the AI message whose content is empty; the actual tool request is in
        #    message.tool_calls, not in the visible text content.
        # c. In Lab 5, this was created by the `client.chat.completions.create(..., tools=TOOLS, tool_choice='auto')`
        #    call, then by appending the returned `first` message to `messages`, and by adding the tool result with
        #    `messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': json.dumps(weather)})`.
        with st.expander("Behind the scenes"):
            for message in result["messages"]:
                st.write(f"**{message.type}**")
                st.write(message.content)
                if message.type == "ai" and getattr(message, "tool_calls", None):
                    st.write(message.tool_calls)

    except ValueError as exc:
        st.session_state["location_error"] = f"{exc} Please enter a valid city and try again."
        st.rerun()
    except requests.RequestException:
        st.error("Could not reach wttr.in. Try again in a moment.")