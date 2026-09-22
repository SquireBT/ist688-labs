import requests
import json
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------ config
CHAT_MODEL = 'gpt-5-mini'
DEFAULT_CITY = 'Syracuse, NY'
FORECAST_BLOCKS = 4          # 4 x 3-hour blocks = next ~12 hours

# ------------------------------------------------------------ openai client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

client = st.session_state.openai_client


# ------------------------------------------------------------ weather tool
def get_current_weather(location):
    url = f'https://wttr.in/{location}?format=j1'
    response = requests.get(url, timeout=10)

    if response.status_code == 404:
        raise ValueError(f'Could not find a location named "{location}".')
    if response.status_code != 200:
        raise Exception(f'wttr.in error: status {response.status_code}')

    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise ValueError(f'Could not find a location named "{location}".')


    # j1 has three top-level sections:
    #   current_condition -- one entry, conditions right now
    #   weather           -- three entries, one per day, each with
    #                        min/max, astronomy, and hourly forecasts
    #   nearest_area      -- the location wttr.in actually matched
    current = data['current_condition'][0]
    today = data['weather'][0]
    area = data['nearest_area'][0]

    # I chose to include time, description, rain chance, and temperature as I believe those
    # to be the most relevant options when determining the outfit one would wear on a day to 
    # day basis.
    hourly = [{'time': f"{int(h['time']) // 100:02d}:00",
               'description': h['weatherDesc'][0]['value'],
               'chance_of_rain': int(h['chanceofrain']),
               'feels_like': float(h['FeelsLikeF'])}
              for h in today['hourly'][:FORECAST_BLOCKS]]

    return {'location': f"{area['areaName'][0]['value']}, {area['region'][0]['value']}, "
                        f"{area['country'][0]['value']}",
            'description': current['weatherDesc'][0]['value'],
            'chance_of_rain': int(current['chanceofrain']) if 'chanceofrain' in current else None,
            'feels_like': float(current['FeelsLikeF']),
            'high': float(today['maxtempF']),
            'low': float(today['mintempF']),
            'units': 'Fahrenheit',
            'rest_of_day': hourly}


TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'Get current conditions and the next ~12 hours of forecast for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': "City name, e.g. 'Syracuse, NY'. Defaults to Syracuse, NY.",
                }
            },
            'required': [],
        },
    },
}]


# ------------------------------------------------------------ prompt
SYSTEM_PROMPT = (
    "You are a 'what to wear' assistant. Call get_current_weather for the city the user "
    "names; if no city is given, use Syracuse, NY. Then reply with three short sections: "
    "a one-line weather summary (include how it changes later in the day), clothing "
    "recommendations with reasoning tied to the actual numbers, and two or three outdoor "
    "activity suggestions that suit the conditions. Be concise and practical. "
    "Always begin by stating the location value returned by the tool, since it may differ "
    "from what the student typed."
)


# ------------------------------------------------------------ ui
st.title('What to Wear Bot')
city = st.text_input('City', placeholder=DEFAULT_CITY)

if 'location_error' in st.session_state:
    st.warning(st.session_state.location_error)

if st.button('Get suggestions'):
    st.session_state.pop('location_error', None)

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'What should I wear today in {city or DEFAULT_CITY}?'},
    ]

    # --- first call: model decides whether it needs the weather
    first = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, tools=TOOLS, tool_choice='auto'
    ).choices[0].message

    # --- second call: hand the weather back to the model for advice
    if first.tool_calls:
        messages.append(first)
        for call in first.tool_calls:
            args = json.loads(call.function.arguments or '{}')
            try:
                weather = get_current_weather(args.get('location') or DEFAULT_CITY)
            except ValueError as e:
                st.session_state.location_error = f'{e} Please enter a valid city and try again.'
                st.rerun()
            except requests.RequestException:
                st.error('Could not reach wttr.in. Try again in a moment.')
                st.stop()

            messages.append({
                'role': 'tool',
                'tool_call_id': call.id,
                'content': json.dumps(weather),
            })

        stream = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, stream=True
        )
        st.write_stream(stream)
    else:
        st.write(first.content)