# app.py
# Main Flask application — entry point for the Weather Prediction Web App.
# Run with:  python app.py
# Then open: http://127.0.0.1:5000

from flask import Flask, render_template, request, jsonify
import requests
import os
import traceback

from model import predict_temperature

app = Flask(__name__)

# ── OpenWeatherMap API config ────────────────────────────────────────────────
API_KEY  = '5381b175ca0a1af5a2fca247e0364d93'          # ← Replace with your real key from openweathermap.org
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'


def get_weather_data(city: str) -> dict:
    """
    Fetches real-time weather from OpenWeatherMap API.
    Returns a clean dict or raises an exception with a user-friendly message.
    """
    try:
        params   = {'q': city, 'appid': API_KEY, 'units': 'metric'}
        response = requests.get(BASE_URL, params=params, timeout=8)
        data     = response.json()

        if response.status_code == 401:
            raise ValueError("Invalid API key. Please update API_KEY in app.py.")

        if response.status_code == 404:
            raise ValueError(f"City '{city}' not found. Check spelling and try again.")

        if response.status_code != 200:
            raise ValueError(f"API error ({response.status_code}): {data.get('message', 'Unknown error')}")

        return {
            'city'        : data['name'],
            'country'     : data['sys']['country'],
            'actual_temp' : round(data['main']['temp'], 1),
            'feels_like'  : round(data['main']['feels_like'], 1),
            'temp_min'    : round(data['main']['temp_min'], 1),
            'temp_max'    : round(data['main']['temp_max'], 1),
            'humidity'    : data['main']['humidity'],
            'pressure'    : data['main']['pressure'],
            'wind_speed'  : round(data['wind']['speed'], 1),
            'description' : data['weather'][0]['description'].title(),
            'icon'        : data['weather'][0]['icon'],
            'visibility'  : round(data.get('visibility', 0) / 1000, 1),  # km
        }

    except requests.exceptions.ConnectionError:
        raise ConnectionError("No internet connection. Please check your network.")

    except requests.exceptions.Timeout:
        raise TimeoutError("Request timed out. OpenWeatherMap API is slow — try again.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Renders the main page (search form)."""
    return render_template('index.html')


@app.route('/weather', methods=['POST'])
def weather():
    """
    Handles form submission:
      1. Fetches real-time weather data.
      2. Passes features into ML model.
      3. Returns results to index.html.
    """
    city = request.form.get('city', '').strip()

    if not city:
        return render_template('index.html', error="Please enter a city name.")

    try:
        weather_data     = get_weather_data(city)
        predicted_temp   = predict_temperature(
            humidity   = weather_data['humidity'] /100,
            pressure   = weather_data['pressure'],
            wind_speed = weather_data['wind_speed'],
        )
        diff             = round(predicted_temp - weather_data['actual_temp'], 1)
        diff_label       = f"+{diff}" if diff >= 0 else str(diff)

        return render_template(
            'index.html',
            weather       = weather_data,
            predicted_temp= predicted_temp,
            diff          = diff_label,
            city_query    = city,
        )

    except (ValueError, ConnectionError, TimeoutError) as e:
        return render_template('index.html', error=str(e), city_query=city)

    except FileNotFoundError as e:
        return render_template('index.html', error=str(e), city_query=city)

    except Exception:
        # Log full traceback to console for debugging, show friendly message to user
        traceback.print_exc()
        return render_template('index.html',
                               error="An unexpected error occurred. Check the console.",
                               city_query=city)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
