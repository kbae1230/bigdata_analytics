# get today weather

def get_weather(city):
    import requests
    import json
    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=b1b15e88fa797225412429c1c50c122a1'.format(city)
    r = requests.get(url)
    data = r.json()

    return data

print(get_weather('London'))