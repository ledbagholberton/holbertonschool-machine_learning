#!/usr/env/python3
"""
By using the Swapi API, create a method that returns the list
of ships that can hold a given number of passengers:


Don’t forget the pagination
If no ship available, return an empty list.
"""
import requests


def availableShips(passengerCount):
    "Function availableShips"
    url = https://swapi-api.hbtn.io/api/starships
    r = requests.get(url)
    print(r.status_code)
    