#!/usr/bin/env python3
"""
By using the Swapi API, create a method that returns the list
of ships that can hold a given number of passengers:
Don’t forget the pagination
If no ship available, return an empty list.
"""
import requests


def availableShips(passengerCount):
    """
    Function availableShips
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    transport = []
    while url is not None:
        r = requests.get(url)
        datos = r.json()["results"]
        for ship in datos:
            a = ship["passengers"]
            a = a.replace(',', '')
            if a.isnumeric() and int(a) >= passengerCount:
                transport.append(ship["name"])
        url = r.json()["next"]
    return transport
