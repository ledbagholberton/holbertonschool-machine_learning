#!/usr/bin/env python3
"""
By using the Swapi API, create a method that returns
the list of names of the home planets of all sentient species.
Don’t forget the pagination
"""
import requests


def sentientPlanets():
    """
    Function availableShips
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []
    while url is not None:
        r = requests.get(url)
        datos = r.json()["results"]
        for iter in datos:
            try:
                if (iter["classification"] == "sentient"
                        or iter["designation"] == "sentient"):
                    new_url = iter["homeworld"]
                    new_r = requests.get(new_url)
                    new_datos = new_r.json()["name"]
                    planets.append(new_datos)
            except requests.exceptions.MissingSchema:
                pass
        url = r.json()["next"]
    return planets
