import pyodbc
import cv2
import numpy as np
import math
import tensorflow as tf

# Create connection to Azure SQL
server = 'tcp:practicumserver-2023.database.windows.net'
database = 'Practicum'
username = 'practicumstudent'
password = 'Salamah2023'

# Connection string
connection_string = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

def get_db_connection():
    return pyodbc.connect(connection_string)

# TODO finish park
def distance_to_latlong(distances_list):
    # This method assumes feet
    # Degrees the camera is pointing clockwise from north
    angle = 0 
    
    conn = get_db_connection
    cursor = conn.cursor()
    cursor.execute("SELECT latitude, longitude FROM dbo.Location WHERE Id IN (SELECT DetectionId FROM dbo.Detections")
    results = cursor.fetchall() 
    # Camera location
    camera_lat = results[0]
    camera_lon = results[1]
    data = []
    heatmap_count = {}
    for [x_distance, y_distance] in distances_list:
        # Calculate the relative angle between the camera and the point
        relative_angle = math.atan2(y_distance, x_distance)
        # Calculate the distance between the camera and the point
        distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
        # Calculate the absolute angle between the north and the point
        absolute_angle = angle + math.degrees(relative_angle)
        # Calculate the lat-long coordinates of the point
        # 1 degree of latitude is equal to 364624 feet
        lat = camera_lat + distance * math.cos(math.radians(absolute_angle)) / 364624.0
        # 1 degree of longitude is equal to 288200 feet
        lng = camera_lon + distance * math.sin(math.radians(absolute_angle)) / (288200.0 * math.cos(math.radians(camera_lat)))
        formatted_lat = "{:.14f}".format(lat) 
        formatted_lng ="{:.14f}".format(lng)
        # If there is a very small difference in location
        # Then add 1 to the count and don't add it to the dict
        if "{:.13f}".format(lat) in heatmap_count.values():
            if "{:.13f}".format(lng) in heatmap_count.values():
                heatmap_count["count"] += 1
                pass
        heatmap_count.update({"lat": {formatted_lat}, "lng":  {formatted_lng}, "count": 1})
        

        # Append the lat-long coordinates to the list
        data.append(heatmap_count)
    distanceJSON = "{" + data + "}"
    query = "SELECT CreatedAt FROM DetectionSessions WHERE SessionID IN SELECT SessionID FROM Detections"
    cursor.execute(query)
    date = cursor.fetchone()


    # TODO finish the park variable assignment
    # Should be along the lines of Select park from [some table] where either
    # SessionID or some other variable is in the table
    park = "UTEP"

    # Ends on this
    cursor.execute("INSERT INTO HeatmapInput date, park, locations VALUES (?, ? ,?)", date.CreatedAt, park, distanceJSON)
    conn.commit()
