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

# Function that returns a list of lists distances from a person to the camera
def distances_from_center():
    # Camera is located at (0, 0, 20), probably should be obtained from drone/gopro itself
    # This code assumes that the camera is pointed at a 45 degree angle
    camera_height = 5

    # Fields of view of the gopro in linear setting
    fov_horizontal = 87
    fov_vertical = 56
    fov_diagonal = 95

    # Pixel resolution of the camera
    resolution_x = 1920
    resolution_y = 1080

    # Connect to db and obtain list of all processed images
    conn = get_db_connection
    cur = conn.cursor()
    # The table needs to have the x_coord, y_coord of the center where the person was detected
    cur.execute('SELECT centers FROM dbo.Detections') 
    

    # List of objects with their x and y pixel coordinates relative to the center of the screen
    coordinates = cur.fetchall()
    distances_list = []


    for list in coordinates:
        # List of distances from object to camera
        distances = []
        for x_pixel, y_pixel in list:
            # Convert pixel coordinates to angular distance from center of screen
            angle_x = math.atan((x_pixel - resolution_x/2) / (resolution_x/2) * math.tan(math.radians(fov_horizontal)/2))
            angle_y = math.atan((y_pixel - resolution_y/2) / (resolution_y/2) * math.tan(math.radians(fov_vertical)/2))
            
            # Calculate the diagonal angle
            angle_diagonal = math.atan(math.sqrt(math.tan(angle_x)**2 + math.tan(angle_y)**2))
            
            # Calculate the distance to the object
            tan_half_fov_diagonal = math.tan(math.radians(fov_diagonal)/2)
            object_distance_from_center = tan_half_fov_diagonal * camera_height / math.sqrt(1 + tan_half_fov_diagonal**2)
            d = object_distance_from_center / math.cos(angle_diagonal)
            distances.append[d]
            # Limit precision to 2 decimal places
            print("Object at ({}, {}): {:.2f} feet away".format(x_pixel, y_pixel, d))
        distances_list.append(distances)
    return distances_list