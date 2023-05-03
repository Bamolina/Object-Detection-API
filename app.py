import os
import json
import time
import pyodbc
import cv2
import numpy as np
import math
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_restful import reqparse, Api, Resource
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
import pymavlink.mavutil as mavutil
from flask_cors import CORS

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
# Initialize Flask application
app = Flask(__name__)
CORS(app, origins=['http://localhost:4200'])



# Setup Flask Restful framework
api = Api(app)
parser = reqparse.RequestParser()

# Create connection to Azure SQL
server = 'tcp:practicumserver-2023.database.windows.net'
database = 'Practicum'
username = 'practicumstudent'
password = 'Salamah2023'

# Connection string
connection_string = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

def get_db_connection():
    return pyodbc.connect(connection_string)

# YOLOv3 configuration
weights_path = './weights/yolov3.tf'
classes_path = './data/labels/coco.names'
size = 416
num_classes = 80

# Load YOLOv3 model
yolo = YoloV3(classes=num_classes)
yolo.load_weights(weights_path).expect_partial()

class_names = [c.strip() for c in open(classes_path).readlines()]

class DetectionSession(Resource):
    def get(self, session_id):

        conn = get_db_connection()
        cursor = conn.cursor()

        query = f"SELECT * FROM DetectionSessions WHERE SessionId = {session_id}"
        cursor.execute(query)
        result = cursor.fetchone()

        if not result:
            return {"error": "No session found for the provided session ID"}, 404

        session = {"SessionId": result[0], "CreatedAt": str(result[1])}
        return session, 200
class DetectionSessionPost(Resource):

    def post(self):
        raw_images = []
        images = request.files.getlist("images")
        local_time = time.localtime()

        # Generate session ID
        session_id = int(f"{local_time.tm_year}{local_time.tm_mon:02d}{local_time.tm_mday:02d}{local_time.tm_hour:02d}{local_time.tm_min:02d}{local_time.tm_sec:02d}")
        img_dict = {}
        for image in images:
            image.save(os.path.join(os.getcwd(), image.filename))
            img_raw = tf.image.decode_image(
                open(image.filename, 'rb').read(), channels=3)
            raw_images.append(img_raw)
            img_dict[img_raw] = image.filename

        response = []
        center_pixels = []
        for j, raw_img in enumerate(raw_images):
            img = tf.expand_dims(raw_img, 0)
            img = transform_images(img, size)
            boxes, scores, classes, nums = yolo(img)
            # Load the image using OpenCV
            image = cv2.imread(img_dict[img_raw])

            # Get the height and width of the input image
            height, width, _ = img.shape
            detected_count = 0
            for i in range(nums[0]):
                if class_names[int(classes[0][i])] == "person" and float("{0:.2f}".format(np.array(scores[0][i]) * 100)) >= 85.50:
                    detected_count += 1
                    # These values are normalized
                    xmin, ymin, xmax, ymax = boxes
                    center_x = int((xmin+xmax)/2 * width)
                    center_y = int((ymin+ymax)/2 * height)
                    #Center_pixels must be actual pixel values rather than normalized values
                    center_pixels.append((center_x, center_y))

            response.append({
                "image_index": j + 1,
                "detections": detected_count,
                "centers": center_pixels
            })

        # Save session and detections into the Azure SQL tables
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("INSERT INTO DetectionSessions (SessionId) VALUES (?)", session_id)

        for r in response:
            cursor.execute("INSERT INTO Detections (SessionId, Detections, ImageIndex, Centers) VALUES (?, ?, ?, ?)", session_id, r['detections'], r['image_index'], r['centers'])

        conn.commit()

        
      
    
class Detection(Resource):
        def get(self, session_id):
            conn = get_db_connection()
            cursor = conn.cursor()

            query = f"""SELECT *
                    FROM DetectionSessions as s
                    LEFT JOIN Detections as d
                    ON s.SessionId = d.SessionId
                    WHERE s.SessionId = {session_id}"""
            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                return {"error": "No detections found for the provided session ID"}, 404

            detections = []
            for result in results:
                detection = {
                    "SessionId": result[0],
                    "DetectionId": result[2],
                    "ImageIndex": result[4],
                }
                detections.append(detection)

            return {"detections": detections}, 200
        

        def delete(self, session_id):
            conn = get_db_connection()
            cursor = conn.cursor()

            # Delete records from the Detections table
             # Delete records from the Detections table
            query_detections = f"DELETE FROM Detections WHERE SessionId = {session_id}"
            cursor.execute(query_detections)

            # Delete record from the DetectionSessions table
            query_sessions = f"DELETE FROM DetectionSessions WHERE SessionId = {session_id}"
            cursor.execute(query_sessions)

            # Check if any rows were affected
            if cursor.rowcount == 0:
                return {"error": f"No session found for the provided session ID: {session_id}"}, 404

            conn.commit()

            return {"result": "Session and related detections deleted successfully"}, 200
        

class UpFlightPlan(Resource):
    def post(self):
        flight_plan = request.json
        print("Flight plan received:", flight_plan)
        conn = get_db_connection()
        cursor = conn.cursor()
        # Connect to the drone
        cursor.execute("INSERT INTO FlightPlans (FlightPlanJSON) VALUES (?)", json.dumps(flight_plan))
        conn.commit()
        
        return {"result": "Flight plan stored successfully"}, 200    
    


class SetFlightPlan(Resource):

    def post(self):
        
        flight_plan = request.json

        # Convert the flight plan to MAVLink mission items
        mission_items = generate_mavlink_mission_items(flight_plan)

        # Store the MAVLink mission items in the database
        store_flight_plan_in_database(flight_plan, mission_items)

        return jsonify({'status': 'success', 'message': 'Flight plan saved'})


def store_flight_plan_in_database(flight_plan, mission_items):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create a new FlightPlan instance
    cursor.execute("INSERT INTO FlightPlans DEFAULT VALUES")
    conn.commit()

    # Get the ID of the last inserted FlightPlan
    flight_plan_id = cursor.execute("SELECT @@IDENTITY").fetchval()

    # Create and store MissionItem instances for each MAVLink mission item
    for mission_item in mission_items:
        cursor.execute("""INSERT INTO MissionItems
                        (FlightPlanId, Seq, Command, Param1, Param2, Param3, Param4, X, Y, Z, Frame, [Current], Autocontinue)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    flight_plan_id, mission_item.seq, mission_item.command,
                    mission_item.param1, mission_item.param2, mission_item.param3, mission_item.param4,
                    mission_item.x, mission_item.y, mission_item.z,
                    mission_item.frame, mission_item.current, mission_item.autocontinue)
        conn.commit()

    conn.close()




def generate_mavlink_mission_items(flight_plan):
    mission_items = []

    for command in flight_plan["flightPlan"]:
        waypoint_id = command.get("waypointId", 0)
        mav_command = command.get("mavCommand", 0)
        mav_parameters = command.get("mavParameters", [0] * 7)

        mission_item = mavutil.mavlink.MAVLink_mission_item_message(
            target_system=0,
            target_component=0,
            seq=waypoint_id,
            command=mav_command,
            param1=mav_parameters[0],
            param2=mav_parameters[1],
            param3=mav_parameters[2],
            param4=mav_parameters[3],
            x=mav_parameters[4],
            y=mav_parameters[5],
            z=mav_parameters[6],
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            current=0,
            autocontinue=1,
        )

        mission_items.append(mission_item)

    return mission_items


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


# TODO finish this
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
        # If there is a very small difference in location then add 1 to the count
        # And don't add it to the dict
        if "{:.13f}".format(lat) in heatmap_count.values():
            if "{:.13f}".format(lng) in heatmap_count.values():
                heatmap_count["count"] += 1
                pass
        heatmap_count.update({"lat": {formatted_lat}, "lng":  {formatted_lng}, "count": 1})
        

        # Append the lat-long coordinates to the list
        data.append(heatmap_count)
    distanceJSON = "{" + data + "}"
    query = "SELECT CreatedAt FROM DetectionSessions WHERE SessionID IN SELECT SessionID FROM Detections"
    date = cursor.execute(query)


    # TODO finish the park variable assignment
    # Should be along the lines of Select park from [some table] where either
    # SessionID or some other variable is in the table
    park = "UTEP"

    # Ends on this
    cursor.execute("INSERT INTO HeatmapInput date, park, locations VALUES (?, ? ,?)", date, park, distanceJSON)





        
api.add_resource(DetectionSession, '/sessions/int:session_id', methods=['GET'])
api.add_resource(DetectionSessionPost, '/sessions', methods=['POST'])
api.add_resource(Detection, '/detection/<int:session_id>')
api.add_resource(UpFlightPlan, '/flight_plan', methods=['POST'])
api.add_resource(SetFlightPlan, '/save_flight_plan', methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)