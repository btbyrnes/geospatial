import math
import os
import numpy as np
import pandas as pd
from . import haversine as h

def generate_label_features(point_sequence,window=2):
    
    labels = [] 
    features = []
    
    for i in range(window,len(point_sequence)):
        
        y = point_sequence[i]
        X = point_sequence[i-window:i]
        
        labels.append(y)
        features.append(X)

    return np.array(labels), np.array(features)


def process_files_list(files,column_names,close_lats,close_lons,_POINT_RADIUS=50/3,_MINIMUM_DISTANCE=40000):

    runs_dictionary = {}
    
    for f in files:

        run = {}
        file_name = os.path.basename(f)

        run["file_name"] = file_name
        run["data"] = pd.read_csv("{}".format(f),names=column_names)
        run["data"] = run["data"][run["data"]["latitude"].notnull() == True]
        run["latitude"] = run["data"]["latitude"].to_numpy() * 180 / math.pow(2,31)
        run["longitude"] = run["data"]["longitude"].to_numpy() * 180 / math.pow(2,31)
        lat = run["latitude"]
        lon = run["longitude"]

        if len(lat) < 1:
            continue
        
        y = np.array([np.mean(lat), np.mean(lon)])
        run["mean_location"] = y
        hav_distances = h.haversine(y,(close_lats,close_lons))
        close_points = np.where(hav_distances < _MINIMUM_DISTANCE)[0]
        run["close_points"] = close_points

        if len(close_points) > 0:
            
            points_visited = []
            
            for y,x in np.nditer((lat,lon)):
                j = 0
                for l,ll in np.nditer([close_lats[close_points], close_lons[close_points]]):
                    distance = h.haversine_single_coordinates((y,x),(l,ll))
                    
                    if distance < _POINT_RADIUS: points_visited.append(close_points[j])
                    
                    j += 1
            
            if len(points_visited) > 0: run["points_visited"] = np.array(points_visited)
        
        if "points_visited" in run.keys():
            pv = run["points_visited"]
            run["point_sequence"] = pv[np.where(np.insert(np.diff(pv),0,1) != 0)]

        if "point_sequence" in run.keys():
            runs_dictionary[file_name] = run

    return runs_dictionary