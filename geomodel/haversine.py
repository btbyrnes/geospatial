import numpy as np

def haversine(coordinates_1, coordinates_2):
    
    if len(coordinates_2) > 1:
        distances = []
        iterator = np.nditer(coordinates_2)
        for lat,lon in iterator:
            coords = np.array([lat,lon])
            distances.append(haversine_single_coordinates(coordinates_1,coords))
        return np.array(distances)
    else:
        return haversine_single_coordinates(coordinates_1,coordinates_2)


def haversine_single_coordinates(coordinates_1, coordinates_2):
    R = 6378.137 * 1e3

    lat1 = coordinates_1[0] * np.pi / 180.0
    lat2 = coordinates_2[0] * np.pi / 180.0

    lon1 = coordinates_1[1] * np.pi / 180.0
    lon2 = coordinates_2[1] * np.pi / 180.0

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    inner = np.sin(delta_lat/2.0)*np.sin(delta_lat/2.0) + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2.0)*np.sin(delta_lon/2.0)

    return 2 * R * np.arcsin( np.sqrt(inner) )