import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import requests
from sklearn.ensemble import RandomForestClassifier
from geopy.distance import geodesic

# Parameters
num_buses = 10
num_time_steps = 100  # Number of time steps for simulation
max_latitude = 47.6250
min_latitude = 47.3000
max_longitude = -121.9500
min_longitude = -122.3000
max_speed = 60  # Maximum speed in mph
max_passengers = 50
max_traffic_level = 5  # 0 (no traffic) to 5 (heavy traffic)
weather_conditions = ['Clear', 'Rain', 'Fog', 'Snow']
emergency_alerts = ['None', 'Accident', 'Mechanical Issue', 'Medical Emergency']

# Coordinates for Maple Valley and Bellevue
maple_valley_coords = (47.3752, -122.1205)
bellevue_coords = (47.6101, -122.2015)

# Function to generate random data
def generate_data(num_buses, num_time_steps):
    data = {
        'bus_id': [],
        'timestamp': [],
        'latitude': [],
        'longitude': [],
        'speed': [],
        'passenger_count': [],
        'battery_level': [],
        'is_operational': [],
        'maintenance_status': [],
        'traffic_level': [],
        'weather_condition': [],
        'route_id': [],
        'emergency_alert': []
    }
    
    for bus_id in range(num_buses):
        for t in range(num_time_steps):
            data['bus_id'].append(bus_id)
            data['timestamp'].append(pd.Timestamp.now() + pd.to_timedelta(t, unit='m'))
            # Generate coordinates along the route
            fraction = np.random.uniform(0, 1)
            lat = maple_valley_coords[0] + fraction * (bellevue_coords[0] - maple_valley_coords[0])
            lon = maple_valley_coords[1] + fraction * (bellevue_coords[1] - maple_valley_coords[1])
            data['latitude'].append(lat)
            data['longitude'].append(lon)
            data['speed'].append(np.random.uniform(0, max_speed))
            data['passenger_count'].append(np.random.randint(0, max_passengers + 1))
            data['battery_level'].append(np.random.uniform(0, 100))
            data['is_operational'].append(np.random.choice([True, False]))
            data['maintenance_status'].append(np.random.choice([0, 1]))  # Random maintenance status
            data['traffic_level'].append(np.random.randint(0, max_traffic_level + 1))  # Random traffic level
            data['weather_condition'].append(np.random.choice(weather_conditions))  # Random weather condition
            data['route_id'].append(np.random.randint(1, 6))  # Random route ID
            data['emergency_alert'].append(np.random.choice(emergency_alerts))  # Random emergency alert
    
    return pd.DataFrame(data)

# Generate the data
df = generate_data(num_buses, num_time_steps)

# Predictive Maintenance Model
X = df[['speed', 'battery_level', 'passenger_count', 'traffic_level']]
y = df['maintenance_status']
model = RandomForestClassifier()
model.fit(X, y)
df['predicted_maintenance'] = model.predict(X)

# Obstacle Detection (Simplified Simulation)
def detect_obstacles(bus_data):
    return np.random.choice([True, False], size=len(bus_data))

df['obstacles_detected'] = detect_obstacles(df)

# Save the data to a CSV file
df.to_csv('autonomous_buses_simulated_data.csv', index=False)

# Create a Folium map centered around the route
map_center = ((maple_valley_coords[0] + bellevue_coords[0]) / 2, 
              (maple_valley_coords[1] + bellevue_coords[1]) / 2)
m = folium.Map(location=map_center, zoom_start=12)

# Generate route points
def generate_route(start_coords, end_coords, num_points=10):
    route = [start_coords]
    for i in range(1, num_points - 1):
        fraction = i / (num_points - 1)
        intermediate_lat = start_coords[0] + fraction * (end_coords[0] - start_coords[0])
        intermediate_lon = start_coords[1] + fraction * (end_coords[1] - start_coords[1])
        route.append((intermediate_lat, intermediate_lon))
    route.append(end_coords)
    return route

route_points = generate_route(maple_valley_coords, bellevue_coords)
folium.PolyLine(route_points, color='blue', weight=2.5, opacity=1).add_to(m)
folium.Marker(location=maple_valley_coords, popup='Maple Valley', icon=folium.Icon(color='green')).add_to(m)
folium.Marker(location=bellevue_coords, popup='Bellevue', icon=folium.Icon(color='red')).add_to(m)

# Save map to an HTML file
m.save('route_map.html')

print("Map has been saved to 'route_map.html'")

# Visualization of the simulated data

# Bus Location Plot
plt.figure(figsize=(14, 8))
for bus_id in df['bus_id'].unique():
    bus_data = df[df['bus_id'] == bus_id]
    plt.plot(bus_data['longitude'], bus_data['latitude'], label=f'Bus {bus_id}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Bus Locations Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Battery Levels and Operational Status
plt.figure(figsize=(14, 10))

# Battery Levels Plot
plt.subplot(2, 1, 1)
for bus_id in df['bus_id'].unique():
    bus_data = df[df['bus_id'] == bus_id]
    plt.plot(bus_data['timestamp'], bus_data['battery_level'], label=f'Bus {bus_id}')
plt.xlabel('Timestamp')
plt.ylabel('Battery Level (%)')
plt.title('Battery Levels Over Time')
plt.legend()
plt.grid(True)

# Operational Status Plot
plt.subplot(2, 1, 2)
for bus_id in df['bus_id'].unique():
    bus_data = df[df['bus_id'] == bus_id]
    plt.plot(bus_data['timestamp'], bus_data['is_operational'].astype(int), label=f'Bus {bus_id}')
plt.xlabel('Timestamp')
plt.ylabel('Operational Status (1=Operational, 0=Not Operational)')
plt.title('Operational Status Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Traffic Levels and Weather Conditions
plt.figure(figsize=(14, 10))

# Traffic Levels Plot
plt.subplot(2, 1, 1)
for bus_id in df['bus_id'].unique():
    bus_data = df[df['bus_id'] == bus_id]
    plt.plot(bus_data['timestamp'], bus_data['traffic_level'], label=f'Bus {bus_id}')
plt.xlabel('Timestamp')
plt.ylabel('Traffic Level (0-5)')
plt.title('Traffic Levels Over Time')
plt.legend()
plt.grid(True)

# Weather Conditions Plot
plt.subplot(2, 1, 2)
weather_counts = df.groupby('weather_condition').size()
sns.barplot(x=weather_counts.index, y=weather_counts.values, palette='viridis')
plt.xlabel('Weather Condition')
plt.ylabel('Frequency')
plt.title('Weather Conditions Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# Emergency Alerts Plot
plt.figure(figsize=(12, 6))
emergency_counts = df['emergency_alert'].value_counts()
sns.barplot(x=emergency_counts.index, y=emergency_counts.values, palette='coolwarm')
plt.xlabel('Emergency Alert')
plt.ylabel('Frequency')
plt.title('Emergency Alerts Distribution')
plt.grid(True)
plt.show()

# Heatmap of Bus Locations
plt.figure(figsize=(12, 8))
heatmap_data = df.groupby(['latitude', 'longitude']).size().reset_index(name='count')
sns.heatmap(heatmap_data.pivot('latitude', 'longitude', 'count'), cmap='YlGnBu')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Heatmap of Bus Locations')
plt.show()

# Function to get real-time traffic data
def get_real_time_traffic_data(api_key, start_coords, end_coords):
    url = f'https://maps.googleapis.com/maps/api/directions/json?origin={start_coords[0]},{start_coords[1]}&destination={end_coords[0]},{end_coords[1]}&key={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Example usage of real-time traffic data
api_key = 'YOUR_GOOGLE_MAPS_API_KEY'
traffic_data = get_real_time_traffic_data(api_key, maple_valley_coords, bellevue_coords)
print(traffic_data)
