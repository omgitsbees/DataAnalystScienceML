Autonomous Electric Bus System for Public Transit
A Project to Simulate and Visualize an AI-Driven Autonomous Bus System Serving Underrepresented Public Transit Routes
Table of Contents

    Overview
    Motivation
    Features
    Technologies Used
    Installation
    Usage
    Visualization Example
    Contributing
    License

Overview

This project simulates an AI-driven autonomous electric bus system designed to improve public transportation between underrepresented Public Transit Routes. The route of interest in this project is from Maple Valley, WA to Bellevue, WA. The system leverages Python, AI technologies, and real-time data visualization to simulate an autonomous bus service, monitor bus performance, manage routes, optimize energy efficiency, and enhance passenger experience.
Motivation

The motivation behind this project is to design an electric bus system that can operate autonomously 24/7, improving transportation options for Public Transit Routes. This project envisions a future where public transit is not only more reliable and available, but also smart, green, and focused on community service.
Features

    AI-based Route Planning: Dynamically adjusts the bus routes based on real-time traffic, passenger demand, and energy efficiency.
    Electric Vehicle Simulation: Simulates battery consumption, energy efficiency, and recharging behavior of electric buses.
    Real-Time GPS Tracking: Uses folium for interactive maps to visualize bus locations, traffic conditions, and environmental factors.
    Autonomous System Behavior: Detects road conditions and obstacles, enabling buses to make informed driving decisions using AI techniques.
    Passenger Load Prediction: Simulates and predicts the passenger load at different times of the day.
    Traffic & Weather Impact: Monitors the impact of weather and traffic on bus speed and performance.
    Energy Optimization: AI-driven decisions on energy-efficient driving behavior, factoring in battery life and charging needs.

Technologies Used

    Python: Primary programming language for simulation and logic.
    Folium: For real-time interactive map visualization.
    NumPy: For data manipulation and calculation.
    Matplotlib: For plotting performance and analytics data.
    Scikit-Learn: For AI-based decision-making and predictions.
    Geopy: For location data and distance calculation.
    Pandas: For data manipulation and analysis.

Installation

To install and run this project locally, follow the steps below:

    Clone the repository:

    bash

git clone https://github.com/your-username/autonomous-bus-system.git
cd autonomous-bus-system

Install the necessary Python packages:

bash

pip install -r requirements.txt

Ensure the following packages are in your requirements.txt:

    folium
    numpy
    matplotlib
    pandas
    scikit-learn
    geopy

Usage

To run the simulation:

    Run the Python script:

    bash

    python autonomous_bus.py

    The simulation will generate:
        Real-time maps showing the bus route.
        Data on traffic, weather, passenger load, and energy usage.
        A dashboard visualizing bus performance and route optimization.

    You can view the map of the bus route from Maple Valley to Bellevue, with real-time bus location, traffic conditions, and environmental information.

Visualization Example

Here’s an example of the real-time bus location tracking using folium:

The system simulates buses moving on the route while displaying energy consumption, battery level, and traffic status. Data visualizations of performance metrics are updated in real-time as well.
