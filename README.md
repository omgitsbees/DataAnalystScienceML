# Fraud Detection System

## Overview

This project is a Python-based application designed to detect and visualize fraudulent credit card transactions. It uses a dataset of credit card transactions to identify and display fraudulent entries in both a spreadsheet view and a geographic map. The application allows users to interactively explore the data and visualize fraud locations on a map.

## Features

- **Data Table View**: Display the first 10 entries of the dataset with specific columns.
- **Fraud Map Visualization**: Show the locations of the first 10 confirmed fraudulent transactions on an interactive map.

## Requirements

- Python 3.7 or higher
- `pandas`: For data manipulation and analysis
- `tkinter`: For creating the graphical user interface (GUI)
- `folium`: For creating interactive maps
- `webbrowser`: For opening the map in the default web browser

## Installation

1. Clone the repository or download the project files.
2. Install the required Python packages:

   ```bash
   pip install pandas folium

Usage

    Load the Dataset: Make sure the dataset file (credit_card_transactions.csv) is available at the specified path in the script.

    Run the Application:

    bash

    python fraud_detection.py

    Interact with the GUI:
        Show Data Table: Click the "Show Data Table" button to view the first 10 entries of the dataset in a table format.
        Show Fraud Map: Click the "Show Fraud Map" button to open an interactive map in your web browser, displaying the locations of the first 10 confirmed fraudulent transactions.

Code Description

    fraud_detection.py: The main script that loads the dataset, creates the GUI with tkinter, and provides functionalities for displaying the data table and fraud map.
        display_table(): Creates and displays a table view of the first 10 entries with selected columns.
        display_map(): Generates an interactive map with markers for the first 10 confirmed fraud locations and opens it in the web browser.

-----------------------------------------------------------------------------------------------------------------------

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
