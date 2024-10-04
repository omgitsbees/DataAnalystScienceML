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

--------------------------------------------------------------------------

Fashion MNIST Generative Adversarial Network (GAN)

This repository contains an implementation of a Generative Adversarial Network (GAN) using PyTorch. The model is trained on the Fashion MNIST dataset to generate new images of fashion items.
Features

    Generator: A neural network that generates new images from random noise (latent vectors).
    Discriminator: A neural network that classifies whether an image is real (from the dataset) or fake (generated).
    Training: The GAN is trained using a combination of adversarial loss for both the generator and discriminator.
    Data Augmentation: The Fashion MNIST dataset is augmented using random horizontal flips and rotations for more robust training.

Dataset

The model uses the Fashion MNIST dataset, a set of grayscale images (28x28 pixels) representing different clothing items such as shirts, pants, and shoes.
Installation

    Clone the repository:

    bash

git clone https://github.com/your_username/fashion-mnist-gan.git
cd fashion-mnist-gan

Install the required Python packages:

bash

    pip install torch torchvision matplotlib numpy

Model Architecture
Generator

The generator network takes a random noise vector (latent space) as input and outputs a 28x28 grayscale image. It consists of fully connected layers with ReLU activation, batch normalization, and dropout.
Discriminator

The discriminator network takes a 28x28 image as input and classifies it as either real or fake. It also consists of fully connected layers with ReLU activation, batch normalization, and dropout.
Training

To train the GAN, run the provided script. It trains the model for 100 epochs and prints the generator and discriminator losses after each epoch.

bash

python gan.py

Results

The model generates a batch of fashion items after training. Below is a sample of generated images after 100 epochs:

Hyperparameters

    Image Size: 784 (28x28)
    Hidden Size: 256
    Latent Size: 100
    Batch Size: 100
    Number of Epochs: 100
    Learning Rate: 0.001 (for both generator and discriminator)

Usage

    Training: Train the GAN by running the script. Training progress (discriminator and generator loss) will be printed after each epoch.
    Sampling: After training, the model will generate 25 samples of fashion items, which will be displayed in a grid.

Example

Here’s how to generate some samples after training:

python

sample_noise = torch.randn((25, latent_size), device=device)
samples = generator(sample_noise)

License

This project is licensed under the MIT License.
