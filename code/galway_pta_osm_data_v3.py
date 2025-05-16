#!/usr/bin/env python
# coding: utf-8

# # Galway Public Transport Accessibility Score Analysis

# ### Import Libraries

# In[91]:


data_dir = '/Users/conall/Projects/atlantech-ai-challenge-2025/data'
code_dir = '/Users/conall/Projects/atlantech-ai-challenge-2025/code'
artifact_dir = '/Users/conall/Projects/atlantech-ai-challenge-2025/artifacts'

# Libraries
# ! pip install langchain openai pandas tabulate
# # ! pip install langchain-community
# ! pip install langchain-ollama
# ! pip install thefuzz 
# ! pip install python-Levenshtein
# ! pip install osmnx
# ! pip install geopandas 
# ! pip install selenium
## ! pip install numpy==1.26.4
## ! pip uninstall numpy -y


import pandas as pd
pd.set_option('display.max_colwidth', None) # display full column width
import os
import requests
import random
from IPython.display import display
import numpy as np
import zipfile
import shutil # For removing the directory if needed for a clean re-extract
from selenium import webdriver # Used for scraping bus timetables from buseireann.ie
import time
import re # For regular expressions
from thefuzz import process, fuzz
import networkx as nx
from math import radians, sin, cos, sqrt, atan2  #for calculating the distance between two points
import osmnx as ox
# import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd #for handling the spatial data
import logging
import warnings
import matplotlib.patheffects as path_effects
import pickle
from io import StringIO
from langchain_ollama import ChatOllama #for building tool-augmented agents


# from langchain.tools import tool, StructuredTool
# from langchain.tools import BaseTool
# from pydantic import BaseModel, Field, root_validator
# ! pip freeze > requirements.txt


# **In this notebook, we perform:**
# 
# - Data Collection, Validation, Cleaning & Transformation
# 
# - Exploratory Data Analysis (EDA) & POI Definition: Perform EDA and define Points of Interest (POIs)
# 
# - Network Graph Construction & Public Transport Mapping: Combine graph construction with the mapping of POIs and bus stops
# 
# - Accessibility Metric Definition, Score Calculation & Analysis 
# 
# - Visualization of POIs on Galway Map
# 

# ### 1. Bus Stop Data Ingestion (Galway API)
# 
# Reference: https://galway-bus.apis.ie
# 
# Attributes: stop_id, stop_name, stop_lat, stop_lon, direction, and route_id

# In[92]:


def load_or_fetch_galway_stops_data(
    route_ids: list, 
    api_url_template: str, 
    output_dir: str
) -> pd.DataFrame:
    """
    Loads gstops_df_v1 from a CSV file if it exists in output_dir. 
    Otherwise, fetches data from the Galway Bus API for the given route_ids,
    processes it (adds custom 'BS' index), saves it to CSV in output_dir, 
    and returns the DataFrame.

    Args:
        route_ids (list): A list of route IDs to process (e.g., [401, 402]).
        api_url_template (str): An API URL template string with a placeholder for route_id.
                                Example: "https://galway-bus.apis.ie/api/groute/{route_id}"
        output_dir (str): Directory where 'gstops_df_v1.csv' will be saved.

    Returns:
        pd.DataFrame: The gstops_df_v1 DataFrame, either loaded or newly fetched and processed.
    """
    print(f"--- Starting Galway Stops Data Processing (gstops_df_v1) ---")
    print(f"Output directory for CSV: {output_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    output_filename = 'gstops_df_v1.csv'
    output_csv_path = os.path.join(output_dir, output_filename)
    
    gstops_df_v1 = pd.DataFrame() # Initialize
    # The 'if' block below ensures the subsequent API fetching logic (outside this selection) is executed.
    if gstops_df_v1.empty: 
        print(f"Always fetching data from API for gstops_df_v1...")
        all_stops = []

        for route_id in route_ids:
            url = api_url_template.format(route_id=route_id)
            print(f"Fetching data for route_id: {route_id} from URL: {url}")
            try:
                response = requests.get(url, timeout=10) # Added a timeout
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    # API sometimes returns a dict for single result, list for multiple. Standardize.
                    if isinstance(results, dict): 
                        results = [results]
                    
                    if not results:
                        print(f"Route id {route_id} information not available or no results from API.")
                        continue
                    
                    for direction_info in results:
                        direction_id = direction_info.get('direction_id')
                        for stop in direction_info.get('g_stops', []):
                            stop_row = {
                                'stop_id': stop.get('stop_id'), # Use .get() for safety
                                'stop_name': stop.get('stop_name'),
                                'stop_lat': stop.get('stop_lat'),
                                'stop_lon': stop.get('stop_lon'),
                                'direction': direction_id,
                                'route_id': route_id
                            }
                            all_stops.append(stop_row)
                else:
                    print(f"Route id {route_id} API request failed (HTTP {response.status_code}).")
            except requests.exceptions.RequestException as e: # Catch specific request errors
                print(f"Error fetching data for route id {route_id} from API: {e}")
            except Exception as e: # Catch other unexpected errors (like JSONDecodeError)
                 print(f"An unexpected error occurred for route id {route_id}: {e}")


        if all_stops:
            gstops_df_v1_from_api = pd.DataFrame(all_stops)
            print(f"\nSuccessfully fetched {len(gstops_df_v1_from_api)} stops from API.")

            print("Creating custom 'BS' indices for new API data...")
            sort_columns = ['route_id', 'direction', 'stop_id']
            if all(col in gstops_df_v1_from_api.columns for col in sort_columns):
                gstops_df_v1_sorted = gstops_df_v1_from_api.sort_values(by=sort_columns).reset_index(drop=True)
            else:
                print(f"Warning: Not all sort columns ({sort_columns}) present. Indexing based on current fetched order.")
                gstops_df_v1_sorted = gstops_df_v1_from_api.reset_index(drop=True)

            bus_stop_indices = [f'BS{i+1}' for i in range(len(gstops_df_v1_sorted))]
            gstops_df_v1_sorted.index = bus_stop_indices
            gstops_df_v1_sorted.index.name = 'bs_index' # Name the index
            gstops_df_v1 = gstops_df_v1_sorted 
            
            if not gstops_df_v1.empty:
                try:
                    gstops_df_v1.to_csv(output_csv_path, index=True) # Save with index
                    print(f"gstops_df_v1 fetched, processed, and saved to {output_csv_path}.")
                except Exception as e:
                    print(f"Error saving gstops_df_v1 to {output_csv_path}: {e}")
            else:
                # This case would occur if all_stops was initially non-empty, 
                # but processing resulted in an empty gstops_df_v1.
                print(f"gstops_df_v1 is empty after processing. "
                      f"Not saving to {output_csv_path}, preserving existing file if present.")
        else:
            print("No stop data collected from API. gstops_df_v1 will be empty.")
            # Create an empty DataFrame with expected columns if no data was fetched and no file loaded
            gstops_df_v1 = pd.DataFrame(columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'direction', 'route_id'])
            gstops_df_v1.index.name = 'bs_index' 

    print("\n--- gstops_df_v1 Final State (inside function) ---")
    if not gstops_df_v1.empty:
        display(gstops_df_v1.head())
    else:
        print("gstops_df_v1 is empty.")
    print(f"Shape: {gstops_df_v1.shape}, Index Name: {gstops_df_v1.index.name if gstops_df_v1.index.name else 'Not set'}")
    print(f"--- Galway Stops Data Processing Finished ---")
    return gstops_df_v1



if __name__ == '__main__':
    # Define parameters
    route_ids_to_fetch = [401, 402, 404, 405, 407, 409, 410, 411, 412, 414]
    galway_api_url_template = "https://galway-bus.apis.ie/api/groute/{route_id}" 
    

    # Call the function
    # IMPORTANT: This will run live API calls if the CSV doesn't exist.
    gstops_df_v1 = load_or_fetch_galway_stops_data(
                                            route_ids_to_fetch,
                                            galway_api_url_template,
                                            artifact_dir 
                                        )

    print("\n--- Function Call Complete (Example Usage) ---")
    if not gstops_df_v1.empty:
        print(f"\nReturned gstops_df_v1 shape: {gstops_df_v1.shape}")
        print("gstops_df_v1 Head:")
        display(gstops_df_v1.head())
        print(f"Show first few gstops_df_v1 Index: {gstops_df_v1.index[:5]}") # Show first few index values
    else:
        print("\nReturned gstops_df_v1 is empty.")


# ### 2. Bus Eireann Timetables: Scraping and Processing
# 
# Reference: https://www.buseireann.ie/routes-and-timetables/401

# In[93]:


def scrape_and_process_bus_eireann_timetables(
    route_ids: list, 
    base_url_template: str, 
    output_dir: str  # Directory to save the output CSV files
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrapes Bus Eireann timetables for given route_ids using Selenium,
    processes the data, saves it to CSV files, and returns the DataFrames.

    Args:
        route_ids (list): A list of route IDs to scrape (e.g., [401, 402]).
        base_url_template (str): A URL template string with a placeholder for route_id.
                                 Example: "https://www.buseireann.ie/routes-and-timetables/{route_id}"
        output_dir (str): Directory where the output CSV files will be saved.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - all_bus_eirean_timetables: DataFrame with all scraped timetable data.
            - bus_timetables: Processed DataFrame with selected columns 
                                       and 'stop_order_on_route'.
    """
    print(f"--- Starting Bus Eireann Timetable Scraping & Processing ---")
    print(f"Output directory for CSVs: {output_dir}")

    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    all_bus_eirean_timetables = pd.DataFrame()
    
    # --- 1. Web Scraping using Selenium ---
    driver = None 
    try:
        print("Attempting to set up Chrome WebDriver...")
        driver = webdriver.Chrome() 

        print("WebDriver setup successful.")

        for route_id in route_ids:
            url = base_url_template.format(route_id=route_id)
            print(f"Processing route: {route_id} from URL: {url}")
            driver.get(url)
            print(f"Opened URL: {url}")
            time.sleep(5) # Wait for dynamic content to load

            print(f"Attempting to get current table for route {route_id}...")
            page_html = driver.page_source
            try:
                list_of_dataframes = pd.read_html(StringIO(str(page_html)))
                if list_of_dataframes:
                    print(f"Found {len(list_of_dataframes)} table(s) for route {route_id}.")
                    timetable_df = list_of_dataframes[0]  
                    timetable_df['route_id'] = route_id
                    print(f"Extracted DataFrame for route {route_id} (Shape: {timetable_df.shape})")
                    all_bus_eirean_timetables = pd.concat([all_bus_eirean_timetables, timetable_df], ignore_index=True)
                else:
                    print(f"No tables found for route {route_id} on page: {url}")
            except ValueError as ve:
                print(f"No tables found by pandas for route {route_id} (pd.read_html error: {ve})")
            except Exception as table_ex:
                print(f"Could not parse tables for route {route_id}: {table_ex}")
        
        # Save the scraped data
        if not all_bus_eirean_timetables.empty:
            # Clean column names immediately after scraping
            all_bus_eirean_timetables.columns = [
                str(col).replace('.', '_').replace(' ', '_').lower() 
                for col in all_bus_eirean_timetables.columns
            ]
            raw_csv_path = os.path.join(output_dir, 'all_bus_eirean_timetables.csv')
            all_bus_eirean_timetables.to_csv(raw_csv_path, index=False)
            print(f"Combined timetable scraped and saved to {raw_csv_path}")
        else:
            print("No timetable data was extracted during scraping.")

    except Exception as e:
        print(f"An error occurred during Selenium operations: {e}")
        # Depending on requirements, you might want to re-raise the exception
        # or return empty DataFrames if scraping is critical.
    finally:
        if driver:
            print("Closing WebDriver.")
            driver.quit()

    # --- 2. Processing the Scraped Data ---
    bus_timetables = pd.DataFrame() # Initialize

    print("\n--- Processing Scraped Timetable Data ---")
    if not all_bus_eirean_timetables.empty:
        print(f"Total rows in scraped DataFrame: {len(all_bus_eirean_timetables)}")
        print(f"Unique route_ids in scraped DataFrame: {all_bus_eirean_timetables['route_id'].unique()}")

        # Define required columns for the processed DataFrame (adjust based on actual scraped column names)
        # Common columns from Bus Eireann timetables often include 'ROUTE' (stop name/description).
        # The actual column name for stop names needs to be verified from your scraped data.
        # Let's assume it's 'route' after our cleaning.
        required_columns = ['route_id', 'route'] # 'route' should be the stop name column.
        
        available_columns = [col for col in required_columns if col in all_bus_eirean_timetables.columns]
        
        if 'route_id' in available_columns and 'route' in available_columns : # Check if essential columns are present
            bus_timetables = all_bus_eirean_timetables[available_columns].copy()
            print(f"\n--- bus_timetables created with columns: {available_columns} ---")
            print(f"Shape of bus_timetables: {bus_timetables.shape}")

            print("\nAdding 'stop_order_on_route' column...")
            # Ensure data is sorted by route_id before applying cumcount
            bus_timetables = bus_timetables.sort_values(by='route_id', kind='mergesort').reset_index(drop=True)
            bus_timetables['stop_order_on_route'] = bus_timetables.groupby('route_id').cumcount()
            
            print("'stop_order_on_route' column added.")
            print(f"Shape of bus_timetables after adding stop_order: {bus_timetables.shape}")

            # Save the processed DataFrame
            processed_csv_path = os.path.join(output_dir, 'bus_timetables.csv')
            bus_timetables.to_csv(processed_csv_path, index=False)
            print(f"Processed bus timetables saved to {processed_csv_path}")
            
            # Print samples
            unique_routes_to_sample = bus_timetables['route_id'].unique()
            if len(unique_routes_to_sample) > 0:
                sample_route_id = unique_routes_to_sample[0]
                print(f"\nSample for route_id '{sample_route_id}':")
                print(bus_timetables[bus_timetables['route_id'] == sample_route_id].head(10))
        else:
            missing_essential = [col for col in ['route_id', 'route'] if col not in available_columns]
            print(f"\nError: Not all essential columns ({missing_essential}) found in scraped data.")
            print(f"Available columns after scraping and cleaning: {list(all_bus_eirean_timetables.columns)}")
            print("Cannot create detailed processed bus_timetables.")
    else:
        print("\nScraped timetable data (all_bus_eirean_timetables) is empty. No processing done.")

    print(f"--- Timetable Scraping & Processing Finished ---")
    return all_bus_eirean_timetables, bus_timetables



if __name__ == '__main__':
       
    route_ids_to_scrape = [401, 402, 404, 405, 407, 409, 410, 411, 412, 414]
    bus_eireann_url_template = "https://www.buseireann.ie/routes-and-timetables/{route_id}" 
    
    all_bus_eirean_timetables, bus_timetables = scrape_and_process_bus_eireann_timetables(
                                                                            route_ids_to_scrape,
                                                                            bus_eireann_url_template,
                                                                            artifact_dir 
                                                                        )

    print("\n--- Function Call Complete (Example Usage) ---")
    if not all_bus_eirean_timetables.empty:
        print(f"\nReturned raw_df (all_bus_eirean_timetables) shape: {all_bus_eirean_timetables.shape}")
        print("Raw DF Head:")
        display(all_bus_eirean_timetables.head())
    else:
        print("\nReturned all_bus_eirean_timetables is empty.")

    if not bus_timetables.empty:
        print(f"\nReturned bus_timetables (bus_timetables_final_df) shape: {bus_timetables.shape}")
        print("Processed DF Head:")
        display(bus_timetables.head())
    else:
        print("\nReturned bus_timetables is empty.")


# ### 3. Bus Route Data Ingestion (Galway API)
# 
# **Attributes:**
# 
# - route_long_name: Full name of the route
# 
# - g_trip_headsign: Destination displayed on the bus
# 
# - route_id: Unique identifier for the route
# 
# - route_short_name: Short route number (e.g. 401, 402)
# 
# - direction_id: Direction of travel (0 or 1)
# 
# - first_stop_id: ID of the first stop
# 
# - last_stop_id: ID of the last stop
# 
# - first_stop_name: Name of the first stop
# 
# - last_stop_name: Name of the last stop
# 
# - num_stops: Total number of stops on the route

# In[94]:


def load_or_fetch_galway_route_variations(
    route_ids: list, 
    api_url_template: str, 
    output_dir: str
) -> pd.DataFrame:
    """
    Loads gvariations_df_v1 from a CSV file if it exists in output_dir.
    Otherwise, fetches route variation data from the Galway Bus API for the given route_ids,
    processes it (adds custom 'BR' index), saves it to CSV in output_dir,
    and returns the DataFrame.

    Args:
        route_ids (list): A list of route IDs to process (e.g., [401, 402]).
        api_url_template (str): An API URL template string with a placeholder for route_id.
                                Example: "https://galway-bus.apis.ie/api/groute/{route_id}"
        output_dir (str): Directory where 'gvariations_df_v1.csv' is stored/will be saved.

    Returns:
        pd.DataFrame: The gvariations_df_v1 DataFrame, either loaded or newly fetched and processed.
    """
    print(f"--- Starting Galway Route Variations Data Processing (gvariations_df_v1) ---")
    print(f"Output directory for CSV: {output_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    output_filename = 'gvariations_df_v1.csv'
    output_csv_path = os.path.join(output_dir, output_filename)
    
    gvariations_df_v1 = pd.DataFrame() # Initialize

    # Try to load the DataFrame from CSV
    if os.path.exists(output_csv_path):
        print(f"Loading existing gvariations_df_v1 from: {output_csv_path}")
        try:
            gvariations_df_v1 = pd.read_csv(output_csv_path, index_col=0)
            gvariations_df_v1.index.name = 'br_index' # Ensure index name is set
            print(f"Loaded gvariations_df_v1 DataFrame with shape: {gvariations_df_v1.shape}")
        except Exception as e:
            print(f"Error loading {output_csv_path}: {e}. Will attempt to fetch from API.")
            gvariations_df_v1 = pd.DataFrame() # Reset if loading failed
            
    if gvariations_df_v1.empty: # If file didn't exist or loading failed
        print(f"File not found at {output_csv_path} or was empty/corrupt. Fetching gvariations data from API...")
        all_variations = []

        for route_id in route_ids:
            url = api_url_template.format(route_id=route_id)
            print(f"Fetching variations for route_id: {route_id} from URL: {url}")
            try:
                response = requests.get(url, timeout=10) # Added timeout
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if isinstance(results, dict): # Standardize results to be a list
                        results = [results]
                    
                    if not results:
                        print(f"Route id {route_id} (variations) - no results from API.")
                        continue
                        
                    for direction_info in results:
                        # Extract top-level info once per direction
                        route_long_name = direction_info.get('route_long_name')
                        g_trip_headsign = direction_info.get('g_trip_headsign')
                        route_short_name = direction_info.get('route_short_name')
                        direction_id = direction_info.get('direction_id')
                        
                        for variation in direction_info.get('g_route_variations', []):
                            row = {
                                'route_long_name': route_long_name,
                                'g_trip_headsign': g_trip_headsign,
                                'route_id': route_id, # Queried route_id
                                'route_short_name': route_short_name,
                                'direction_id': direction_id,
                                'variation_route_id': variation.get('route_id'), # ID from g_route_variations
                                'first_stop_id': variation.get('first_stop_id'),
                                'last_stop_id': variation.get('last_stop_id'),
                                'first_stop_name': variation.get('first_stop_name'),
                                'last_stop_name': variation.get('last_stop_name'),
                                'num_stops': variation.get('num_stops')
                            }
                            all_variations.append(row)
                else:
                    print(f"Route id {route_id} (variations) - API request failed (HTTP {response.status_code}).")
            except requests.exceptions.RequestException as e:
                print(f"Error fetching variations data for route id {route_id} from API: {e}")
            except Exception as e:
                 print(f"An unexpected error occurred for route id {route_id} (variations): {e}")


        if all_variations:
            gvariations_df_from_api = pd.DataFrame(all_variations)
            print(f"\nSuccessfully fetched {len(gvariations_df_from_api)} route variations from API.")

            # --- Create custom 'BR' indices ---
            print("Creating custom 'BR' indices for new API gvariations_data...")
            sort_columns = ['route_id', 'direction_id', 'variation_route_id', 'first_stop_id']
            # Ensure all sort columns exist before attempting to sort by them
            existing_sort_columns = [col for col in sort_columns if col in gvariations_df_from_api.columns]
            if len(existing_sort_columns) == len(sort_columns) :
                 gvariations_df_sorted = gvariations_df_from_api.sort_values(by=existing_sort_columns).reset_index(drop=True)
            else:
                print(f"Warning: Not all columns for sorting ({existing_sort_columns}) present in gvariations. Indexing based on current order.")
                gvariations_df_sorted = gvariations_df_from_api.reset_index(drop=True)

            bus_route_variation_indices = [f'BR{i+1}' for i in range(len(gvariations_df_sorted))]
            gvariations_df_sorted.index = bus_route_variation_indices
            gvariations_df_sorted.index.name = 'br_index' # Name the index
            gvariations_df_v1 = gvariations_df_sorted
            
            try:
                gvariations_df_v1.to_csv(output_csv_path, index=True) # Save with index
                print(f"gvariations_df_v1 fetched, processed, and saved to {output_csv_path}.")
            except Exception as e:
                print(f"Error saving gvariations_df_v1 to {output_csv_path}: {e}")
        else:
            print("No route variation data collected from API. gvariations_df_v1 will be empty.")
            # Create an empty DataFrame with expected columns
            gvariations_df_v1 = pd.DataFrame(columns=[
                'route_long_name', 'g_trip_headsign', 'route_id', 'route_short_name',
                'direction_id', 'variation_route_id', 'first_stop_id', 'last_stop_id',
                'first_stop_name', 'last_stop_name', 'num_stops'
            ])
            gvariations_df_v1.index.name = 'br_index'

    print("\n--- gvariations_df_v1 Final State (inside function) ---")
    if not gvariations_df_v1.empty:

        try:
            display(gvariations_df_v1.head()) 
        except NameError:
            print(gvariations_df_v1.head())   
    else:
        print("gvariations_df_v1 is empty.")
    print(f"Shape: {gvariations_df_v1.shape}, Index Name: {gvariations_df_v1.index.name if gvariations_df_v1.index.name else 'Not set'}")
    print(f"--- Galway Route Variations Data Processing Finished ---")
    return gvariations_df_v1


if __name__ == '__main__':
    # Define parameters
    route_ids_to_fetch_variations = [401, 402, 404, 405, 407, 409, 410, 411, 412, 414]
    galway_api_url_template_variations = "https://galway-bus.apis.ie/api/groute/{route_id}" 
    
    # Call the function
    gvariations_df_v1 = load_or_fetch_galway_route_variations(
        route_ids_to_fetch_variations,
        galway_api_url_template_variations,
        artifact_dir 
    )

    print("\n--- Function Call Complete (Example Usage for Variations) ---")
    if not gvariations_df_v1.empty:
        print(f"\nReturned gvariations_df_v1 shape: {gvariations_df_v1.shape}")
        print("gvariations_dataframe Head:")
      
        try:
            display(gvariations_df_v1.head())
        except NameError:
            print(gvariations_df_v1.head())
        print(f"gvariations_df_v1 Index: {gvariations_df_v1.index[:5]}")
    else:
        print("\nReturned gvariations_df_v1 is empty.")


# ### 4. Fetch Geofabrik Shapefiles (Ireland)
# 
# https://libguides.ucd.ie/gisguide/findspatialdata  
# 
# https://download.geofabrik.de/europe/ireland-and-northern-ireland.html 

# In[95]:


def ensure_osm_shapefiles_are_ready(
    download_url: str,
    target_data_dir: str, 
    local_zip_leaf_name: str, 
    extracted_subdir_leaf_name: str, 
    expected_marker_filename: str 
) -> str | None:
    """
    Ensures OSM shapefile data is downloaded from a URL and extracted.
    - Checks if extracted data (indicated by a marker file) already exists.
    - If not, checks if the zip file exists; if not, downloads it.
    - Extracts the zip file.
    - Returns the path to the directory containing extracted files if successful, None otherwise.

    Args:
        download_url (str): URL to download the zip file from.
        target_data_dir (str): The base directory to store the zip file and 
                               the extracted subdirectory.
        local_zip_leaf_name (str): The leaf name for the downloaded zip file 
                                   (e.g., "data.zip").
        extracted_subdir_leaf_name (str): The leaf name for the subdirectory where 
                                          contents will be extracted. This directory 
                                          will be created inside target_data_dir.
        expected_marker_filename (str): A key file name expected within the 
                                        extracted subdirectory to confirm 
                                        successful extraction (e.g., "roads.shp").

    Returns:
        str | None: The full path to the extracted subdirectory if data is ready, 
                    None on failure.
    """
    print(f"--- Ensuring OSM Shapefile data from {download_url} is ready ---")

    # Construct full paths based on the leaf names and the target data directory
    local_zip_full_path = os.path.join(target_data_dir, local_zip_leaf_name)
    extraction_target_full_path = os.path.join(target_data_dir, extracted_subdir_leaf_name)
    expected_marker_full_path = os.path.join(extraction_target_full_path, expected_marker_filename)

    print(f"Target data directory: {target_data_dir}")
    print(f"Local zip path to be used/created: {local_zip_full_path}")
    print(f"Extraction target directory to be used/created: {extraction_target_full_path}")
    print(f"Expected marker file for validation: {expected_marker_full_path}")

    try:
        # Ensure the base target_data_dir exists (for storing the zip and the extracted folder)
        if not os.path.exists(target_data_dir):
            os.makedirs(target_data_dir)
            print(f"Created target data directory: {target_data_dir}")

        # 1. Check if the final extracted data (indicated by marker file) already exists
        if os.path.exists(expected_marker_full_path):
            print(f"Marker file '{expected_marker_full_path}' found.")
            print(f"Extracted data already present at: {extraction_target_full_path}")
            return extraction_target_full_path  # no subsequent download/extraction is needed if file already exists

        # 2. If extracted data not found, check for/download the zip file
        if not os.path.exists(local_zip_full_path):
            print(f"Downloading from {download_url} to {local_zip_full_path}...")
            # Use a timeout for the request
            response = requests.get(download_url, stream=True, timeout=60) 
            response.raise_for_status() # Raise an error for bad status codes
            with open(local_zip_full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192 * 4): # Use a reasonable chunk size
                    f.write(chunk)
            print("Download complete.")
        else:
            print(f"Zip file already exists at {local_zip_full_path}. Proceeding to extraction.")

        # 3. Extract the zip file
        print(f"Extracting {local_zip_full_path} to {extraction_target_full_path}...")
        # Ensure the specific directory for extraction exists before calling extractall
        if not os.path.exists(extraction_target_full_path):
             os.makedirs(extraction_target_full_path)
             print(f"Created extraction target directory: {extraction_target_full_path}")
        
        with zipfile.ZipFile(local_zip_full_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_target_full_path)
        print("Extraction complete.")
        
        # 4. Verify extraction by checking for the marker file again
        if os.path.exists(expected_marker_full_path):
            print(f"Extraction verified. Data ready at: {extraction_target_full_path}")
            return extraction_target_full_path
        else:
            print(f"Error: Extraction seemed complete, but marker file '{expected_marker_full_path}' not found.")
            if os.path.exists(extraction_target_full_path):
                 print(f"Contents of '{extraction_target_full_path}': {os.listdir(extraction_target_full_path)}")
            else:
                print(f"Extraction target directory '{extraction_target_full_path}' does not exist after attempt.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(local_zip_full_path): # Attempt to clean up partial or corrupted download
            try: 
                os.remove(local_zip_full_path)
                print(f"Removed potentially partial/corrupted zip file: {local_zip_full_path}")
            except OSError as ose: 
                print(f"Error removing zip file {local_zip_full_path}: {ose}")
        return None
    except zipfile.BadZipFile:
        print(f"Error: File at {local_zip_full_path} is not a valid zip file or is corrupted.")
        if os.path.exists(local_zip_full_path): # Attempt to clean up corrupted zip
            try: 
                os.remove(local_zip_full_path)
                print(f"Removed corrupted zip file: {local_zip_full_path}")
            except OSError as ose: 
                print(f"Error removing zip file {local_zip_full_path}: {ose}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == '__main__':

    config_geofabrik_url = "https://download.geofabrik.de/europe/ireland-and-northern-ireland-latest-free.shp.zip"
    
    
    config_zip_leaf_name = "ireland-and-northern-ireland-latest-free.shp.zip"
    config_extracted_subdir_name = "ireland-and-northern-ireland-latest-free.shp" # The folder where shapefiles will go
    config_marker_filename = 'gis_osm_roads_free_1.shp' # this is for testing check if the extraction was successful by checking for roads.shp


    # Call the function
    shapefile_base_path = ensure_osm_shapefiles_are_ready(
        download_url=config_geofabrik_url,
        target_data_dir= data_dir,  # data_dir is defined in the global scope
        local_zip_leaf_name=config_zip_leaf_name,
        extracted_subdir_leaf_name=config_extracted_subdir_name,
        expected_marker_filename=config_marker_filename  # this is to check if the extraction was successful by checking for roads.shp
    )

    if shapefile_base_path:
        print(f"\nOSM data is ready and located in: {shapefile_base_path}")
    else:
        print("\nFailed to prepare OSM data. Please check the errors reported above.")


# ### 5. Rahoon-Portershed Public Transport Accessibility Map for Galway 

# In[96]:


def galway_transport_map(
    place_name: str,
    shapefile_base_dir: str,
    shapefile_layers: dict,
    gstops_df_v1: pd.DataFrame,
    gvariations_df_v1: pd.DataFrame 
):
    """
    Processes geospatial data for a given place, generates relevant DataFrames,
    and plots a transport map.

    Args:
        place_name (str): The name of the place to process (e.g., "Galway, Ireland").
        shapefile_base_dir (str): Path to the directory containing shapefiles.
        shapefile_layers (dict): Dictionary mapping layer names to shapefile filenames.
        gstops_df_v1 (pd.DataFrame): DataFrame containing bus stop data.
        gvariations_df_v1 (pd.DataFrame): DataFrame containing bus route variation data.

    Returns:
        tuple: A tuple containing three DataFrames:
            - bus_stops_gdf (gpd.GeoDataFrame or None)
            - galway_places_summary_df1 (pd.DataFrame or None)
            - galway_buildings_summary_df1 (pd.DataFrame or None)
    """

    # --- Initialize variables that will be returned ---
    # These are often initialized as None or empty within the original code blocks
    # ensuring they have a defined state even if some processing steps are skipped or fail.
    boundary_gdf = None
    galway_gdfs = {}
    bus_stops_gdf = None
    bus_routes_gdf = None # Though not returned, it's created and used internally
    galway_places_summary_df = None
    galway_places_summary_df1 = None
    galway_buildings_summary_df = None
    galway_buildings_summary_df1 = None
    rahoon_place_id = None

    logging.basicConfig(level=logging.INFO) 
    warnings.filterwarnings("ignore")      

    print(f"\n--- Processing Data for: {place_name} ---")
    print(f"Using Shapefile directory: {shapefile_base_dir}")

    try:
        # --- *** GET GALWAY BOUNDARY *** ---
        print("\nFetching boundary for Galway...")
        boundary_gdf = ox.geocode_to_gdf(place_name).to_crs("EPSG:4326")
        if boundary_gdf.empty:
            raise ValueError(f"Could not geocode '{place_name}'.")
        print(f"Boundary fetched. CRS set to: {boundary_gdf.crs}")

        # --- *** LOAD IRELAND SHAPEFILES & CLIP TO GALWAY BOUNDARY *** ---
        print("\nLoading and clipping Ireland-wide layers to Galway boundary...")
        # galway_gdfs is initialized above
        for layer_name_key, shp_filename in shapefile_layers.items(): # layer_name_key to avoid clash if layer_name is used later
            shp_path = os.path.join(shapefile_base_dir, shp_filename)
            print(f"--- Processing layer: {layer_name_key} ---")
            if not os.path.exists(shp_path):
                print(f"*** WARNING: Shapefile not found: {shp_path} - Skipping layer '{layer_name_key}' ***")
                continue
            try:
                ireland_layer_gdf = gpd.read_file(shp_path)
                if boundary_gdf is not None and ireland_layer_gdf.crs != boundary_gdf.crs: # Check boundary_gdf
                    ireland_layer_gdf = ireland_layer_gdf.to_crs(boundary_gdf.crs)
                
                if boundary_gdf is not None: # Check boundary_gdf before clipping
                    clipped_gdf = gpd.clip(ireland_layer_gdf, boundary_gdf, keep_geom_type=True)
                    if not clipped_gdf.empty:
                        galway_gdfs[layer_name_key] = clipped_gdf
                    else:
                        print(f"Note: No features found for layer '{layer_name_key}' after clipping.")
                else:
                    print(f"Warning: Boundary GDF not available for clipping layer '{layer_name_key}'. Skipping clipping.")


            except Exception as e:
                print(f"*** ERROR processing layer '{layer_name_key}': {e} ***")

        # --- *** PREPARE BUS STOP GEODATAFRAME FROM GSTOPS_DF_V1 *** --- 
        print("\nPreparing Galway Bus Stop data from gstops_df_v1...")
        # bus_stops_gdf is initialized above
        if isinstance(gstops_df_v1, pd.DataFrame) and not gstops_df_v1.empty:
            if 'stop_lat' in gstops_df_v1.columns and 'stop_lon' in gstops_df_v1.columns:
                try:
                    temp_stops_df = gstops_df_v1.dropna(subset=['stop_lat', 'stop_lon']).copy()
                    
                    if not temp_stops_df.empty:
                        bus_stops_gdf = gpd.GeoDataFrame(
                            temp_stops_df,
                            geometry=gpd.points_from_xy(temp_stops_df['stop_lon'], temp_stops_df['stop_lat']),
                            crs="EPSG:4326"  
                        )
                        print(f"Created GeoDataFrame 'bus_stops_gdf' with {len(bus_stops_gdf)} stops from gstops_df_v1.")
                        if boundary_gdf is not None and bus_stops_gdf.crs != boundary_gdf.crs: # Check boundary_gdf
                            print(f"Reprojecting bus stops GDF to {boundary_gdf.crs}...");
                            bus_stops_gdf = bus_stops_gdf.to_crs(boundary_gdf.crs)
                            print("Reprojection complete.")
                    else:
                        print("Warning: No valid coordinates found in gstops_df_v1 after cleaning.")
                except Exception as e:
                    print(f"*** ERROR converting gstops_df_v1 data: {e} ***")
                    bus_stops_gdf = None 
            else:
                print("Warning: 'stop_lat' or 'stop_lon' columns not found in gstops_df_v1.")
        else:
            print("Warning: 'gstops_df_v1' DataFrame not provided or is empty.")

        # --- *** PREPARE BUS ROUTES GEODATAFRAME FROM gvariations_df_v1 *** --- 
        # bus_routes_gdf is initialized above
        if isinstance(gvariations_df_v1, pd.DataFrame) and not gvariations_df_v1.empty and \
           isinstance(bus_stops_gdf, gpd.GeoDataFrame) and not bus_stops_gdf.empty:

            print("\nEnriching gvariations_df_v1 with first/last stop Point geometries...")
            bus_routes_gdf = gvariations_df_v1.copy()
            
            if 'stop_id' in bus_stops_gdf.columns and 'geometry' in bus_stops_gdf.columns:
                bus_stops_gdf_unique_locations = bus_stops_gdf.drop_duplicates(subset=['stop_id'], keep='first')
                stop_id_to_point_geometry = bus_stops_gdf_unique_locations.set_index('stop_id')['geometry']
            else:
                print("Error: 'stop_id' or 'geometry' column not found in bus_stops_gdf. Cannot map stop Point geometries.")
                stop_id_to_point_geometry = pd.Series(dtype='object') 

            bus_routes_gdf['first_stop_point'] = bus_routes_gdf['first_stop_id'].map(stop_id_to_point_geometry)
            bus_routes_gdf['last_stop_point'] = bus_routes_gdf['last_stop_id'].map(stop_id_to_point_geometry)
            
            num_first_stops_mapped = bus_routes_gdf['first_stop_point'].notna().sum()
            num_last_stops_mapped = bus_routes_gdf['last_stop_point'].notna().sum()
            
            print(f"Successfully mapped Point geometry for {num_first_stops_mapped} first stops.")
            print(f"Successfully mapped Point geometry for {num_last_stops_mapped} last stops.")

            if bus_routes_gdf['first_stop_point'].isnull().any() or bus_routes_gdf['last_stop_point'].isnull().any():
                print("Warning: Some first/last stop points could not be mapped (resulting in NaNs).")
            
            print("\n--- bus_routes_gdf (with Point geometries) ---")
            display_cols = ['first_stop_id', 'first_stop_point', 'last_stop_id', 'last_stop_point']
            if 'route_id' in bus_routes_gdf.columns: display_cols.insert(0, 'route_id')
            if 'direction_id' in bus_routes_gdf.columns: display_cols.insert(1, 'direction_id')
            
            try: display(bus_routes_gdf[display_cols].head()) 
            except NameError: print(bus_routes_gdf[display_cols].head()) # fallback

            print(f"Shape of bus_routes_gdf: {bus_routes_gdf.shape}")
        else:
            print("\nPrerequisite DataFrames ('gvariations_df_v1' or 'bus_stops_gdf') not available or empty. Cannot create bus_routes_gdf.")

        # --- *** CREATE PLACE SUMMARY DATAFRAME *** ---
        print("\nCreating DataFrame for Galway Place Names and Coordinates...")
        # galway_places_summary_df is initialized above
        if 'places_poly' in galway_gdfs and not galway_gdfs['places_poly'].empty:
            places_data = []
            if 'name' not in galway_gdfs['places_poly'].columns:
                print("Warning: 'name' column not found in places_poly layer. Cannot extract place names.")
            else:
                for idx, row_place in galway_gdfs['places_poly'][galway_gdfs['places_poly']['name'].notna() & galway_gdfs['places_poly'].geometry.is_valid].iterrows():
                    place_name_val = row_place['name']; geometry = row_place.geometry; rep_point = None
                    if hasattr(geometry, 'representative_point'):
                        try: rep_point = geometry.representative_point()
                        except Exception: rep_point = geometry.centroid 
                    else: rep_point = geometry.centroid 
                    if rep_point and rep_point.is_valid:
                        places_data.append({'place_name': place_name_val,'latitude': rep_point.y,'longitude': rep_point.x})
                if places_data:
                    galway_places_summary_df = pd.DataFrame(places_data)
                    print(f"Created DataFrame 'galway_places_summary_df' with {len(galway_places_summary_df)} places.")
                    try: display(galway_places_summary_df.head())
                    except NameError: print(galway_places_summary_df.head())
                else: print("No valid places with names found to create summary DataFrame.")
        else: print("Clipped 'places_poly' GeoDataFrame not found or is empty.")

        # galway_places_summary_df1 is initialized above
        if isinstance(galway_places_summary_df, pd.DataFrame) and not galway_places_summary_df.empty:
            galway_places_summary_df1 = galway_places_summary_df.copy()
            if 'place_name' in galway_places_summary_df1.columns:
                galway_places_summary_df1 = galway_places_summary_df1.sort_values('place_name').reset_index(drop=True)
            else:
                print("Warning: 'place_name' column not found for sorting. Index will be based on current order.")
            place_indices = [f'P{i+1}' for i in range(len(galway_places_summary_df1))]
            galway_places_summary_df1.index = place_indices
            print("\nCreated DataFrame 'galway_places_summary_df1' with custom 'P' indices:")
            print(f"Number of places: {len(galway_places_summary_df1)}")
            print("\nFirst few rows of 'galway_places_summary_df1':")
            try: display(galway_places_summary_df1.head())
            except NameError: print(galway_places_summary_df1.head())
        else:
            print("Cannot create 'galway_places_summary_df1' as 'galway_places_summary_df' is not available or is empty.")

        # --- *** CHECK RAHOON PLACE ID FOR PLOTTING *** ---
        # rahoon_place_id is initialized above
        if isinstance(galway_places_summary_df1, pd.DataFrame) and not galway_places_summary_df1.empty:
            if 'place_name' in galway_places_summary_df1.columns:
                rahoon_search_results = galway_places_summary_df1[galway_places_summary_df1['place_name'].str.contains('Rahoon', case=False, na=False)]
                if not rahoon_search_results.empty:
                    print(f"\n--- Found 'Rahoon' in galway_places_summary_df1 ---")
                    rahoon_place_data = rahoon_search_results.iloc[0]
                    rahoon_place_id = rahoon_place_data.name 
                    print(f"Place Name: {rahoon_place_data['place_name']}")
                    print(f"Index (ID): {rahoon_place_id}")
                    print(f"Latitude: {rahoon_place_data['latitude']}")
                    print(f"Longitude: {rahoon_place_data['longitude']}")
                else:
                    print("\nPlace name containing 'Rahoon' not found in galway_places_summary_df1.")
            else:
                print("\n'place_name' column not found in galway_places_summary_df1.")
        else:
            print("\nDataFrame 'galway_places_summary_df1' not available for searching 'Rahoon'.")

        # --- *** CREATE BUILDINGS SUMMARY DATAFRAME *** ---
        print("\nCreating DataFrame for Galway Buildings with Type and Coordinates...")
        # galway_buildings_summary_df is initialized above
        if 'buildings' in galway_gdfs and not galway_gdfs['buildings'].empty:
            buildings_data = []
            print(f"Available columns in buildings layer: {galway_gdfs['buildings'].columns.tolist()}")
            for idx, row_building in galway_gdfs['buildings'][galway_gdfs['buildings'].geometry.is_valid].iterrows():
                osm_id = row_building.get('osm_id', None)
                name = row_building.get('name', None)
                building_type = None
                for type_col in ['fclass', 'type', 'building']: # Check multiple potential type columns
                    if type_col in row_building and pd.notna(row_building[type_col]): # Check for existence and non-NaN
                        building_type = row_building[type_col]; break
                try:
                    centroid = row_building.geometry.centroid
                    if centroid and centroid.is_valid:
                        buildings_data.append({
                            'building_name': name, 'osm_id': osm_id, 'building_type': building_type,
                            'latitude': centroid.y, 'longitude': centroid.x
                        })
                except Exception as e: print(f"Error calculating centroid for building {osm_id}: {e}")
            if buildings_data:
                galway_buildings_summary_df = pd.DataFrame(buildings_data)
                print(f"Created DataFrame 'galway_buildings_summary_df' with {len(galway_buildings_summary_df)} buildings.")
                try: display(galway_buildings_summary_df.head())
                except NameError: print(galway_buildings_summary_df.head())
            else: print("No valid building data found to create summary DataFrame.")
        else: print("Clipped 'buildings' GeoDataFrame not found or is empty.")

        # galway_buildings_summary_df1 is initialized above
        if galway_buildings_summary_df is not None and not galway_buildings_summary_df.empty:
            galway_buildings_summary_df1 = galway_buildings_summary_df[galway_buildings_summary_df['building_name'].notnull()].copy()
            if not galway_buildings_summary_df1.empty: # Check if still has rows after filtering
                 galway_buildings_summary_df1 = galway_buildings_summary_df1.sort_values('building_name')
                 building_indices = [f'B{i+1}' for i in range(len(galway_buildings_summary_df1))]
                 galway_buildings_summary_df1.index = building_indices
                 print("\nCreated filtered DataFrame 'galway_buildings_summary_df1' with named buildings:")
                 print(f"Number of named buildings: {len(galway_buildings_summary_df1)}")
                 print("\nFirst few rows of filtered DataFrame:")
                 try: display(galway_buildings_summary_df1.head())
                 except NameError: print(galway_buildings_summary_df1.head())
            else:
                print("No named buildings found after filtering 'galway_buildings_summary_df'. 'galway_buildings_summary_df1' is empty.")
                galway_buildings_summary_df1 = pd.DataFrame() # Ensure it's an empty DataFrame if no named buildings
        else: print("Cannot create filtered DataFrame as galway_buildings_summary_df is None or empty.")

        # --- *** PLOTTING CLIPPED GALWAY DATA *** ---
        print("\nPlotting clipped Galway map layers...")
        fig, ax = plt.subplots(figsize=(18, 18), facecolor='white', dpi=250)

        color_water = '#a8dff5'; color_land = '#f2f4f6'; color_parks = '#cceac4'
        color_buildings_osm = '#d8cabc' 
        color_roads = '#aaaaaa'; color_rail = '#a0a0a0';color_place_text = '#36454F'  
        color_bus_stops_blue = '#1E90FF' 
        ax.set_facecolor(color_land)
        zorder_landuse=1; zorder_water_poly=2; zorder_parks=3; zorder_buildings_layer=4
        zorder_waterways=5; zorder_railways=6; zorder_roads=7;
        zorder_bus_stops_plot = 8; zorder_place_text = 9
        zorder_building_b422_point = 10; zorder_building_b422_text = 11  
        zorder_rahoon_place_point = 10; zorder_rahoon_place_text = 11  
        zorder_boundary = 12
        
        if 'landuse' in galway_gdfs: galway_gdfs['landuse'].plot(ax=ax, column='fclass', categorical=True, cmap='Pastel2', alpha=0.4, zorder=zorder_landuse)
        if 'water_poly' in galway_gdfs: galway_gdfs['water_poly'].plot(ax=ax, color=color_water, edgecolor='none', zorder=zorder_water_poly)
        if 'landuse' in galway_gdfs and 'fclass' in galway_gdfs['landuse'].columns:
            parks_gdf = galway_gdfs['landuse'][galway_gdfs['landuse']['fclass'] == 'park']
            if not parks_gdf.empty: parks_gdf.plot(ax=ax, color=color_parks, edgecolor='none', zorder=zorder_parks)
        if 'buildings' in galway_gdfs: galway_gdfs['buildings'].plot(ax=ax, facecolor=color_buildings_osm, alpha=0.7, lw=0.5, edgecolor=color_buildings_osm, zorder=zorder_buildings_layer)
        if 'waterways' in galway_gdfs: galway_gdfs['waterways'].plot(ax=ax, color=color_water, linewidth=1.0, zorder=zorder_waterways)
        if 'railways' in galway_gdfs:
            galway_gdfs['railways'].plot(ax=ax, color='#ffffff', linewidth=2.0, linestyle='-', zorder=zorder_railways)
            galway_gdfs['railways'].plot(ax=ax, color=color_rail, linewidth=1.0, linestyle='-', zorder=zorder_railways + 0.1)
        if 'roads' in galway_gdfs: galway_gdfs['roads'].plot(ax=ax, color=color_roads, linewidth=0.8, zorder=zorder_roads)

        if bus_stops_gdf is not None and not bus_stops_gdf.empty:
            bus_stops_gdf.plot(ax=ax, color=color_bus_stops_blue, marker='o', markersize=15, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=zorder_bus_stops_plot, label='Bus Stops (All)')
            print(f"Plotted {len(bus_stops_gdf)} bus stops from gstops_df_v1 as blue dots.")
        else: print("No bus stops from gstops_df_v1 to plot.")

        if galway_places_summary_df is not None and not galway_places_summary_df.empty:
            print(f"Plotting {len(galway_places_summary_df)} place names...")
            plotted_place_names_map = set()
            for idx, row_place_plot in galway_places_summary_df.iterrows(): # Renamed row variable
                label = row_place_plot['place_name']; point_x = row_place_plot['longitude']; point_y = row_place_plot['latitude']
                if label not in plotted_place_names_map:
                    ax.text(point_x, point_y + 0.0002, label, fontsize=8, color=color_place_text, ha='center', va='bottom', zorder=zorder_place_text, fontweight='normal', path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground='w')])
                    plotted_place_names_map.add(label)
            print("Place names plotted.")

        if galway_buildings_summary_df1 is not None and not galway_buildings_summary_df1.empty:
            building_point_color = '#FF5733'; building_text_color = '#000000'
            if 'B422' in galway_buildings_summary_df1.index:
                row_b422 = galway_buildings_summary_df1.loc['B422']
                point_x_b422 = row_b422['longitude']; point_y_b422 = row_b422['latitude']
                building_name_b422 = row_b422['building_name']
                plt.scatter(point_x_b422, point_y_b422, s=60, color=building_point_color, edgecolor='black', linewidth=1, alpha=0.9, zorder=zorder_building_b422_point, label=f'Building: {building_name_b422}')
                ax.text(point_x_b422, point_y_b422 + 0.0003, building_name_b422, fontsize=7, color=building_text_color, ha='center', va='bottom', zorder=zorder_building_b422_text, fontweight='bold', path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground='white')])
                print(f"Plotted orange circle and name label for building B422 ('{building_name_b422}').")
            else: print("Building B422 not found in the DataFrame 'galway_buildings_summary_df1'.")
        else: print("DataFrame 'galway_buildings_summary_df1' not available for plotting B422.")
        
        if rahoon_place_id is not None and galway_places_summary_df1 is not None and not galway_places_summary_df1.empty:
            if rahoon_place_id in galway_places_summary_df1.index:
                place_row_rahoon = galway_places_summary_df1.loc[rahoon_place_id]
                point_x_rahoon = place_row_rahoon['longitude']; point_y_rahoon = place_row_rahoon['latitude']
                place_name_label_rahoon = place_row_rahoon['place_name']
                place_point_color = '#9400D3'; place_text_color = '#000000'
                plt.scatter(point_x_rahoon, point_y_rahoon, s=70, color=place_point_color, edgecolor='black', linewidth=1, alpha=0.9, zorder=zorder_rahoon_place_point, label=f'Place: {place_name_label_rahoon}')
                ax.text(point_x_rahoon, point_y_rahoon + 0.00035, place_name_label_rahoon, fontsize=7.5, color=place_text_color, ha='center', va='bottom', zorder=zorder_rahoon_place_text, fontweight='bold', path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground='white')])
                print(f"Plotted distinct circle and name label for place: '{place_name_label_rahoon}' (ID: {rahoon_place_id}).")
            else: print(f"Place with ID '{rahoon_place_id}' (expected to be Rahoon) not found in galway_places_summary_df1.index for plotting.")
        else: print("Rahoon was not identified or 'galway_places_summary_df1' is not available for plotting specific place.")

        if boundary_gdf is not None:
             boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, linestyle='--', zorder=zorder_boundary)

        minx, miny, maxx, maxy = (None, None, None, None)
        if 'roads' in galway_gdfs and not galway_gdfs['roads'].empty:
            minx, miny, maxx, maxy = galway_gdfs['roads'].total_bounds
        elif boundary_gdf is not None: 
            minx, miny, maxx, maxy = boundary_gdf.total_bounds
        
        if all(v is not None for v in [minx, miny, maxx, maxy]):
            margin_factor = 0.02
            margin_x = (maxx - minx) * margin_factor
            margin_y = (maxy - miny) * margin_factor
            ax.set_xlim(minx - margin_x, maxx + margin_x)
            ax.set_ylim(miny - margin_y, maxy + margin_y)
        else:
            print("Warning: Cannot set map bounds as neither roads nor boundary GDF provided sufficient bounds.")
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Galway Map with Bus Stops (from gstops_df_v1)", color='black', fontsize=16)
        plt.legend(loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
     
    except FileNotFoundError as e:
        print(f"\n--- File Error ---\n{e}\nPlease ensure file paths are correct.")
    except ImportError as e:
        print(f"\n--- Import Error Occurred ---\nError: {e}\nPlease ensure required libraries are installed.")
    except ValueError as e:
        print(f"\n--- Value Error ---\n{e}")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---\nError: {e}")
        import traceback
        traceback.print_exc()
    
    # Return the specified DataFrames
    return bus_stops_gdf, galway_places_summary_df1, galway_buildings_summary_df1



place_input = "Galway, Ireland"
shapefiles_dir_input = '/Users/conall/Projects/atlantech-ai-challenge-2025/data/ireland-and-northern-ireland-latest-free.shp' 

shapefile_layers_input = {
    'roads': 'gis_osm_roads_free_1.shp',
    'water_poly': 'gis_osm_water_a_free_1.shp',
    'railways': 'gis_osm_railways_free_1.shp',
    'waterways': 'gis_osm_waterways_free_1.shp',
    'landuse': 'gis_osm_landuse_a_free_1.shp',
    'buildings': 'gis_osm_buildings_a_free_1.shp',
    'places_poly': 'gis_osm_places_a_free_1.shp'
}
bus_stops_gdf, galway_places_summary_df1, galway_buildings_summary_df1 = galway_transport_map(
    place_name=place_input,
    shapefile_base_dir=shapefiles_dir_input,
    shapefile_layers=shapefile_layers_input,
    gstops_df_v1=gstops_df_v1, # created in previous cell
    gvariations_df_v1=gvariations_df_v1 # created in previous cell
)


# ### 6. Normalize Bus Stop Names and Timetable Text

# In[97]:


def normalize_bus_data_text(
    bus_stops_input_df: pd.DataFrame, 
    bus_timetables_input_df: pd.DataFrame
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Normalizes text in specified columns of bus_stops_gdf and bus_timetables DataFrames.
    - Lowercases and strips text.
    - Expands common abbreviations.
    - Removes parentheses and consolidates spaces.
    - Adds new columns with normalized text ('_norm_explicit', '_norm_expanded').

    Args:
        bus_stops_input_df (pd.DataFrame): DataFrame containing bus stop information,
                                           expected to have a 'stop_name' column.
                                           Can be a GeoDataFrame.
        bus_timetables_input_df (pd.DataFrame): DataFrame containing bus timetable information,
                                                expected to have a 'ROUTE' column.

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: A tuple containing the modified:
            - bus_stops_df (or None if input was None)
            - bus_timetables_df (or None if input was None)
    """

    # --- Abbreviation Expansion and Parentheses Removal Helper Function ---
    def expand_abbr_and_remove_paren(text_to_process):
        if pd.isna(text_to_process):
            return text_to_process

        current_text = str(text_to_process) 
        
        abbreviations = {
            r'\brd\b': 'road',       # Using defined strings for regex patterns
            r'\bst\b': 'street',
            r'\blwr\b': 'lower',
            r'\bav\b': 'avenue',
            r'\bave\b': 'avenue',
            r'\bopp\b': 'opposite',
            r'\bind est\b': 'industrial estate',
        }
        
        expanded_text = current_text
        for abbr, expansion in abbreviations.items():
            expanded_text = re.sub(abbr, expansion, expanded_text) 
        text_after_paren_open_removed = re.sub(r'\s*\(\s*', ' ', expanded_text) 
        text_after_paren_close_removed = re.sub(r'\s*\)\s*', ' ', text_after_paren_open_removed)
        
        # Consolidate multiple spaces into one
        final_text = re.sub(r'\s+', ' ', text_after_paren_close_removed).strip()
            
        return final_text

    print("--- Normalizing Text in Bus DataFrames ---")
    bus_stops_df = bus_stops_input_df.copy() if bus_stops_input_df is not None else None
    if bus_stops_input_df is None:
        print("Input bus_stops_df was None. Skipping its normalization.")

    bus_timetables_df = bus_timetables_input_df.copy() if bus_timetables_input_df is not None else None
    if bus_timetables_input_df is None:
        print("Input bus_timetables_df was None. Skipping its normalization.")

    # --- Prepare bus_stops_df 
    if bus_stops_df is not None and not bus_stops_df.empty:
        if 'stop_name' in bus_stops_df.columns:    
            bus_stops_df['stop_name_norm_explicit'] = bus_stops_df['stop_name'].astype(str).str.lower().str.strip()
            bus_stops_df['stop_name_norm_expanded'] = bus_stops_df['stop_name_norm_explicit'].apply(expand_abbr_and_remove_paren)
            print("Text normalization applied to bus_stops_df.")
        else:
            print("Warning: 'stop_name' column not found in bus_stops_df. No normalization applied to it.")
    elif bus_stops_input_df is not None: # Original was not None but became empty (e.g. all rows dropped before call)
        print("Input bus_stops_df is empty. Skipping normalization.")


    # --- Prepare bus_timetables_df ---
    if bus_timetables_df is not None and not bus_timetables_df.empty:
        if 'route' in bus_timetables_df.columns: 
            bus_timetables_df['route_norm_explicit'] = bus_timetables_df['route'].astype(str).str.lower().str.strip()
            bus_timetables_df['route_norm_expanded'] = bus_timetables_df['route_norm_explicit'].apply(expand_abbr_and_remove_paren)
            print("Text normalization applied to bus_timetables_df.")
        else:
            print("Warning: 'route' column not found in bus_timetables_df. No normalization applied to it.")
    elif bus_timetables_input_df is not None:
        print("Input bus_timetables_df is empty. Skipping normalization.")
        
    print("--- Text Normalization Finished ---")
    return bus_stops_df, bus_timetables_df


if __name__ == '__main__':


    # Call the normalization function
    bus_stops_gdf, bus_timetables = normalize_bus_data_text(
        bus_stops_gdf, 
        bus_timetables
    )

    print("\nAfter normalization:")
    if bus_stops_gdf is not None:
        print("Normalized Bus Stops DF Head:")
        try:display(bus_stops_gdf.head(2))
        except ImportError: print(bus_stops_gdf.head())
        
    if bus_timetables is not None:
        print("\nNormalized Bus Timetables DF Head:")
        try:display(bus_timetables.head(2))
        except ImportError: print(bus_timetables.head())

    # Save bus_stops_gdf
    if bus_stops_gdf is not None:
        bus_stops_gdf_path = os.path.join(artifact_dir, 'bus_stops_gdf.csv')
        try:
            bus_stops_gdf.to_csv(bus_stops_gdf_path, index=False)
            print(f"Normalized bus_stops_gdf saved to {bus_stops_gdf_path}")
        except Exception as e:
            print(f"Error saving bus_stops_gdf to {bus_stops_gdf_path}: {e}")
        
    # Save bus_timetables and replace if it already exists
    if bus_timetables is not None:
        bus_timetables_path = os.path.join(artifact_dir, 'bus_timetables.csv')
        try:
            bus_timetables.to_csv(bus_timetables_path, index=False)
            print(f"Normalized bus_timetables saved to {bus_timetables_path}")
        except Exception as e:
            print(f"Error saving bus_timetables to {bus_timetables_path}: {e}")


# ### 7. Linking Timetable Descriptions to Bus Stop IDs via Multi-Stage Matching

# In[98]:


def map_timetable_routes_to_stops(
    bus_timetables_input_df: pd.DataFrame,
    bus_stops_input_gdf: pd.DataFrame # GeoDataFrame is a subclass of DataFrame
) -> pd.DataFrame | None:
    """
    Maps route descriptions in bus_timetables to stop_ids from bus_stops_gdf using
    a multi-stage matching process (exact, fuzzy, token overlap).
    Adds mapping details (stop_id_mapped, match_method, etc.) as new columns
    to the bus_timetables DataFrame.

    Args:
        bus_timetables_input_df (pd.DataFrame): DataFrame containing bus timetable information,
                                                expected to have 'ROUTE_norm_expanded'.
        bus_stops_input_gdf (pd.DataFrame): GeoDataFrame (or DataFrame) with bus stop details,
                                            expected to have 'stop_name_norm_expanded', 
                                            'stop_id', and 'stop_name'.

    Returns:
        pd.DataFrame | None: A new DataFrame based on bus_timetables_input_df with added 
                              mapping columns. Rows where 'stop_id_mapped' could not be 
                              determined (and is NA after processing) are dropped.
                              Returns None if bus_timetables_input_df is None.
    """

    if bus_timetables_input_df is None:
        print("Input bus_timetables_input_df is None. Cannot perform mapping.")
        return None
    
    # Work on a copy to avoid modifying the original DataFrame passed by the caller
    bus_timetables_df = bus_timetables_input_df.copy()

    # --- MATCHING FUNCTION (Exact -> Fuzzy -> Token Overlap) ---
    # This is the exact helper function from your snippet, now an inner function.
    def find_stop_id_expanded_match(route_norm_expanded_to_match, stops_df_internal):
        # Renamed stops_df to stops_df_internal to avoid confusion with outer scope
        if pd.isna(route_norm_expanded_to_match):
            return None, "Input route_norm_expanded is NaN", None, np.nan, np.nan

        if not isinstance(stops_df_internal, pd.DataFrame) or stops_df_internal.empty:
            return None, "stops_df_internal (bus_stops_gdf) is invalid or empty", None, np.nan, np.nan

        required_cols = ['stop_name_norm_expanded', 'stop_id', 'stop_name']
        if not all(col in stops_df_internal.columns for col in required_cols):
            missing = [col for col in required_cols if col not in stops_df_internal.columns]
            return None, f"Missing required columns in stops_df_internal: {missing}", None, np.nan, np.nan

        matched_stop_id = None
        match_method = "No Initial Match" 
        matched_original_stop_name_in_gdf = None
        fuzz_ratio_score = np.nan
        fuzz_wratio_score = np.nan

        # Stage 1: Exact match
        exact_match_gdf_internal = stops_df_internal[stops_df_internal['stop_name_norm_expanded'] == route_norm_expanded_to_match]
        if not exact_match_gdf_internal.empty:
            matched_stop_id = exact_match_gdf_internal.iloc[0]['stop_id']
            matched_original_stop_name_in_gdf = exact_match_gdf_internal.iloc[0]['stop_name']
            match_method = "Exact Expanded Name" + (" (Multiple GDF matches, took first)" if len(exact_match_gdf_internal) > 1 else "")
            fuzz_ratio_score, fuzz_wratio_score = 100, 100
            return matched_stop_id, match_method, matched_original_stop_name_in_gdf, fuzz_ratio_score, fuzz_wratio_score

        # Stage 2: Fuzzy match
        choices = stops_df_internal['stop_name_norm_expanded'].dropna().tolist()
        if choices: # Ensure choices is not empty
            # Using score_cutoff =88. We can tune this  parameter to improve matching
            best_match_wratio_tuple = process.extractOne(route_norm_expanded_to_match, choices, scorer=fuzz.WRatio, score_cutoff=88)
            if best_match_wratio_tuple:
                best_matched_expanded_name_in_gdf = best_match_wratio_tuple[0]
                fuzz_wratio_score = best_match_wratio_tuple[1]
                fuzz_ratio_score = fuzz.ratio(route_norm_expanded_to_match, best_matched_expanded_name_in_gdf)
                
                gdf_row_for_best_match = stops_df_internal[stops_df_internal['stop_name_norm_expanded'] == best_matched_expanded_name_in_gdf]
                if not gdf_row_for_best_match.empty:
                    matched_stop_id = gdf_row_for_best_match.iloc[0]['stop_id']
                    matched_original_stop_name_in_gdf = gdf_row_for_best_match.iloc[0]['stop_name']
                    match_method = f"Fuzzy Expanded Name (WRatio: {fuzz_wratio_score:.0f})"
                    return matched_stop_id, match_method, matched_original_stop_name_in_gdf, fuzz_ratio_score, fuzz_wratio_score
                else: 
                    match_method = "Fuzzy Match Found but GDF Row Missing" 
                    # keeping scores as they are relevant from the fuzzy match attempt
                    return None, match_method, None, fuzz_ratio_score, fuzz_wratio_score 
        
        # Stage 3: Token-based Overlap Match
        if matched_stop_id is None and isinstance(route_norm_expanded_to_match, str) and len(route_norm_expanded_to_match.strip()) > 0:
            route_tokens = set(route_norm_expanded_to_match.split())
            if not route_tokens: # Check if route_tokens became empty (e.g., if input was just spaces)
                 return None, "No Final Match (Empty Route Tokens)", None, np.nan, np.nan

            best_overlap_score = 0
            candidate_stop_id = None
            candidate_original_name = None
            best_gdf_name_for_token_match = None
            min_required_common_tokens = 1 

            for index, row in stops_df_internal.iterrows():
                gdf_stop_name_expanded = row['stop_name_norm_expanded']
                if pd.isna(gdf_stop_name_expanded): continue
                
                gdf_tokens = set(gdf_stop_name_expanded.split())
                if not gdf_tokens: continue
                    
                common_tokens = route_tokens.intersection(gdf_tokens)
                current_overlap_score = len(common_tokens)
                
                if current_overlap_score >= min_required_common_tokens:
                    if current_overlap_score > best_overlap_score:
                        best_overlap_score = current_overlap_score
                        candidate_stop_id = row['stop_id']
                        candidate_original_name = row['stop_name']
                        best_gdf_name_for_token_match = gdf_stop_name_expanded
                    elif current_overlap_score == best_overlap_score:
                        if candidate_original_name is None or \
                           (best_gdf_name_for_token_match and gdf_stop_name_expanded and \
                            len(gdf_stop_name_expanded) < len(best_gdf_name_for_token_match)): # Ensure gdf_stop_name_expanded is not None
                            candidate_stop_id = row['stop_id']
                            candidate_original_name = row['stop_name']
                            best_gdf_name_for_token_match = gdf_stop_name_expanded
                                
            if candidate_stop_id is not None:
                matched_stop_id = candidate_stop_id
                matched_original_stop_name_in_gdf = candidate_original_name
                match_method = f"Token Overlap Match (Score: {best_overlap_score})"
                return matched_stop_id, match_method, matched_original_stop_name_in_gdf, np.nan, np.nan

        final_match_method = "No Final Match" if match_method == "No Initial Match" else match_method
        return None, final_match_method, None, np.nan, np.nan


    # --- APPLYING THE MATCHING TO bus_timetables_df ---
    print("\n--- Starting Stop ID Mapping for Timetables ---")
    if (not bus_timetables_df.empty and
        'route_norm_expanded' in bus_timetables_df.columns and
        bus_stops_input_gdf is not None and not bus_stops_input_gdf.empty and
        'stop_name_norm_expanded' in bus_stops_input_gdf.columns and 
        'stop_name' in bus_stops_input_gdf.columns and 
        'stop_id' in bus_stops_input_gdf.columns):

        print("Mapping route_norm_expanded to stop_id (Exact -> Fuzzy -> Token Overlap)...")
        
        match_results_tuples = bus_timetables_df['route_norm_expanded'].apply(
            lambda x: find_stop_id_expanded_match(x, bus_stops_input_gdf.copy()) 
        )
        
        bus_timetables_df['stop_id_mapped'] = [res[0] for res in match_results_tuples]
        bus_timetables_df['match_method'] = [res[1] for res in match_results_tuples]
        bus_timetables_df['matched_stop_name_in_gdf'] = [res[2] for res in match_results_tuples]
        bus_timetables_df['fuzz_ratio_score'] = [res[3] for res in match_results_tuples] 
        bus_timetables_df['fuzz_wratio_score'] = [res[4] for res in match_results_tuples]

        print("Mapping application complete.")
        print("\n--- bus_timetables with mapped stop_id (Raw results before final cleanup) ---")
        
        # Assembling display_cols
        display_cols_base = ['route', 'route_norm_expanded', 'stop_id_mapped', 'match_method', 
                             'matched_stop_name_in_gdf', 'fuzz_ratio_score', 'fuzz_wratio_score']
        if 'route_id' in bus_timetables_df.columns:
            display_cols_base.insert(2, 'route_id') 
        
        final_display_cols = [col for col in display_cols_base if col in bus_timetables_df.columns]
        
        if final_display_cols: # Check if there are any columns to display
            try: display(bus_timetables_df[final_display_cols].head())
            except NameError: print(bus_timetables_df[final_display_cols].head()) # Fallback

        if 'match_method' in bus_timetables_df.columns:
            print("\nMatch Method Distribution:")
            try: display(bus_timetables_df['match_method'].value_counts(dropna=False))
            except NameError: print(bus_timetables_df['match_method'].value_counts(dropna=False))

        unmapped_count = bus_timetables_df['stop_id_mapped'].isna().sum()
        print(f"\nNumber of routess not mapped to a stop_id (before final cleanup): {unmapped_count} out of {len(bus_timetables_df)}")
        
        if unmapped_count > 0 and unmapped_count < len(bus_timetables_df) : # Only show sample if some are mapped
            print("Sample of unmapped routes:")
            unmapped_sample_cols_base = ['route', 'route_norm_expanded', 'match_method']
            if 'route_id' in bus_timetables_df.columns: unmapped_sample_cols_base.append('route_id')
            
            actual_unmapped_cols = [col for col in unmapped_sample_cols_base if col in bus_timetables_df.columns]
            if actual_unmapped_cols:
                try: display(bus_timetables_df[bus_timetables_df['stop_id_mapped'].isna()][actual_unmapped_cols].head(10))
                except NameError: print(bus_timetables_df[bus_timetables_df['stop_id_mapped'].isna()][actual_unmapped_cols].head(10))
        
        # --- Final Cleanup as per original code ---
        # Replace empty strings that might have come from mapping with proper NA
        if 'stop_id_mapped' in bus_timetables_df.columns:
            bus_timetables_df['stop_id_mapped'] = bus_timetables_df['stop_id_mapped'].replace('', pd.NA)
            original_row_count = len(bus_timetables_df)
            bus_timetables_df = bus_timetables_df.dropna(subset=['stop_id_mapped'])
            rows_dropped = original_row_count - len(bus_timetables_df)
            print(f"\nDropped {rows_dropped} rows where 'stop_id_mapped' was NA after processing.")
        else:
            print("\n'stop_id_mapped' column not created. Skipping final cleanup based on it.")

    else:
        print("\nCannot perform mapping. Prerequisites not met (check input DataFrames and required columns).")
        print(f"bus_timetables_df empty or missing 'route_norm_expanded': {bus_timetables_df.empty if isinstance(bus_timetables_df, pd.DataFrame) else 'is None'}")
        if isinstance(bus_timetables_df, pd.DataFrame): print(f"'route_norm_expanded' in bus_timetables_df: {'route_norm_expanded' in bus_timetables_df.columns}")
        print(f"bus_stops_input_gdf empty or missing required columns: {bus_stops_input_gdf.empty if isinstance(bus_stops_input_gdf, pd.DataFrame) else 'is None'}")
        if isinstance(bus_stops_input_gdf, pd.DataFrame):
             print(f"'stop_name_norm_expanded' in bus_stops_input_gdf: {'stop_name_norm_expanded' in bus_stops_input_gdf.columns}")
             print(f"'stop_name' in bus_stops_input_gdf: {'stop_name' in bus_stops_input_gdf.columns}")
             print(f"'stop_id' in bus_stops_input_gdf: {'stop_id' in bus_stops_input_gdf.columns}")
        # bus_timetables_df will be returned as a copy of the input, possibly without new columns.
        
    print("--- Stop ID Mapping for Timetables Finished ---")
    return bus_timetables_df


if __name__ == '__main__':

    print("Bus stops(we use for mapping):")
    display(bus_stops_gdf.head(5))
    
    print("Bus Timetables before mapping:")
    display(bus_timetables.head(5))

    # Call the mapping function
    bus_timetables = map_timetable_routes_to_stops(
        bus_timetables,
        bus_stops_gdf
    )
    print("Bus Timetables after mapping:")
    display(bus_timetables.head(5))


# ### 8. Construct the Multi-Modal Transport Graph for Galway
# 
# Network structure with nodes (places, buildings, bus stops) and edges (walking connections, bus routes)
# 
# add places, buildings and bus stops Nodes 
# 
# add Access/Egress Edges - walking connections i.e., place-TO-nearby-bus-stop / nearby-bus-stop-TO-building
# 
# add Directed Transit Edges - bus route between consecutive stops
# 
# **User Input:** MAX_ACCESS_DISTANCE_METERS = 800

# In[99]:


def create_galway_transport_graph(
    galway_places_summary_df1_input: pd.DataFrame,
    galway_buildings_summary_df1_input: pd.DataFrame,
    bus_stops_gdf_input: pd.DataFrame, # GeoDataFrame
    bus_timetables_input: pd.DataFrame,
    MAX_ACCESS_DISTANCE_METERS_input: int = 800 # Default value 
) -> nx.DiGraph | None:
    """
    Constructs a multi-modal transport graph for Galway.

    Args:
        galway_places_summary_df1_input (pd.DataFrame): DataFrame of places.
        galway_buildings_summary_df1_input (pd.DataFrame): DataFrame of buildings.
        bus_stops_gdf_input (pd.DataFrame): GeoDataFrame of bus stops.
        bus_timetables_input (pd.DataFrame): DataFrame of bus timetables with mapped stop_ids.
        MAX_ACCESS_DISTANCE_METERS_input (int): Max distance for access/egress edges.

    Returns:
        nx.DiGraph | None: The constructed transport graph G, or None if critical inputs are missing.
    """
    print("\n--- Starting Transport Graph Construction ---")

    # --- Input Validations---
    if galway_places_summary_df1_input is None or galway_places_summary_df1_input.empty:
        print("Error: galway_places_summary_df1_input is None or empty. Cannot build graph.")
        return None
    if galway_buildings_summary_df1_input is None or galway_buildings_summary_df1_input.empty:
        print("Error: galway_buildings_summary_df1_input is None or empty. Cannot build graph.")
        return None
    if bus_stops_gdf_input is None or bus_stops_gdf_input.empty:
        print("Error: bus_stops_gdf_input is None or empty. Cannot build graph.")
        return None
    if bus_timetables_input is None or bus_timetables_input.empty:
        print("Error: bus_timetables_input is None or empty. Cannot build graph.")
        return None
        
    galway_places_summary_df1 = galway_places_summary_df1_input.copy()
    galway_buildings_summary_df1 = galway_buildings_summary_df1_input.copy()
    bus_stops_gdf = bus_stops_gdf_input.copy()
    bus_timetables = bus_timetables_input.copy()
    MAX_ACCESS_DISTANCE_METERS = MAX_ACCESS_DISTANCE_METERS_input # Use the input parameter

    # --- Helper Function for Haversine Distance ---
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers
        try:
            # It is to ensure inputs are explicitly converted to float before radians
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        except (ValueError, TypeError): # Handle cases where conversion to float might fail
            return float('inf') # Return infinity if coordinates are invalid
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance_km = R * c
        return distance_km * 1000 # Convert to meters

    # --- 1. Initialize Graph ---
    G = nx.DiGraph()
    print("Graph initialized.")

    # --- Step 1a: Add Place Nodes ---
    print("\nAdding place nodes (general POIs)...")
  
    for index, row in galway_places_summary_df1.iterrows():
        place_node_id = row['place_name'] # place_name is an attribute in galway_places_summary_df1
        if pd.notna(place_node_id): 
             G.add_node(place_node_id, type='place', name=row['place_name'], 
                        latitude=row.get('latitude'), longitude=row.get('longitude'))
    print(f"Nodes after general places: {G.number_of_nodes()}")

    # --- Step 1b: Add Building Nodes---
    print("\nAdding building nodes...")
    for index, row in galway_buildings_summary_df1.iterrows():
        building_node_id = row['building_name'] # building_name is unique and an attribute in galway_buildings_summary_df1
        if pd.notna(building_node_id):
            G.add_node(building_node_id, type='building', name=row['building_name'], 
                       osm_id=row.get('osm_id'), building_type=row.get('building_type'), 
                       latitude=row.get('latitude'), longitude=row.get('longitude'))
    print(f"Nodes after adding buildings: {G.number_of_nodes()}")

    # --- Step 1c: Add Unique Bus Stop Nodes ---
    print("\nAdding unique bus stop nodes...")
    added_stop_ids = set() # To keep track of added stop_ids 
    for index, row in bus_stops_gdf.iterrows():
        stop_id = row['stop_id']
        if pd.notna(stop_id) and stop_id not in added_stop_ids: 
            G.add_node(stop_id, type='bus_stop', name=row.get('stop_name'), 
                       latitude=row.get('stop_lat'), longitude=row.get('stop_lon'),
                       direction=row.get('direction'), original_route_id=row.get('route_id'), 
                       geometry=row.get('geometry'), norm_explicit=row.get('stop_name_norm_explicit'), 
                       norm_expanded=row.get('stop_name_norm_expanded'))
            added_stop_ids.add(stop_id)
    print(f"Total nodes after bus stops: {G.number_of_nodes()}")

    # --- Step 2: Add Access/Egress Edges
    print("\nAdding access/egress edges...")
    access_edge_count = 0
    # Pre-filter nodes
    place_nodes_data = {node_id: data for node_id, data in G.nodes(data=True) if data.get('type') == 'place'}
    building_nodes_data = {node_id: data for node_id, data in G.nodes(data=True) if data.get('type') == 'building'}
    bus_stop_nodes_data_for_access = {node_id: data for node_id, data in G.nodes(data=True) if data.get('type') == 'bus_stop'} # Renamed to avoid conflict

    for place_node_id, place_data in place_nodes_data.items():
        place_lat = place_data.get('latitude'); place_lon = place_data.get('longitude')
        if place_lat is None or place_lon is None or pd.isna(place_lat) or pd.isna(place_lon): continue
        for stop_node_id, stop_data in bus_stop_nodes_data_for_access.items():
            stop_lat = stop_data.get('latitude'); stop_lon = stop_data.get('longitude')
            if stop_lat is None or stop_lon is None or pd.isna(stop_lat) or pd.isna(stop_lon): continue
            walking_distance_m = haversine(place_lat, place_lon, stop_lat, stop_lon)
            if walking_distance_m <= MAX_ACCESS_DISTANCE_METERS: # Using the input parameter
                edge_attrs = {'type':'access_egress', 'mode':'walk', 'distance_m': walking_distance_m}
                G.add_edge(place_node_id, stop_node_id, **edge_attrs)
                G.add_edge(stop_node_id, place_node_id, **edge_attrs)
                access_edge_count += 2

    for building_node_id, building_data in building_nodes_data.items():
        building_lat = building_data.get('latitude'); building_lon = building_data.get('longitude')
        if building_lat is None or building_lon is None or pd.isna(building_lat) or pd.isna(building_lon): continue
        for stop_node_id, stop_data in bus_stop_nodes_data_for_access.items():
            stop_lat = stop_data.get('latitude'); stop_lon = stop_data.get('longitude')
            if stop_lat is None or stop_lon is None or pd.isna(stop_lat) or pd.isna(stop_lon): continue
            walking_distance_m = haversine(building_lat, building_lon, stop_lat, stop_lon)
            if walking_distance_m <= MAX_ACCESS_DISTANCE_METERS: # Using the input parameter
                edge_attrs = {'type':'access_egress', 'mode':'walk', 'distance_m': walking_distance_m}
                G.add_edge(building_node_id, stop_node_id, **edge_attrs)
                G.add_edge(stop_node_id, building_node_id, **edge_attrs)
                access_edge_count += 2
    print(f"Added {access_edge_count} access/egress edges in total.")

    # --- Step 3: Add Directed Transit Edges ---
    print("\nAdding directed transit edges...")
    transit_edge_count = 0
    # filter for valid bus stop nodes already in the graph
    valid_graph_stop_node_ids = {node_id for node_id, data in G.nodes(data=True) if data.get('type') == 'bus_stop'}
    print(f"Debug: Found {len(valid_graph_stop_node_ids)} valid bus_stop nodes in the graph for transit edges.")
    print(f"Debug: bus_timetables has {len(bus_timetables)} rows for transit edge creation.")

    # verify required columns exist in bus_timetables
    required_timetable_cols = ['route_id', 'stop_order_on_route', 'stop_id_mapped']
    if not all(col in bus_timetables.columns for col in required_timetable_cols):
        missing_cols = [col for col in required_timetable_cols if col not in bus_timetables.columns]
        print(f"Warning: Missing required columns in bus_timetables for transit edges: {missing_cols}. Skipping transit edge creation.")
    else:
        for route_id_timetable, group in bus_timetables.groupby('route_id'):
            route_stops = group.sort_values(by='stop_order_on_route')
            for i in range(len(route_stops) - 1):
                from_stop_id_mapped = route_stops.iloc[i]['stop_id_mapped']
                to_stop_id_mapped = route_stops.iloc[i+1]['stop_id_mapped']

                # check if mapped stop_ids actually exist as nodes in our graph
                from_node_exists = from_stop_id_mapped in valid_graph_stop_node_ids
                to_node_exists = to_stop_id_mapped in valid_graph_stop_node_ids

                if from_node_exists and to_node_exists:
                    from_stop_node_data = G.nodes[from_stop_id_mapped]
                    to_stop_node_data = G.nodes[to_stop_id_mapped]
                    from_lat = from_stop_node_data.get('latitude'); from_lon = from_stop_node_data.get('longitude')
                    to_lat = to_stop_node_data.get('latitude'); to_lon = to_stop_node_data.get('longitude')

                    if None not in [from_lat, from_lon, to_lat, to_lon] and \
                       all(pd.notna(coord) for coord in [from_lat, from_lon, to_lat, to_lon]):
                        segment_distance_m = haversine(from_lat, from_lon, to_lat, to_lon)
                        edge_attrs = {'type':'transit', 'route_id':route_id_timetable, 
                                      'hop_count':1, 'distance_m':segment_distance_m}
                        G.add_edge(from_stop_id_mapped, to_stop_id_mapped, **edge_attrs)
                        transit_edge_count += 1
    print(f"Added {transit_edge_count} directed transit edges.")

    # --- Node Relabeling ---
    old_name = "Portershed a Dó" # old name
    new_name = "Portershed"       # new name
    mapping = {old_name: new_name}

    if G.has_node(old_name):
        print(f"Node '{old_name}' exists in the graph before relabeling.")
        G = nx.relabel_nodes(G, mapping, copy=False) # copy=False modifies in place
        print(f"Node '{old_name}' has been relabeled to '{new_name}'.")
        if G.has_node(new_name) and not G.has_node(old_name):
            print(f"Verification successful: '{new_name}' is now in the graph, and '{old_name}' is not.")
        else:
            print(f"Verification failed or '{old_name}' was not found initially or relabeling error.")
    else:
        print(f"Node '{old_name}' was not found in the graph. No relabeling done for this specific node.")

    # --- Final Graph Summary ---
    print("\n--- Graph Construction Complete ---")
    print(f"Total nodes in graph: {G.number_of_nodes()}")
    print(f"Total edges in graph: {G.number_of_edges()}")
    node_types_list = [data.get('type', 'Unknown') for node, data in G.nodes(data=True)]
    if node_types_list:
        print("\nNode type counts:\n", pd.Series(node_types_list).value_counts())
    else:
        print("\nNode type counts: No nodes to summarize.")
        
    edge_summary = []
    if G.number_of_edges() > 0:
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            edge_summary.append(f"{edge_type}_{data.get('mode', '')}" if edge_type == 'access_egress' else edge_type)
        if edge_summary:
            print("\nEdge type counts:\n", pd.Series(edge_summary).value_counts())
        else:
            print("\nEdge type counts: No edges to summarize (or edges lack 'type' attribute).")
    else:
        print("\nEdge type counts: No edges in the graph.")

    print("\n--- Explicit check of G.edges(data=True) (Sample) ---")
    all_edges_with_data = list(G.edges(data=True))
    if not all_edges_with_data: 
        print("G.edges(data=True) is EMPTY.")
    else:
        
        print(f"Found {len(all_edges_with_data)} edges with data. Sample (random up to 3):")
    
        sample_size_edges = min(3, len(all_edges_with_data))
        if all_edges_with_data: # Check if list is not empty before sampling
            random_sample_edges = random.sample(all_edges_with_data, sample_size_edges)
            for edge_tuple in random_sample_edges:
                print(edge_tuple)
        else:
            print("No edges to sample.")

    print("\n--- Explicit check of G.nodes(data=True) (Sample) ---")
    all_nodes_with_data = list(G.nodes(data=True))
    if not all_nodes_with_data:
        print("G.nodes(data=True) is EMPTY.")
    else:
        print(f"Found {len(all_nodes_with_data)} nodes with data. Sample (random up to 3):")
    
        sample_size_nodes = min(3, len(all_nodes_with_data))
        if all_nodes_with_data: 
            random_sample_nodes = random.sample(all_nodes_with_data, sample_size_nodes)
            for node_tuple in random_sample_nodes:
                print(node_tuple)
        else:
            print("No nodes to sample.")
            
    return G


if __name__ == '__main__':
    
    MAX_ACCESS_DIST = 800

    # Call the graph creation function
    G = create_galway_transport_graph(
        galway_places_summary_df1_input=galway_places_summary_df1,
        galway_buildings_summary_df1_input=galway_buildings_summary_df1,
        bus_stops_gdf_input=bus_stops_gdf,
        bus_timetables_input=bus_timetables,
        MAX_ACCESS_DISTANCE_METERS_input=MAX_ACCESS_DIST
    )

    if G:
        print(f"\n--- Example: Graph successfully created ---")
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    else:
        print("\n--- Example: Graph creation failed (returned None) ---")


# ### 9. Finding Bus Stops within Proximity to Places and Buildings

# In[100]:


# ---  get_nearby_stops  ---
def get_nearby_stops(graph, poi_node_id, max_distance):
    """
    Finds bus stops connected to a POI node(place or building) via 'access_egress' edges
    within a specified maximum distance, and returns their IDs and distances.
    """
    nearby_stops_info = [] # list to store dictionaries

    if not graph.has_node(poi_node_id):
        print(f"Warning: POI node '{poi_node_id}' not found in the graph.")
        return nearby_stops_info # Return empty list
    
    for u, v, data in graph.out_edges(poi_node_id, data=True):
        edge_type = data.get('type')
        edge_distance = data.get('distance_m', float('inf')) #  infinity if no distance

        # Check if the edge is an access/egress edge
        if edge_type == 'access_egress':
            # Check if the connected node 'v' is a bus stop
            if graph.has_node(v) and graph.nodes[v].get('type') == 'bus_stop':
                # Check if the distance is within the threshold
                if edge_distance <= max_distance:
                    # Add a dictionary with stop_id and distance
                    nearby_stops_info.append({'stop_id': v, 'distance_m': edge_distance}) 
            
    return poi_node_id, nearby_stops_info


# # For tool1: journey and accessibility check:
# places_ids_list= ['Ballybrit', 'Rahoon', 'Renmore', 'Shantallow', 'Knocknacarragh', 'Doughiska', 'NUIG Education', ]
# buildings_ids_list= ['Salthill Lodge', 'Eyre Square Centre', 'Galway Cathedral', 'Merchants Quay', 'Spanish Arch Car Park', 'Merlin Park', 'Portershed' ]


if __name__ == '__main__':
    print("--- Running Nearby Stops Identification for Key POIs ---")

    if 'G' not in locals() or not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
        print("Error: Graph G is not defined or is empty. Cannot proceed with nearby stops identification.")
    else:
        print("Graph G found. Proceeding...")


    # Parameters 
    MAX_ACCESS_DISTANCE_METERS = 800
    place_node_id = "NUIG Education"  # This is the specific node ID for Rahoon
    building_node_id = "Ballybrit" # This is the specific node ID for Portershed


    # --- Check if these specific nodes exist  ---
    if G.has_node(place_node_id):
        print(f"Node '{place_node_id}' place_node_id FOUND in graph G.")
    else:
        print(f"Warning: Node '{place_node_id}' place_node_id NOT FOUND in graph G.")
        
    if G.has_node(building_node_id):
        print(f"Node '{building_node_id}' building_node_id FOUND in graph G.")
    else:
        print(f"Warning: Node '{building_node_id}' building_node_id NOT FOUND in graph G.")

    # --- Call get_nearby_stops for Rahoon ---
    place_node_id, place_nearby_stops = get_nearby_stops(G, place_node_id, MAX_ACCESS_DISTANCE_METERS)
    
    # --- Call get_nearby_stops for Portershed ---
    building_node_id, building_nearby_stops = get_nearby_stops(G, building_node_id, MAX_ACCESS_DISTANCE_METERS)
    
    # --- Process and print results (using the node_id variables in f-strings) ---
    place_nearby_stop_ids = [info['stop_id'] for info in place_nearby_stops]
    building_nearby_stop_ids = [info['stop_id'] for info in building_nearby_stops]
    
    print(f"\nProximity Threshold Used: {MAX_ACCESS_DISTANCE_METERS} meters.")
    
    print(f"\nNearby stops for '{place_node_id}':")
    if place_nearby_stops:
        for stop_info in place_nearby_stops:
            print(f"  - Stop ID: {stop_info['stop_id']}, Distance: {stop_info['distance_m']:.0f}m")
    else:
        print(f"  No nearby stops found for '{place_node_id}' within the threshold.")
        
    print(f"\nNearby stops for '{building_node_id}':")
    if building_nearby_stops:
        for stop_info in building_nearby_stops:
            print(f"  - Stop ID: {stop_info['stop_id']}, Distance: {stop_info['distance_m']:.0f}m")
    else:
        print(f"  No nearby stops found for '{building_node_id}' within the threshold.")
        
    print("\n--- End of Nearby Stops Identification ---")


# ### 10. Direct Transit Connections (Place to Building)

# In[101]:


try:
    from IPython.display import display
except ImportError:
    display = print # Fallback to simple print 

def get_enriched_transit_connections(
    G_input: nx.DiGraph, 
    bus_timetables_input: pd.DataFrame, 
    origin_poi_name_param: str, 
    origin_nearby_stops_info_param: list, 
    destination_poi_name_param: str, 
    destination_nearby_stops_info_param: list
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:

    print(f"\n--- Starting Enriched Transit Connection Analysis ---")
    print(f"From: {origin_poi_name_param} To: {destination_poi_name_param}")


    G = G_input # Graph reference 
    bus_timetables = bus_timetables_input.copy() if bus_timetables_input is not None else None
    
    # These will be used by the inner function and subsequent processing.
    origin_poi_name = origin_poi_name_param
    origin_nearby_stops_info = origin_nearby_stops_info_param
    destination_poi_name = destination_poi_name_param
    destination_nearby_stops_info = destination_nearby_stops_info_param

    # --- Inner Function: direct_transit_conn_between_places (Copied as is) ---
    def direct_transit_conn_between_places(G_inner, bus_timetables_inner, origin_poi_name_inner, 
                                           origin_nearby_stops_info_inner, destination_poi_name_inner, 
                                           destination_nearby_stops_info_inner):
        """
        Analyzes direct public transit connections between two sets of nearby stops 
        for given points of interests (POIs).

        Args:
            G (nx.DiGraph): The NetworkX graph containing transit network data. 
                            Edges should have 'type' ('transit'), 'route_id', and 'distance_m'.
            bus_timetables (pd.DataFrame): DataFrame with bus timetable information, including
                                        'route_id', 'stop_id_mapped', and 'stop_order_on_route'.
            origin_poi_name (str): Name of the origin POI (e.g., "Rahoon").
            origin_nearby_stops_info (list): List of dictionaries for stops near the origin POI.
                                            Each dict: {'stop_id': str, 'distance_m': float}
            destination_poi_name (str): Name of the destination POI (e.g., "Portershed").
            destination_nearby_stops_info (list): List of dictionaries for stops near the destination POI.
                                                Each dict: {'stop_id': str, 'distance_m': float}

        Returns:
            tuple: A tuple containing two DataFrames:
                - connections_df (pd.DataFrame): DataFrame of direct transit connections found.
                - no_connection_df (pd.DataFrame): DataFrame of routes serving origin stops 
                                                    but not connecting to destination stops in sequence.
                Returns (None, None) if critical input errors occur.
        """

        if not origin_nearby_stops_info_inner or not destination_nearby_stops_info_inner:
            print(f"Warning: Input list 'origin_nearby_stops_info_inner' or 'destination_nearby_stops_info_inner' is empty.")

        origin_target_stop_ids = [info['stop_id'] for info in origin_nearby_stops_info_inner]
        destination_target_stop_ids = [info['stop_id'] for info in destination_nearby_stops_info_inner]

        print(f"Origin ({origin_poi_name_inner}) Target Stops: {origin_target_stop_ids}")
        print(f"Destination ({destination_poi_name_inner}) Target Stops: {destination_target_stop_ids}")

        direct_transit_connections = []
        origin_routes_no_destination_connection = []

        required_cols = ['route_id', 'stop_id_mapped', 'stop_order_on_route']
        if not isinstance(bus_timetables_inner, pd.DataFrame) or not all(col in bus_timetables_inner.columns for col in required_cols):
            print(f"Error: 'bus_timetables_inner' DataFrame is not valid or is missing required columns: {required_cols}")
            return pd.DataFrame(), pd.DataFrame() # Return empty DFs for robustness
        if not hasattr(G_inner, 'edges'):
            print("Error: NetworkX graph 'G_inner' is not valid.")
            return pd.DataFrame(), pd.DataFrame()
        if not origin_target_stop_ids or not destination_target_stop_ids:
            print(f"Error: Either Origin ({origin_poi_name_inner}) or Destination ({destination_poi_name_inner}) target stop lists are empty. Cannot proceed.")
            return pd.DataFrame(columns=['origin_poi', 'destination_poi', 'route_id', 'origin_stop_id', 
                                         'destination_stop_id', 'origin_stop_order', 'destination_stop_order', 
                                         'hops', 'transit_distance_m']), \
                   pd.DataFrame(columns=['origin_stop_id', 'route_id', 'message'])

        print(f"\nAnalyzing routes from {origin_poi_name_inner} area stops towards {destination_poi_name_inner} area stops...")

        for r_stop_id in origin_target_stop_ids:
            routes_serving_r_stop_df = bus_timetables_inner[bus_timetables_inner['stop_id_mapped'] == r_stop_id]
            if routes_serving_r_stop_df.empty: continue
            unique_routes_for_this_r_stop = routes_serving_r_stop_df['route_id'].unique()

            for route_id_val in unique_routes_for_this_r_stop:
                route_sequence_df = bus_timetables_inner[bus_timetables_inner['route_id'] == route_id_val].sort_values(by='stop_order_on_route')
                if route_sequence_df.empty: continue
                stop_to_order_map = pd.Series(route_sequence_df['stop_order_on_route'].values, index=route_sequence_df['stop_id_mapped']).to_dict()
                if r_stop_id not in stop_to_order_map: continue
                r_stop_order = stop_to_order_map[r_stop_id]
                found_connection_on_this_route_for_r_stop = False

                for p_stop_id in destination_target_stop_ids:
                    if p_stop_id in stop_to_order_map:
                        p_stop_order = stop_to_order_map[p_stop_id]
                        if r_stop_order < p_stop_order:
                            found_connection_on_this_route_for_r_stop = True; hops = p_stop_order - r_stop_order
                            current_distance_m = 0.0; path_found_in_graph = True
                            path_segment_df = route_sequence_df[(route_sequence_df['stop_order_on_route'] >= r_stop_order) & (route_sequence_df['stop_order_on_route'] <= p_stop_order)]
                            actual_stops_in_path_sequence = path_segment_df['stop_id_mapped'].tolist()
                            
                            if len(actual_stops_in_path_sequence) < 2:
                                if r_stop_id == p_stop_id: current_distance_m = 0.0
                                else: path_found_in_graph = False
                            else:
                                for i in range(len(actual_stops_in_path_sequence) - 1):
                                    from_s, to_s = actual_stops_in_path_sequence[i], actual_stops_in_path_sequence[i+1]
                                    if G_inner.has_edge(from_s, to_s):
                                        edge_data = G_inner.get_edge_data(from_s, to_s)
                                        if edge_data.get('type') == 'transit' and edge_data.get('route_id') == route_id_val:
                                            current_distance_m += edge_data.get('distance_m', 0.0)
                                        else: path_found_in_graph = False; break
                                    else: path_found_in_graph = False; break
                            if not path_found_in_graph: current_distance_m = None 
                            direct_transit_connections.append({
                                'origin_poi': origin_poi_name_inner, 'destination_poi': destination_poi_name_inner,
                                'route_id': route_id_val, 'origin_stop_id': r_stop_id, 'destination_stop_id': p_stop_id,
                                'origin_stop_order': r_stop_order, 'destination_stop_order': p_stop_order,
                                'hops': hops, 'transit_distance_m': current_distance_m })
                if not found_connection_on_this_route_for_r_stop and r_stop_id in stop_to_order_map :
                     origin_routes_no_destination_connection.append({
                         'origin_stop_id': r_stop_id, 'route_id': route_id_val,
                         'message': f"Route {route_id_val} serves {origin_poi_name_inner} area stop {r_stop_id} but does not connect to any target {destination_poi_name_inner} area stops in sequence."})
        connections_df_result = pd.DataFrame()
        if direct_transit_connections:
            connections_df_result = pd.DataFrame(direct_transit_connections)
            connections_df_result = connections_df_result.sort_values(by=['route_id', 'origin_stop_order', 'hops'])
        no_connection_df_result = pd.DataFrame()
        if origin_routes_no_destination_connection:
            no_connection_df_result = pd.DataFrame(origin_routes_no_destination_connection).drop_duplicates()
        return connections_df_result, no_connection_df_result
 

    # --- Call the inner function ---
    place_to_building_connections_df, place_to_building_no_connection_df = direct_transit_conn_between_places(
        G, bus_timetables, origin_poi_name, origin_nearby_stops_info, 
        destination_poi_name, destination_nearby_stops_info
    )

    
    if place_to_building_connections_df is not None and not place_to_building_connections_df.empty:
        print("\nEnrich connections with walking distances...")
        # --- Calculate Walking Distance from Origin POI to its nearby bus stops ---
        unique_origin_stops = place_to_building_connections_df['origin_stop_id'].unique()
        origin_walking_distances_map = {}
        for stop_id in unique_origin_stops:
            walking_distance = np.nan # Default
            if G.has_node(stop_id):
                # Use origin_poi_name 
                if G.has_edge(origin_poi_name, stop_id): 
                    edge_data = G.get_edge_data(origin_poi_name, stop_id)
                    if edge_data.get('type') == 'access_egress':
                        walking_distance = edge_data.get('distance_m', np.nan)
            else:
                print(f"Warning: Origin stop ID '{stop_id}' from connections_df not found in graph G.")
            origin_walking_distances_map[stop_id] = walking_distance
        place_to_building_connections_df['walking_distance_from_origin_poi_m'] = place_to_building_connections_df['origin_stop_id'].map(origin_walking_distances_map)

        # --- Calculate Walking Distance from Destination bus stops to Destination POI ---
        unique_dest_stops = place_to_building_connections_df['destination_stop_id'].unique()
        dest_walking_distances_map = {}
        for stop_id in unique_dest_stops:
            walking_distance = np.nan # Default
            if G.has_node(stop_id):
                # Use destination_poi_name 
                if G.has_edge(stop_id, destination_poi_name): 
                    edge_data = G.get_edge_data(stop_id, destination_poi_name)
                    if edge_data.get('type') == 'access_egress': 
                        walking_distance = edge_data.get('distance_m', np.nan)
            else:
                print(f"Warning: Destination stop ID '{stop_id}' from connections_df not found in graph G.")
            dest_walking_distances_map[stop_id] = walking_distance
        place_to_building_connections_df['walking_distance_to_dest_poi_m'] = place_to_building_connections_df['destination_stop_id'].map(dest_walking_distances_map)

        # --- Create total travel cost ---
        if 'transit_distance_m' in place_to_building_connections_df.columns:
            place_to_building_connections_df['numeric_origin_walk_dist'] = pd.to_numeric(place_to_building_connections_df['walking_distance_from_origin_poi_m'], errors='coerce')
            place_to_building_connections_df['numeric_transit_dist'] = pd.to_numeric(place_to_building_connections_df['transit_distance_m'], errors='coerce')
            place_to_building_connections_df['numeric_dest_walk_dist'] = pd.to_numeric(place_to_building_connections_df['walking_distance_to_dest_poi_m'], errors='coerce')
            
            place_to_building_connections_df['total_journey_distance_m'] = place_to_building_connections_df[
                ['numeric_origin_walk_dist', 'numeric_transit_dist', 'numeric_dest_walk_dist']
            ].sum(axis=1, min_count=3) 
                
            # Drop temporary numeric columns
            cols_to_drop_temp = ['numeric_origin_walk_dist', 'numeric_transit_dist', 'numeric_dest_walk_dist']
            place_to_building_connections_df = place_to_building_connections_df.drop(columns=[col for col in cols_to_drop_temp if col in place_to_building_connections_df.columns])
            
        else:
            print("Warning: 'transit_distance_m' not found. Cannot calculate total_journey_distance_m.")
    else:
        print("No connections found (place_to_building_connections_df is None or empty). Skipping enrichment.")

    print("--- Enriched Transit Connection Analysis Finished ---")
    return place_to_building_connections_df, place_to_building_no_connection_df




if __name__ == '__main__':

    place_to_building_connections_df, no_conn_df = get_enriched_transit_connections(
        G_input=G,
        bus_timetables_input=bus_timetables,
        origin_poi_name_param=place_node_id,
        origin_nearby_stops_info_param=place_nearby_stops,
        destination_poi_name_param=building_node_id,
        destination_nearby_stops_info_param=building_nearby_stops
    )

    print("\n--- Results from get_enriched_transit_connections (Example) ---")
    if place_to_building_connections_df is not None and not place_to_building_connections_df.empty:
        print("Enriched Connections DataFrame:")
        try: display(place_to_building_connections_df.head())
        except NameError: print(place_to_building_connections_df.head())
        if 'total_journey_distance_m' in place_to_building_connections_df.columns:
            print("\nSorted by total journey distance:")
            try: display(place_to_building_connections_df.sort_values(by='total_journey_distance_m').head())
            except NameError: print(place_to_building_connections_df.sort_values(by='total_journey_distance_m').head())
    else:
        print("No enriched connections found or DataFrame is None.")

    if no_conn_df is not None and not no_conn_df.empty:
        print("\nNo Connection DataFrame:")
        try: display(no_conn_df.head())
        except NameError: print(no_conn_df.head())
    else:
        print("No 'no connection' routes identified or DataFrame is None.")


# ### 11. Accessibility Score

# In[102]:


# import for display
try:
    from IPython.display import display
except ImportError:
    display = print

def calculate_and_display_accessibility_score(
    place_to_building_connections_df_input: pd.DataFrame
):
    """
    Calculates an accessibility score based on journey distances in the provided
    DataFrame and displays it with a color-coded message.

    Args:
        place_to_building_connections_df_input (pd.DataFrame): DataFrame containing journey data,
            expected to have 'origin_poi', 'destination_poi', and a distance column
            (default 'total_journey_distance_m').
    """

    if place_to_building_connections_df_input is None or place_to_building_connections_df_input.empty:
        print("Error: Input DataFrame 'place_to_building_connections_df_input' is None or empty.")
        print("\nAccessibility Score cannot be calculated.")
        return

    # Use a copy to avoid modifying the original DataFrame 
    place_to_building_connections_df = place_to_building_connections_df_input.copy()

    # --- Inner Helper Function: calculate_exp_decay_accessibility_score ---
    def calculate_exp_decay_accessibility_score(connections_df_internal,
                                                distance_column='total_journey_distance_m',
                                                beta=0.0001, 
                                                weights={'d1': 0.50, 'd2': 0.30, 'd3': 0.20}):
        """
        Calculates an accessibility score (0-100, higher is better) based on exponential
        decay of the top three unique shortest journey distances, with given weights.
        (This is an inner function)
        """
        if distance_column not in connections_df_internal.columns:
            print(f"Error: Distance column '{distance_column}' not found in DataFrame for score calculation.")
            return 0.0  

        valid_distances_df = connections_df_internal.dropna(subset=[distance_column]).copy()
        if valid_distances_df.empty:
            print("No valid journey distances available to calculate score.")
            return 0.0

        valid_distances_df = valid_distances_df.sort_values(by=distance_column)
        unique_shortest_distances = valid_distances_df[distance_column].unique()
        
        if len(unique_shortest_distances) == 0:
            print("No unique shortest distances found after filtering.")
            return 0.0

        distances_to_score = {'d1': np.nan, 'd2': np.nan, 'd3': np.nan}
        if len(unique_shortest_distances) >= 1: distances_to_score['d1'] = unique_shortest_distances[0]
        if len(unique_shortest_distances) >= 2: distances_to_score['d2'] = unique_shortest_distances[1]
        if len(unique_shortest_distances) >= 3: distances_to_score['d3'] = unique_shortest_distances[2]

        print(f"Calculating score using beta (decay parameter): {beta}")
        print(f"Distances used for scoring (d1, d2, d3): {distances_to_score}")
       
        score_components = {}
        for key, d_val in distances_to_score.items():
            if not np.isnan(d_val) and d_val >= 0: 
                component_score = 100 * np.exp(-beta * d_val)
                score_components[key] = component_score
            else:
                score_components[key] = 0.0 

        final_accessibility_score_calc = (weights.get('d1', 0) * score_components.get('d1', 0.0)) + \
                                     (weights.get('d2', 0) * score_components.get('d2', 0.0)) + \
                                     (weights.get('d3', 0) * score_components.get('d3', 0.0))
        
        return final_accessibility_score_calc
    # --- End of inner helper function ---

    print("\n--- Calculating and Displaying Accessibility Score ---")
    
    # Parameters for the accessibility score calculation
    distance_col_name = 'total_journey_distance_m'
    beta_param = 0.0001
    score_weights = {'d1': 0.50, 'd2': 0.30, 'd3': 0.20}

    # Calculate accessibility score using the inner function
    accessibility_score_raw = calculate_exp_decay_accessibility_score(
        connections_df_internal=place_to_building_connections_df,
        distance_column=distance_col_name,
        beta=beta_param, 
        weights=score_weights
    )
    accessibility_score = round(accessibility_score_raw, 2)


    # Get unique origin and destination from the connections dataframe
    origin = "Unknown Origin"
    destination = "Unknown Destination"
    
    required_poi_cols = ['origin_poi', 'destination_poi']
    if all(col in place_to_building_connections_df.columns for col in required_poi_cols) and \
       not place_to_building_connections_df.empty:
        try:
            origin = place_to_building_connections_df['origin_poi'].iloc[0]
            destination = place_to_building_connections_df['destination_poi'].iloc[0]
        except IndexError:
            print("Warning: Could not extract origin/destination POI names from DataFrame")
    else:
        print(f"Warning: Missing '{required_poi_cols[0]}' or '{required_poi_cols[1]}' columns, or DataFrame is empty.")


    # Determine color based on score 
    if accessibility_score >= 80:
        color = '\033[38;5;22m'   # Dark green
    elif accessibility_score >= 60:
        color = '\033[38;5;27m'   # Blue
    elif accessibility_score >= 30:
        color = '\033[38;5;214m'  # Orange
    else:
        color = '\033[38;5;196m'  # Red

    # Print the final score 
    print(f"{color}\033[1m\nAccessibility Score between {origin} and {destination} is: {accessibility_score}\033[0m")
    print("--- Accessibility Score Calculation and Display Finished ---")

    return accessibility_score


if __name__ == '__main__':


    print("--- Calling calculate_and_display_accessibility_score ---")
    if 'place_to_building_connections_df' in locals() and isinstance(place_to_building_connections_df, pd.DataFrame):
        calculate_and_display_accessibility_score(place_to_building_connections_df)
 
    else:
        print("Dummy 'place_to_building_connections_df' not defined for example usage.")


# # LangChain

# ### 12.1. LangChain Configuration 

# In[103]:


## Model configuration 1
llm = ChatOllama(model="llama3") 
# print("ChatOllama initialized successfully with the new package.")
from langchain.tools import tool #to define the tool for an agent
from langchain import hub #to pull the PromptTemplate from the hub
from langchain.agents import AgentExecutor, create_react_agent #to create the agent
from pydantic import BaseModel, Field #to define the schema of the agent
from typing import List, Type, Dict, Any #to define the type of the agent
# !pip install -U langchain-openai


# ### 12.2. LangChain tools 

# ### 12.2.1.  LangChain tool 1: get accessiblity score

# In[104]:


# this tool responds based on 'place_to_building_connections_df'

def get_journey_accessibility_info_v2(
    origin_poi_name: str,       # Changed from origin_stop_id
    destination_poi_name: str,  # Changed from destination_stop_id
    connections_df: pd.DataFrame 
) -> str:
    """
    Provides accessibility information for a journey between a specified origin POI 
    and destination POI using the provided connections_df. 
    It dynamically calculates the accessibility score and details the top journey option(s).
    """
    if connections_df is None or connections_df.empty:
        return "Error: Input connections_df is not provided or is empty."

    # Clean the input POI names for robust comparison
    clean_origin_input = origin_poi_name.strip().lower()
    clean_destination_input = destination_poi_name.strip().lower()

    print(f"[get_journey_accessibility_info_v2 DEBUG] Filtering DataFrame for origin_poi='{clean_origin_input}', destination_poi='{clean_destination_input}'")
    
    # --- KEY CHANGE: Filter by POI names using the function's input parameters ---
    journey_df_filtered = connections_df[
        (connections_df['origin_poi'].astype(str).str.strip().str.lower() == clean_origin_input) &
        (connections_df['destination_poi'].astype(str).str.strip().str.lower() == clean_destination_input)
    ].copy() 

    print(f"[get_journey_accessibility_info_v2 DEBUG] Rows found after filtering by POI name: {len(journey_df_filtered)}")

    if journey_df_filtered.empty:
        # Provide more context in the error message
        sample_origins = connections_df['origin_poi'].astype(str).str.strip().str.lower().unique()[:5]
        sample_destinations = connections_df['destination_poi'].astype(str).str.strip().str.lower().unique()[:5]
        return (f"No journey data found in the provided connections_df for "
                f"origin POI '{origin_poi_name}' (searched as '{clean_origin_input}') to "
                f"destination POI '{destination_poi_name}' (searched as '{clean_destination_input}').\n"
                f"Check names against DataFrame content. For reference, some unique lowercased/stripped origin POIs in data: {list(sample_origins)}\n"
                f"Some unique lowercased/stripped destination POIs in data: {list(sample_destinations)}")

    try:
        actual_origin_poi_name = journey_df_filtered.iloc[0]['origin_poi']
        actual_destination_poi_name = journey_df_filtered.iloc[0]['destination_poi']
        print('chck2:', actual_origin_poi_name, actual_destination_poi_name) 
    except (IndexError, KeyError) as e:
        print(f"[get_journey_accessibility_info_v2 DEBUG] Could not retrieve POI names from filtered data: {e}")
        # Fallback to cleaned input names if retrieval fails (shouldn't if journey_df_filtered is not empty)
        actual_origin_poi_name = origin_poi_name 
        actual_destination_poi_name = destination_poi_name


    try:
        accessibility_score = calculate_and_display_accessibility_score(journey_df_filtered) 
    except Exception as e:
        return f"Error calculating accessibility score: {e}. 'calculate_and_display_accessibility_score' is correctly defined. Data for score calc had {len(journey_df_filtered)} rows."

    top_options_df = journey_df_filtered.sort_values(by='total_journey_distance_m').reset_index(drop=True)

    if top_options_df.empty: 
        # This should ideally not be reached if journey_df_filtered was not empty initially
        return (f"Could not determine top options for {actual_origin_poi_name} to {actual_destination_poi_name} after sorting.")

    shortest_journey = top_options_df.iloc[0]

    output_str = f"Accessibility Information for journey from '{actual_origin_poi_name}' to '{actual_destination_poi_name}':\n" # Simpler title
    output_str += f"- Calculated Accessibility Score: {accessibility_score:.2f}\n"
    output_str += f"- This score is based on an exponential decay of the shortest unique journey distances.\n"
    
    output_str += f"\nDetails of the overall shortest journey option found:\n"
    output_str += f"  - Route ID(s) involved: {shortest_journey.get('route_id', 'N/A')}\n"
    output_str += f"  - Origin Bus Stop ID: {shortest_journey.get('origin_stop_id', 'N/A')}\n" # Stop IDs are still useful info
    output_str += f"  - Destination Bus Stop ID: {shortest_journey.get('destination_stop_id', 'N/A')}\n"
    
    walk_origin_dist = shortest_journey.get('walking_distance_from_origin_poi_m', 'N/A')
    transit_dist = shortest_journey.get('transit_distance_m', 'N/A')
    walk_dest_dist = shortest_journey.get('walking_distance_to_dest_poi_m', 'N/A')
    total_dist = shortest_journey.get('total_journey_distance_m', 'N/A')

    def format_dist(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            return f"{val:.2f}"
        return 'N/A'

    output_str += f"  - Walking distance from {actual_origin_poi_name}: {format_dist(walk_origin_dist)} meters\n"
    output_str += f"  - Transit distance: {format_dist(transit_dist)} meters\n"
    output_str += f"  - Walking distance to {actual_destination_poi_name}: {format_dist(walk_dest_dist)} meters\n"
    output_str += f"  - Total Journey Distance: {format_dist(total_dist)} meters\n"
    
    unique_shortest_distances = top_options_df['total_journey_distance_m'].dropna().unique()
    if len(unique_shortest_distances) > 0:
        output_str += f"\nThe accessibility score considers the following shortest unique total journey distances (meters) from the data for this O-D pair:\n"
        for i, dist_val in enumerate(sorted(unique_shortest_distances)[:3]): 
             output_str += f"  - d{i+1}: {dist_val:.2f}\n"

    return output_str


#decorator function for get_accessibility_and_journey_details
@tool
def get_accessibility_and_journey_details(origin_and_destination_poi_query: str) -> str:
    """
    Provides detailed accessibility information for a public transport journey 
    between an origin POI (Place of Interest) and a destination POI.
    The input 'origin_and_destination_poi_query' should be a string clearly stating 
    the origin and destination POI names, for example: 'journey from Rahoon to Portershed' 
    or simply 'Rahoon to Portershed'. 
    The tool will attempt to parse out the origin and destination POI names.
    """
    print(f"\n[TOOL DEBUG] get_accessibility_and_journey_details received raw POI query: '{origin_and_destination_poi_query}'")
    cleaned_query = origin_and_destination_poi_query.strip()
    print(f"[TOOL DEBUG] Cleaned POI query: '{cleaned_query}'")
    
    origin_poi_name = None
    destination_poi_name = None

    match_from_to = re.search(r"(?:from\s+)?(.*?)\s+to\s+(.*)", cleaned_query, re.IGNORECASE)
    
    
    # Pattern 1: Explicitly match "from ORIGIN to DESTINATION"
    match_pattern1 = re.search(r"from\s+(.+?)\s+to\s+(.+)", cleaned_query, re.IGNORECASE)
    if match_pattern1:
        origin_poi_name = match_pattern1.group(1).strip()
        destination_poi_name = match_pattern1.group(2).strip()
    else:
        # Pattern 2: Match "ORIGIN to DESTINATION" (if "from" is not present)
        match_pattern2 = re.search(r"(.+?)\s+to\s+(.+)", cleaned_query, re.IGNORECASE)
        if match_pattern2:
            origin_poi_name = match_pattern2.group(1).strip()
            destination_poi_name = match_pattern2.group(2).strip()

    print(f"[TOOL DEBUG] Parsed POI names by regex: origin='{origin_poi_name}', destination='{destination_poi_name}'")
    
    # Fallback to comma split if regex parsing failed or didn't yield both names
    if not origin_poi_name or not destination_poi_name:
        print(f"[TOOL DEBUG] Regex parsing failed to get both names. Trying comma split for: '{cleaned_query}'")
        parts = cleaned_query.split(',')
        if len(parts) == 2:
            origin_poi_name = parts[0].strip()
            # If the first part still contains "from", remove it
            if origin_poi_name.lower().startswith("from "):
                origin_poi_name = origin_poi_name[5:].strip() # len("from ") is 5
            destination_poi_name = parts[1].strip()
            print(f"[TOOL DEBUG] Parsed POI names by comma split: origin='{origin_poi_name}', destination='{destination_poi_name}'")
        else:
            print(f"[TOOL DEBUG] Parsing POI names failed for: '{cleaned_query}' using all methods.")
            return (f"Error: Could not reliably parse origin and destination POI names from the input: '{cleaned_query}'. "
                    f"Please ensure the input clearly separates the origin and destination.")

    if not origin_poi_name or not destination_poi_name:
        return (f"Error: Failed to extract both valid origin POI ('{origin_poi_name}') and destination POI ('{destination_poi_name}') "
                f"from input: '{cleaned_query}'.")

    if 'place_to_building_connections_df' not in globals() or globals()['place_to_building_connections_df'].empty:
        print("[TOOL DEBUG] Error: place_to_building_connections_df not found or empty in global scope.")
        return "Error: Main journey data (place_to_building_connections_df) is not loaded or is empty."
    
    current_connections_df = globals()['place_to_building_connections_df']
  
    
    print(f"[TOOL DEBUG] Calling get_journey_accessibility_info_v2 with: origin_poi_name='{origin_poi_name}', destination_poi_name='{destination_poi_name}'")
    result = get_journey_accessibility_info_v2(
        origin_poi_name,
        destination_poi_name,
        current_connections_df
    )
    print(f"[TOOL DEBUG] Result from get_journey_accessibility_info_v2 (first 100 chars): {result[:100]}...")
    return result

#test case for get_accessibility_and_journey_details
test_query_1 = f"journey from {place_node_id} to {building_node_id}"
result_1 = get_accessibility_and_journey_details(test_query_1)
print(f"Query: '{test_query_1}'\nResult:\n{result_1}\n")


# ### 12.2.2.  LangChain tool 2: get nearby bus stops

# In[105]:


#This tool responds based on 'G' graph . Hence tool 1 and tool 2 are independent of each other in terms of data.
#we can pass any node id (places or buildings) to this tool independent of our previously selected place_node_id and building_node_id

@tool
def get_nearby_bus_stops(poi_name_query: str) -> str:
    """
    Finds bus stops directly connected to a given Place of Interest (POI) name 
    in the graph via 'access_egress' (walking/cycling) edges.
    The POI name must exist as a node in the graph.
    Example input: "Portershed"
    """
    print(f"\n--- Executing get_nearby_bus_stops_simple ---")
    print(f"Received POI Name Query: '{poi_name_query}'")
    
    cleaned_poi_name = poi_name_query.strip() 

    if not cleaned_poi_name:
        return "Error: POI name query provided was empty."

    if 'G' not in globals() or not isinstance(G, (nx.Graph, nx.DiGraph)):
        print("[TOOL DEBUG] Error: Graph G is not loaded or is not a valid NetworkX graph.")
        return "Error: Main graph data (G) is not loaded or is not a valid graph."

    current_graph = globals()['G']
    
    node_id_to_use_in_graph = None
    # Try direct match first (respects case if graph has it)
    if current_graph.has_node(cleaned_poi_name):
        node_id_to_use_in_graph = cleaned_poi_name
        print(f"[TOOL DEBUG] Found POI directly in graph: '{node_id_to_use_in_graph}'")
    else:
        # Fallback to case-insensitive search
        print(f"[TOOL DEBUG] POI '{cleaned_poi_name}' not found directly. Attempting case-insensitive search...")
        for node_in_graph in current_graph.nodes():
            if str(node_in_graph).lower() == cleaned_poi_name.lower():
                node_id_to_use_in_graph = node_in_graph
                print(f"[TOOL DEBUG] Found case-insensitive match: Graph node='{node_id_to_use_in_graph}' for query='{cleaned_poi_name}'")
                break
    
    if not node_id_to_use_in_graph:
        print(f"[TOOL DEBUG] POI '{cleaned_poi_name}' (from query '{poi_name_query}') not found in graph G after all checks.")
        return f"Error: POI name '{poi_name_query}' not found in the graph G."
    
    nearby_stops_info = []
    edges_to_check = []
    if isinstance(current_graph, nx.DiGraph):
        edges_to_check.extend(current_graph.out_edges(node_id_to_use_in_graph, data=True))
        edges_to_check.extend(current_graph.in_edges(node_id_to_use_in_graph, data=True)) # For stops connecting TO POI
    else: # Undirected Graph
        edges_to_check.extend(current_graph.edges(node_id_to_use_in_graph, data=True))

    unique_stops_found = {} # To handle duplicates 

    for u, v, data in edges_to_check:
        edge_type = data.get('type')
        # We are REMOVING the explicit distance_m check against max_distance_meters here for simplicity
        # We only care that it's an access_egress edge.
        if edge_type == 'access_egress':
            potential_stop_node_id = None
            node_u_type = current_graph.nodes[u].get('type')
            node_v_type = current_graph.nodes[v].get('type')

            if u == node_id_to_use_in_graph and node_v_type == 'bus_stop':
                potential_stop_node_id = v
            elif v == node_id_to_use_in_graph and node_u_type == 'bus_stop':
                potential_stop_node_id = u
            
            if potential_stop_node_id:
                if potential_stop_node_id not in unique_stops_found: # Add if new
                    stop_name = current_graph.nodes[potential_stop_node_id].get('name', 'N/A')
                    edge_distance = data.get('distance_m') # Still useful to get distance if available
                    unique_stops_found[potential_stop_node_id] = {
                        'stop_id': potential_stop_node_id,
                        'stop_name': stop_name,
                        'distance_m': edge_distance # Store it, even if not filtering by it
                    }
    
    nearby_stops_info = list(unique_stops_found.values())

    if not nearby_stops_info:
        return f"No bus stops found directly linked by an 'access_egress' edge to '{node_id_to_use_in_graph}' (queried as '{poi_name_query}') in the graph."

    output_str = f"Nearby bus stops for '{node_id_to_use_in_graph}' (queried as '{poi_name_query}'):\n"
    # Sort by distance if distance_m is available and numeric, otherwise just list them
    try:
        # sort by distance if 'distance_m' is present and numeric
        # Filter out entries where distance_m might be None before sorting
        sortable_stops = [s for s in nearby_stops_info if isinstance(s.get('distance_m'), (int, float))]
        non_sortable_stops = [s for s in nearby_stops_info if not isinstance(s.get('distance_m'), (int, float))]
        
        for stop_info in sorted(sortable_stops, key=lambda x: x['distance_m']):
            dist_str = f", Distance: {stop_info['distance_m']:.2f} meters" if stop_info['distance_m'] is not None else ""
            output_str += f"  - Stop ID: {stop_info['stop_id']}, Name: {stop_info['stop_name']}{dist_str}\n"
        for stop_info in non_sortable_stops: # Add stops without valid distance at the end
            output_str += f"  - Stop ID: {stop_info['stop_id']}, Name: {stop_info['stop_name']}\n"

    except TypeError: # Fallback if sorting fails (e.g. distance_m is not always present/numeric)
        for stop_info in nearby_stops_info:
             output_str += f"  - Stop ID: {stop_info['stop_id']}, Name: {stop_info['stop_name']}\n"

    # print(f"[TOOL DEBUG] Returning: {output_str[:200]}...")
    return output_str



# we can select any place or building id , independent of our previously selected place_node_id and building_node_id
display(get_nearby_bus_stops(building_node_id))
display(get_nearby_bus_stops(place_node_id))


# ### 12.3. Initialize LangChain Agent for User Queries

# In[106]:


# create a list of these tools for the agent
tools = [get_accessibility_and_journey_details, get_nearby_bus_stops]
print(f"Defined {len(tools)} tools for the LangChain agent.")


# This prompt provides the LLM with instructions on how to reason and use tools.
prompt_template = hub.pull("hwchase17/react") # Harrison Chase Reasoning and Acting agent framework(ReAct)
prompt_template
agent = create_react_agent(llm, tools, prompt_template)

#LangChain Agent
# The AgentExecutor runs the agent, calls tools, and gets responses
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # Set to True to see the agent's thought process and actions
    handle_parsing_errors=True, # Helps with robustness if LLM output is not perfectly formatted
    max_iterations=10 # Prevents runaway agents if it gets stuck in a loop, adjust as needed
)
print("LangChain ReAct Agent Executor created successfully and ready to use!")


# ### 12.4. User Queries
# 
# This step executes a query using the LangChain agent (agent_executor). The agent will process the natural language input for example-("Give details about nearby bus stops of Galway Cathedral. Recommend me what it is famous for?") to find relevant information, checks which tool would provide best information and uses the tool (for example get_nearby_bus_stops tool) and use own general knowledge, and then prints the agent's response.

# In[107]:


# --Query 1 --
response = agent_executor.invoke({
    "input":"Give details about nearby bus stops of Galway Cathedral. It could be  galway cathedral building. It is a church. Recommend me what it is famous for?"
})
print(response["output"])


# In[108]:


# --Query 2 --

response = agent_executor.invoke({
    "input": "What is the accessibility score and journey details for rahoon to portershed. Would you like to give any recommendations based on your knowledge?"
})
print(response["output"])


# In[109]:


# --Query 3 --

response = agent_executor.invoke({
    "input": "What is the accessibility score and journey details for rahoon to portershed. Would you like to give any recommendations based on your knowledge?"
})
print(response["output"])


# In[110]:


# --Query 4 --

response = agent_executor.invoke({
    "input": "Is Ballybrit place much far from Galway city center or Eyre Square. Any recommendations about Eyre Square and Ballybrit?"
})
print(response["output"])


# In[111]:


# --Query 5 --

response = agent_executor.invoke({
    "input":"Find bus stops within 500 meters of Galway Cathedral. Give recommendations based on your knowledge"
})
print(response["output"])


# In[112]:


# --Query 6 --

response = agent_executor.invoke({
    "input":"Find bus stops within 500 meters of Galway Cathedral. Give recommendations based on your knowledge"
})
print(response["output"])


# In[113]:


# --Query 7 --

response = agent_executor.invoke({
    "input":"what are the nearby bus stops within 600 meters of Portershed?. Any suggestions about portershed?"
})
print(response["output"])


# In[114]:


# --Query 8 --

response = agent_executor.invoke({
    "input": "Is Ballybrit place much far from Galway city center or Eyre Square. Any recommendations about Eyre Square and Ballybrit?"
})
print(response["output"])


# In[115]:


# --Query 9 --

response = agent_executor.invoke({
    "input": "Is 'Renmore Community Centre' far from Galway city center or Eyre Square. Any suggestions about Renmore Community Centre?"
})
print(response["output"])


# In[116]:


# --Query 10 --

response = agent_executor.invoke({
    "input": " what is the accessibility score between Knocknacarragh and Portershed.Suggest communities around Knocknacarragh?"
})
print(response["output"])


# In[117]:


# --Query 11 --

response = agent_executor.invoke({
    "input": " Near by bus stops of 'Ballybrit'. Is it easyily reachable from Galway city center?"
})
print(response["output"]) 


# ### 13. Accessibility Map 

# In[120]:


# --- Find the Shortest Journey ---
shortest_journey_path_details = None # To store details of the shortest path

if 'place_to_building_connections_df' in locals() and \
   not place_to_building_connections_df.empty and \
   'total_journey_distance_m' in place_to_building_connections_df.columns:

    # Drop rows where total journey distance is NaN
    valid_journeys_df = place_to_building_connections_df.dropna(subset=['total_journey_distance_m']).copy()
    
    if not valid_journeys_df.empty:
        valid_journeys_df = valid_journeys_df.sort_values(by='total_journey_distance_m')
        shortest_journey_row = valid_journeys_df.iloc[0].copy() # Get the top row (shortest)

        # --- Store the necessary details for plotting ---
        shortest_journey_path_details = {
            'origin_poi_node_id': place_node_id, # variable for Rahoon POI node ID
            'origin_bus_stop_id': shortest_journey_row['origin_stop_id'],
            'route_id': shortest_journey_row['route_id'],
            'origin_bus_stop_order': shortest_journey_row['origin_stop_order'],
            'destination_bus_stop_order': shortest_journey_row['destination_stop_order'],
            'destination_bus_stop_id': shortest_journey_row['destination_stop_id'],
            'destination_poi_node_id': building_node_id, # var for Portershed POI
            'total_distance': shortest_journey_row['total_journey_distance_m']
        }
        print("\n--- Shortest Journey Details for Plotting ---")
        print(f"Origin POI: {shortest_journey_path_details['origin_poi_node_id']}")
        print(f"Origin Bus Stop: {shortest_journey_path_details['origin_bus_stop_id']}")
        print(f"Route ID: {shortest_journey_path_details['route_id']}")
        print(f"Destination Bus Stop: {shortest_journey_path_details['destination_bus_stop_id']}")
        print(f"Destination POI: {shortest_journey_path_details['destination_poi_node_id']}")
        print(f"Total Distance: {shortest_journey_path_details['total_distance']:.2f}m")

        # Get the sequence of transit stops for this shortest path
        route_seq_df_shortest = bus_timetables[
            bus_timetables['route_id'] == shortest_journey_path_details['route_id']
        ].sort_values(by='stop_order_on_route')
        
        path_segment_df_shortest = route_seq_df_shortest[
            (route_seq_df_shortest['stop_order_on_route'] >= shortest_journey_path_details['origin_bus_stop_order']) &
            (route_seq_df_shortest['stop_order_on_route'] <= shortest_journey_path_details['destination_bus_stop_order'])
        ]
        shortest_journey_path_details['transit_stop_sequence_ids'] = path_segment_df_shortest['stop_id_mapped'].tolist()
        print(f"Transit Stop Sequence: {shortest_journey_path_details['transit_stop_sequence_ids']}")
    else:
        print("No valid journeys with calculated total distances found to select the shortest.")
else:
    print("place_to_building_connections_df not found, empty, or 'total_journey_distance_m' column missing.")



###################################################################################################

# Configure osmnx settings and logging
# ox.config(log_console=True, use_cache=False)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# --- Configuration ---
place_name = "Galway, Ireland"
shapefile_base_dir = '/Users/conall/Projects/atlantech-ai-challenge-2025/data/ireland-and-northern-ireland-latest-free.shp' 

shapefile_layers = {
    'roads': 'gis_osm_roads_free_1.shp',
    'water_poly': 'gis_osm_water_a_free_1.shp',
    'railways': 'gis_osm_railways_free_1.shp',
    'waterways': 'gis_osm_waterways_free_1.shp',
    'landuse': 'gis_osm_landuse_a_free_1.shp',
    'buildings': 'gis_osm_buildings_a_free_1.shp',
    'places_poly': 'gis_osm_places_a_free_1.shp'
}

print(f"\n--- Processing Data for: {place_name} ---")
print(f"Using Shapefile directory: {shapefile_base_dir}")

try:
    # --- *** GET GALWAY BOUNDARY *** ---
    print("\nFetching boundary for Galway...")
    boundary_gdf = ox.geocode_to_gdf(place_name).to_crs("EPSG:4326")
    if boundary_gdf.empty:
        raise ValueError(f"Could not geocode '{place_name}'.")
    print(f"Boundary fetched. CRS set to: {boundary_gdf.crs}")



    # --- *** LOAD IRELAND SHAPEFILES & CLIP TO GALWAY BOUNDARY *** ---
    print("\nLoading and clipping Ireland-wide layers to Galway boundary...")
    galway_gdfs = {}
    for layer_name, shp_filename in shapefile_layers.items():
        shp_path = os.path.join(shapefile_base_dir, shp_filename)
        print(f"--- Processing layer: {layer_name} ---")
        if not os.path.exists(shp_path):
            print(f"*** WARNING: Shapefile not found: {shp_path} - Skipping layer '{layer_name}' ***")
            continue
        try:
            ireland_layer_gdf = gpd.read_file(shp_path)
            if ireland_layer_gdf.crs != boundary_gdf.crs:
                ireland_layer_gdf = ireland_layer_gdf.to_crs(boundary_gdf.crs)
            clipped_gdf = gpd.clip(ireland_layer_gdf, boundary_gdf, keep_geom_type=True)
            if not clipped_gdf.empty:
                galway_gdfs[layer_name] = clipped_gdf
            else:
                print(f"Note: No features found for layer '{layer_name}'.")
        except Exception as e:
            print(f"*** ERROR processing layer '{layer_name}': {e} ***")



    # --- *** PREPARE BUS STOP GEODATAFRAME FROM GSTOPS_DF_V1 *** --- 
    print("\nPreparing Galway Bus Stop data from gstops_df_v1...")
    bus_stops_gdf = None
    if 'gstops_df_v1' in locals() and isinstance(gstops_df_v1, pd.DataFrame) and not gstops_df_v1.empty:
        # Ensure 'stop_lat' and 'stop_lon' columns exist
        if 'stop_lat' in gstops_df_v1.columns and 'stop_lon' in gstops_df_v1.columns:
            try:
                # Drop rows with invalid (NaN) coordinates before creating GeoDataFrame
                temp_stops_df = gstops_df_v1.dropna(subset=['stop_lat', 'stop_lon']).copy()
                
                if not temp_stops_df.empty:
                    bus_stops_gdf = gpd.GeoDataFrame(
                        temp_stops_df,
                        geometry=gpd.points_from_xy(temp_stops_df['stop_lon'], temp_stops_df['stop_lat']),
                        crs="EPSG:4326"  
                    )
                    print(f"Created GeoDataFrame 'bus_stops_gdf' with {len(bus_stops_gdf)} stops from gstops_df_v1.")
                    # Reproject if CRS doesn't match the boundary CRS
                    if bus_stops_gdf.crs != boundary_gdf.crs:
                        print(f"Reprojecting bus stops GDF to {boundary_gdf.crs}...");
                        bus_stops_gdf = bus_stops_gdf.to_crs(boundary_gdf.crs)
                        print("Reprojection complete.")
                else:
                    print("Warning: No valid coordinates found in gstops_df_v1 after cleaning.")
            except Exception as e:
                print(f"*** ERROR converting gstops_df_v1 data: {e} ***")
                bus_stops_gdf = None
        else:
            print("Warning: 'stop_lat' or 'stop_lon' columns not found in gstops_df_v1.")
    else:
        print("Warning: 'gstops_df_v1' DataFrame not found or is empty. Please load it first.")


    # --- *** PREPARE BUS ROUTES GEODATAFRAME FROM gvariations_df_v1 *** --- 
    bus_routes_gdf = None # Initialize

    if 'gvariations_df_v1' in locals() and isinstance(gvariations_df_v1, pd.DataFrame) and not gvariations_df_v1.empty and \
    'bus_stops_gdf' in locals() and isinstance(bus_stops_gdf, gpd.GeoDataFrame) and not bus_stops_gdf.empty:

        print("\nEnriching gvariations_df_v1 with first/last stop Point geometries...")
        
        bus_routes_gdf = gvariations_df_v1.copy()
        
        if 'stop_id' in bus_stops_gdf.columns and 'geometry' in bus_stops_gdf.columns:
            
            # --- Handle duplicate stop_ids in bus_stops_gdf to get a unique map ---
            # A single physical stop_id has one location, regardless of how many route directions use it.
            # We keep the first occurrence of each stop_id to get its unique geometry.
            bus_stops_gdf_unique_locations = bus_stops_gdf.drop_duplicates(subset=['stop_id'], keep='first')
            
            # Create the mapping series from this de-duplicated DataFrame
            stop_id_to_point_geometry = bus_stops_gdf_unique_locations.set_index('stop_id')['geometry']
            
        else:
            print("Error: 'stop_id' or 'geometry' column not found in bus_stops_gdf. Cannot map stop Point geometries.")
            stop_id_to_point_geometry = pd.Series(dtype='object') 

        # Map first stop Point geometry
        bus_routes_gdf['first_stop_point'] = bus_routes_gdf['first_stop_id'].map(stop_id_to_point_geometry)
        
        # Map last stop Point geometry
        bus_routes_gdf['last_stop_point'] = bus_routes_gdf['last_stop_id'].map(stop_id_to_point_geometry)
        
        num_first_stops_mapped = bus_routes_gdf['first_stop_point'].notna().sum()
        num_last_stops_mapped = bus_routes_gdf['last_stop_point'].notna().sum()
        
        print(f"Successfully mapped Point geometry for {num_first_stops_mapped} first stops.")
        print(f"Successfully mapped Point geometry for {num_last_stops_mapped} last stops.")

        # Check if any mappings failed (resulting in NaNs)
        if bus_routes_gdf['first_stop_point'].isnull().any() or bus_routes_gdf['last_stop_point'].isnull().any():
            print("Warning: Some first/last stop points could not be mapped (resulting in NaNs).")
            # You could print these rows for inspection:
            # print(bus_routes_gdf[bus_routes_gdf['first_stop_point'].isnull() | bus_routes_gdf['last_stop_point'].isnull()])


        print("\n--- bus_routes_gdf (with Point geometries) ---")
        # Display relevant columns to check the mapping
        display_cols = ['first_stop_id', 'first_stop_point', 'last_stop_id', 'last_stop_point']
        # Add other columns from gvariations_df_v1 if they provide context
        if 'route_id' in bus_routes_gdf.columns: display_cols.insert(0, 'route_id')
        if 'direction_id' in bus_routes_gdf.columns: display_cols.insert(1, 'direction_id')

        print(bus_routes_gdf[display_cols].head())
        print(f"Shape of bus_routes_gdf: {bus_routes_gdf.shape}")

    else:
        print("\nPrerequisite DataFrames ('gvariations_df_v1' or 'bus_stops_gdf') not available or empty. Cannot create bus_routes_gdf.")


      # --- *** CREATE PLACE SUMMARY DATAFRAME *** ---
    print("\nCreating DataFrame for Galway Place Names and Coordinates...")
    galway_places_summary_df = None # Initialize
    if 'places_poly' in galway_gdfs and not galway_gdfs['places_poly'].empty:
        places_data = []
        # Check if the 'name' column exists
        if 'name' not in galway_gdfs['places_poly'].columns:
            print("Warning: 'name' column not found in places_poly layer. Cannot extract place names.")
        else:
            # Iterate through valid polygons with names
            for idx, row in galway_gdfs['places_poly'][galway_gdfs['places_poly']['name'].notna() & galway_gdfs['places_poly'].geometry.is_valid].iterrows():
                place_name_val = row['name']; geometry = row.geometry; rep_point = None
                # Get representative point (or centroid as fallback)
                if hasattr(geometry, 'representative_point'):
                    try: rep_point = geometry.representative_point()
                    except Exception: rep_point = geometry.centroid # Fallback if representative_point fails
                else: rep_point = geometry.centroid # Fallback if method doesn't exist
                # Append if point is valid
                if rep_point and rep_point.is_valid:
                    places_data.append({'place_name': place_name_val,'latitude': rep_point.y,'longitude': rep_point.x})
            # Create DataFrame if data was extracted
            if places_data:
                galway_places_summary_df = pd.DataFrame(places_data)
                print(f"Created DataFrame 'galway_places_summary_df' with {len(galway_places_summary_df)} places.")
                print(galway_places_summary_df.head())
            else: print("No valid places with names found to create summary DataFrame.")
    else: print("Clipped 'places_poly' GeoDataFrame not found or is empty.")

    galway_places_summary_df1 = None # Initialize

    if 'galway_places_summary_df' in locals() and isinstance(galway_places_summary_df, pd.DataFrame) and not galway_places_summary_df.empty:
        galway_places_summary_df1 = galway_places_summary_df.copy()
        if 'place_name' in galway_places_summary_df1.columns:
            galway_places_summary_df1 = galway_places_summary_df1.sort_values('place_name').reset_index(drop=True)
        else:
            print("Warning: 'place_name' column not found for sorting. Index will be based on current order.")

        # Create custom indices starting with 'P'
        place_indices = [f'P{i+1}' for i in range(len(galway_places_summary_df1))]
        galway_places_summary_df1.index = place_indices

        print("\nCreated DataFrame 'galway_places_summary_df1' with custom 'P' indices:")
        print(f"Number of places: {len(galway_places_summary_df1)}")
        print("\nFirst few rows of 'galway_places_summary_df1':")
        print(galway_places_summary_df1.head())
    else:
        print("Cannot create 'galway_places_summary_df1' as 'galway_places_summary_df' is not available or is empty.")
    # --- *** END PLACES SECTION *** ---



# --- *** CHECK RAHOON PLACE ID FOR PLOTTING *** ---
    rahoon_place_id = None # To store the 'P' index if Rahoon is found
    if 'galway_places_summary_df1' in locals() and isinstance(galway_places_summary_df1, pd.DataFrame) and not galway_places_summary_df1.empty:
        if 'place_name' in galway_places_summary_df1.columns:
            # Search for 'Rahoon' in the 'place_name' column 
            rahoon_search_results = galway_places_summary_df1[galway_places_summary_df1['place_name'].str.contains('Rahoon', case=False, na=False)]

            if not rahoon_search_results.empty:
                print(f"\n--- Found 'Rahoon' in galway_places_summary_df1 ---")
                rahoon_place_data = rahoon_search_results.iloc[0]
                rahoon_place_id = rahoon_place_data.name 
                print(f"Place Name: {rahoon_place_data['place_name']}")
                print(f"Index (ID): {rahoon_place_id}")
                print(f"Latitude: {rahoon_place_data['latitude']}")
                print(f"Longitude: {rahoon_place_data['longitude']}")
            else:
                print("\nPlace name containing 'Rahoon' not found in galway_places_summary_df1.")
        else:
            print("\n'place_name' column not found in galway_places_summary_df1.")
    else:
        print("\nDataFrame 'galway_places_summary_df1' not available for searching 'Rahoon'.")


# --- *** CREATE BUILDINGS SUMMARY DATAFRAME *** ---
    print("\nCreating DataFrame for Galway Buildings with Type and Coordinates...")
    galway_buildings_summary_df = None # Initialize
    if 'buildings' in galway_gdfs and not galway_gdfs['buildings'].empty:
        buildings_data = []

        # Check what columns are available in the buildings layer
        print(f"Available columns in buildings layer: {galway_gdfs['buildings'].columns.tolist()}")

        # Extract building info - name, osm_id, and type (typically in fclass or type column)
        for idx, row in galway_gdfs['buildings'][galway_gdfs['buildings'].geometry.is_valid].iterrows():
            osm_id = row.get('osm_id', None)
            name = row.get('name', None)
            building_type = None
            for type_col in ['fclass', 'type', 'building']:
                if type_col in row and row[type_col] is not None:
                    building_type = row[type_col]; break
            try:
                centroid = row.geometry.centroid
                if centroid and centroid.is_valid:
                    buildings_data.append({
                        'building_name': name, 'osm_id': osm_id, 'building_type': building_type,
                        'latitude': centroid.y, 'longitude': centroid.x
                    })
            except Exception as e: print(f"Error calculating centroid for building {osm_id}: {e}")

        if buildings_data:
            galway_buildings_summary_df = pd.DataFrame(buildings_data)
            print(f"Created DataFrame 'galway_buildings_summary_df' with {len(galway_buildings_summary_df)} buildings.")
            print(galway_buildings_summary_df.head())
        else: print("No valid building data found to create summary DataFrame.")
    else: print("Clipped 'buildings' GeoDataFrame not found or is empty.")

    # --- *** REFINE BUILDING SUMMARY DATAFRAME *** ---
    galway_buildings_summary_df1 = None # Initialize
    if galway_buildings_summary_df is not None:
        galway_buildings_summary_df1 = galway_buildings_summary_df[galway_buildings_summary_df['building_name'].notnull()].copy()
        galway_buildings_summary_df1 = galway_buildings_summary_df1.sort_values('building_name')
        building_indices = [f'B{i+1}' for i in range(len(galway_buildings_summary_df1))]
        galway_buildings_summary_df1.index = building_indices
        print("\nCreated filtered DataFrame 'galway_buildings_summary_df1' with named buildings:")
        print(f"Number of named buildings: {len(galway_buildings_summary_df1)}")
        print("\nFirst few rows of filtered DataFrame:")
        print(galway_buildings_summary_df1.head())
    else: print("Cannot create filtered DataFrame as galway_buildings_summary_df is None")

    # --- *** END BUILDINGS SECTION *** ---



 
    bus_stops_near_rahoon_gdf = None
    bus_stops_near_portershed_gdf = None

    if 'bus_stops_gdf' in locals() and bus_stops_gdf is not None and not bus_stops_gdf.empty:
        if place_nearby_stop_ids:
            bus_stops_near_rahoon_gdf = bus_stops_gdf[bus_stops_gdf['stop_id'].isin(place_nearby_stop_ids)]
        if building_nearby_stop_ids:
            bus_stops_near_portershed_gdf = bus_stops_gdf[bus_stops_gdf['stop_id'].isin(building_nearby_stop_ids)]
    else:
        print("Warning: bus_stops_gdf not available for filtering nearby stops.")




    # --- *** PLOTTING CLIPPED GALWAY DATA *** ---
    print("\nPlotting clipped Galway map layers...")
    fig, ax = plt.subplots(figsize=(18, 18), facecolor='white', dpi=250)

    # Define base colors
    color_water = '#a8dff5'; color_land = '#f2f4f6'; color_parks = '#cceac4'
    color_buildings_osm = '#d8cabc' # Renamed to avoid conflict
    color_roads = '#aaaaaa'; color_rail = '#a0a0a0';color_place_text = '#36454F'  # Charcoal for place labels
    
    # Define bus stop color
    color_bus_stops_blue = '#1E90FF' # Dodger blue for all bus stops


    
    # Define NEW colors for nearby stops
    color_nearby_rahoon_stops = '#32CD32'  # Lime Green
    color_nearby_portershed_stops = '#FFD700' # Gold (or choose another distinct color like a different shade of green)
    nearby_stop_marker_size = 35 # Slightly larger than general, smaller than POIs
  

    # Set background
    ax.set_facecolor(color_land)

    # Define approximate z-orders
    zorder_landuse=1; zorder_water_poly=2; zorder_parks=3; zorder_buildings_layer=4 # General buildings layer
    zorder_waterways=5; zorder_railways=6; zorder_roads=7;
    zorder_bus_stops_plot = 8    # Z-order for general bus stops
    zorder_nearby_stops_plot = zorder_bus_stops_plot + 0.1 
    zorder_place_text = 9        # Z-order for general place name labels

    # Z-orders for the specific B422 building highlight - Portershed
    zorder_building_b422_point = 10  
    zorder_building_b422_text = 11  

    # Z-orders for the specific 'Rahoon' place highlight
    zorder_rahoon_place_point = 10 
    zorder_rahoon_place_text = 11  


    zorder_boundary = 12   # Boundary should be having highest zorder to frame everything
    

    # Plot base layers
    if 'landuse' in galway_gdfs: galway_gdfs['landuse'].plot(ax=ax, column='fclass', categorical=True, cmap='Pastel2', alpha=0.4, zorder=zorder_landuse)
    if 'water_poly' in galway_gdfs: galway_gdfs['water_poly'].plot(ax=ax, color=color_water, edgecolor='none', zorder=zorder_water_poly)
    if 'landuse' in galway_gdfs and 'fclass' in galway_gdfs['landuse'].columns:
        parks_gdf = galway_gdfs['landuse'][galway_gdfs['landuse']['fclass'] == 'park']
        if not parks_gdf.empty: parks_gdf.plot(ax=ax, color=color_parks, edgecolor='none', zorder=zorder_parks)
    if 'buildings' in galway_gdfs: galway_gdfs['buildings'].plot(ax=ax, facecolor=color_buildings_osm, alpha=0.7, lw=0.5, edgecolor=color_buildings_osm, zorder=zorder_buildings_layer)
    if 'waterways' in galway_gdfs: galway_gdfs['waterways'].plot(ax=ax, color=color_water, linewidth=1.0, zorder=zorder_waterways)
    if 'railways' in galway_gdfs:
        galway_gdfs['railways'].plot(ax=ax, color='#ffffff', linewidth=2.0, linestyle='-', zorder=zorder_railways)
        galway_gdfs['railways'].plot(ax=ax, color=color_rail, linewidth=1.0, linestyle='-', zorder=zorder_railways + 0.1)
    if 'roads' in galway_gdfs: galway_gdfs['roads'].plot(ax=ax, color=color_roads, linewidth=0.8, zorder=zorder_roads)

    # --- Plot ALL Bus Stops from gstops_df_v1 as BLUE DOTS ---
    if bus_stops_gdf is not None and not bus_stops_gdf.empty:
        bus_stops_gdf.plot(
            ax=ax,
            color=color_bus_stops_blue, # Use the defined blue color
            marker='o',
            markersize=15,             
            edgecolor='black',        
            linewidth=0.5,
            alpha=0.9,
            zorder=zorder_bus_stops_plot, # Ensure they are on top of most layers
            label='Bus Stops (All)'
        )
        print(f"Plotted {len(bus_stops_gdf)} bus stops from gstops_df_v1 as blue dots.")
    else:
        print("No bus stops from gstops_df_v1 to plot.")

    
    # --- *** NEW: Plot Bus Stops Near Rahoon *** ---
    if bus_stops_near_rahoon_gdf is not None and not bus_stops_near_rahoon_gdf.empty:
        bus_stops_near_rahoon_gdf.plot(
            ax=ax,
            color=color_nearby_rahoon_stops,
            marker='o',
            markersize=nearby_stop_marker_size, # Use new smaller size
            edgecolor='black',
            linewidth=0.7,
            alpha=0.9,
            zorder=zorder_nearby_stops_plot, # Higher z-order
            label='Stops near Rahoon'
        )
        print(f"Plotted {len(bus_stops_near_rahoon_gdf)} bus stops near Rahoon.")

            # --- *** NEW: Plot Bus Stops Near Portershed *** ---
    if bus_stops_near_portershed_gdf is not None and not bus_stops_near_portershed_gdf.empty:
        bus_stops_near_portershed_gdf.plot(
            ax=ax,
            color=color_nearby_portershed_stops,
            marker='o',
            markersize=nearby_stop_marker_size, # Use new smaller size
            edgecolor='black',
            linewidth=0.7,
            alpha=0.9,
            zorder=zorder_nearby_stops_plot, # Higher z-order
            label='Stops near Portershed'
        )
        print(f"Plotted {len(bus_stops_near_portershed_gdf)} bus stops near Portershed.")
    

    

    # --- Plot Place Names (No Circles) ---
    if galway_places_summary_df is not None and not galway_places_summary_df.empty:
        print(f"Plotting {len(galway_places_summary_df)} place names...")
        plotted_place_names_map = set()
        for idx, row in galway_places_summary_df.iterrows():
            label = row['place_name']; point_x = row['longitude']; point_y = row['latitude']
            if label not in plotted_place_names_map:
                ax.text(point_x, point_y + 0.0002, label, fontsize=8, color=color_place_text,
                        ha='center', va='bottom', zorder=zorder_place_text, fontweight='normal',
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground='w')])
                plotted_place_names_map.add(label)
        print("Place names plotted.")

    # --- *** PLOT B422 BUILDING - PORTERSHED *** ---
    if 'galway_buildings_summary_df1' in locals() and galway_buildings_summary_df1 is not None and not galway_buildings_summary_df1.empty:
        building_point_color = '#FF5733' # Orange
        building_text_color = '#000000'  # Black
        plotted_b422 = False
        # Ensure B422 exists in your dataframe's index
        if 'B422' in galway_buildings_summary_df1.index:
            row = galway_buildings_summary_df1.loc['B422']
            point_x = row['longitude']
            point_y = row['latitude']
            building_name = row['building_name']
            
            # Plot orange circle for B422
            plt.scatter(point_x, point_y, s=60, color=building_point_color, edgecolor='black', # Increased size (s=60)
                        linewidth=1, alpha=0.9, zorder=zorder_building_b422_point, label=f'Building: {building_name}')
            
            # Plot name label for B422
            ax.text(point_x, point_y + 0.0003, building_name, fontsize=7, color=building_text_color, 
                    ha='center', va='bottom', zorder=zorder_building_b422_text, fontweight='bold',
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground='white')])
            plotted_b422 = True
            print(f"Plotted orange circle and name label for building B422 ('{building_name}').")
        else:
            print("Building B422 not found in the DataFrame 'galway_buildings_summary_df1'.")
    else:
        print("DataFrame 'galway_buildings_summary_df1' not available for plotting B422.")
    # --- *** END OF B422 PLOTTING CODE *** ---   



    # --- *** PLOT SPECIFIC PLACE 'RAHOON' *** ---
    if 'rahoon_place_id' in locals() and rahoon_place_id is not None and \
       'galway_places_summary_df1' in locals() and galway_places_summary_df1 is not None and \
       not galway_places_summary_df1.empty:

        if rahoon_place_id in galway_places_summary_df1.index:
            place_row = galway_places_summary_df1.loc[rahoon_place_id]
            point_x = place_row['longitude']
            point_y = place_row['latitude']
            place_name_label = place_row['place_name'] 

            place_point_color = '#9400D3' # Dark Violet 
            place_text_color = '#000000'   # Black

            # Plot distinct circle for 'Rahoon'
            plt.scatter(point_x, point_y, s=70, color=place_point_color, edgecolor='black', 
                        linewidth=1, alpha=0.9, zorder=zorder_rahoon_place_point, label=f'Place: {place_name_label}')

            # Plot name label for 'Rahoon'
            ax.text(point_x, point_y + 0.00035, place_name_label, fontsize=7.5, color=place_text_color,
                    ha='center', va='bottom', zorder=zorder_rahoon_place_text, fontweight='bold',
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground='white')])
            print(f"Plotted distinct circle and name label for place: '{place_name_label}' (ID: {rahoon_place_id}).")
        else:
            print(f"Place with ID '{rahoon_place_id}' (expected to be Rahoon) not found in galway_places_summary_df1.index for plotting.")
    else:
        print("Rahoon was not identified or 'galway_places_summary_df1' is not available for plotting specific place.")
    # --- *** END OF 'RAHOON' PLOTTING CODE *** ---


    # Plot boundary outline for context last
    boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, linestyle='--', zorder=zorder_boundary)

    # --- Set Map Bounds ---
    if 'roads' in galway_gdfs and not galway_gdfs['roads'].empty:
        minx, miny, maxx, maxy = galway_gdfs['roads'].total_bounds
    else:
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
    margin_factor = 0.02
    margin_x = (maxx - minx) * margin_factor
    margin_y = (maxy - miny) * margin_factor
    ax.set_xlim(minx - margin_x, maxx + margin_x)
    ax.set_ylim(miny - margin_y, maxy + margin_y)
    ax.set_aspect('equal', adjustable='box')

    # (Inside your main plotting cell [12], within the `try` block, in the
#  "PLOT THE SHORTEST JOURNEY PATH" section)

# --- *** PLOT THE SHORTEST JOURNEY PATH *** ---
    if shortest_journey_path_details:
        print("\nPlotting the shortest journey path...")
        
        # Define styles for the path
        walk_color = 'dimgrey' # Darker grey 
        walk_linestyle = '--'
        walk_linewidth = 1.8
        
  
        transit_color = '#E60000' 
        transit_linestyle = '-'
        transit_linewidth = 3.0
        transit_alpha = 0.85 # transparency

        path_zorder = zorder_boundary + 1 

        # Helper to get coordinates 
        def get_node_coords(node_id, graph, places_gdf, buildings_gdf, stops_gdf):
    
            # return (longitude, latitude) for any given node ID
            if graph.has_node(node_id):
                node_data = graph.nodes[node_id]
                if 'x' in node_data and 'y' in node_data:
                    return node_data['x'], node_data['y']
                
                # Fallback logic using 'type' and 'id' if present in G.nodes[node_id]
                node_type = node_data.get('type')
                original_id = node_data.get('id', node_id) # 

                if node_type == 'place' and places_gdf is not None:
                    # places_gdf index is the ID we need (e.g. 'P1', 'P2')
                    if original_id in places_gdf.index:
                        geom = places_gdf.loc[original_id].geometry
                        return geom.x, geom.y
                elif node_type == 'building' and buildings_gdf is not None:
                    # buildings_gdf index is the ID (e.g. 'B1', 'B22')
                    if original_id in buildings_gdf.index:
                        geom = buildings_gdf.loc[original_id].geometry
                        return geom.x, geom.y
                elif node_type == 'bus_stop' and stops_gdf is not None and 'stop_id' in stops_gdf.columns:
                    # 'original_id' value to match in 'stop_id' column
                    stop_row = stops_gdf[stops_gdf['stop_id'] == original_id]
                    if not stop_row.empty:
                        geom = stop_row.iloc[0].geometry
                        return geom.x, geom.y
            return None, None


        try:
            # 1. Origin POI to Origin Bus Stop (Walk)
            o_poi_x, o_poi_y = get_node_coords(shortest_journey_path_details['origin_poi_node_id'], G, galway_places_summary_df1, galway_buildings_summary_df1, bus_stops_gdf)
            o_bs_x, o_bs_y = get_node_coords(shortest_journey_path_details['origin_bus_stop_id'], G, galway_places_summary_df1, galway_buildings_summary_df1, bus_stops_gdf)
            walk_path_label_set = False
            if o_poi_x and o_bs_x:
                ax.plot([o_poi_x, o_bs_x], [o_poi_y, o_bs_y], color=walk_color, linestyle=walk_linestyle, 
                        linewidth=walk_linewidth, zorder=path_zorder, label='Shortest Path (Walk)', alpha=transit_alpha)
                walk_path_label_set = True
        

            # 2. Transit Segment (Bus)
            transit_nodes_sequence = shortest_journey_path_details['transit_stop_sequence_ids']
            if len(transit_nodes_sequence) >= 2:
                transit_path_label_set = False
                for i in range(len(transit_nodes_sequence) - 1):
                    from_node_id = transit_nodes_sequence[i]
                    to_node_id = transit_nodes_sequence[i+1]
                    from_x, from_y = get_node_coords(from_node_id, G, galway_places_summary_df1, galway_buildings_summary_df1, bus_stops_gdf)
                    to_x, to_y = get_node_coords(to_node_id, G, galway_places_summary_df1, galway_buildings_summary_df1, bus_stops_gdf)
                    
                    if from_x and to_x:
                        current_label = 'Shortest Path (Transit)' if not transit_path_label_set else None
                        line, = ax.plot([from_x, to_x], [from_y, to_y], color=transit_color, linestyle=transit_linestyle, 
                                    linewidth=transit_linewidth, zorder=path_zorder, label=current_label, alpha=transit_alpha)
                        # Add path effect
                        line.set_path_effects([path_effects.Stroke(linewidth=transit_linewidth + 1.5, foreground='white', alpha=0.6),
                                            path_effects.Normal()])
                        if not transit_path_label_set: transit_path_label_set = True
                

            # 3. Destination Bus Stop to Destination POI (Walk)
            d_bs_x, d_bs_y = get_node_coords(shortest_journey_path_details['destination_bus_stop_id'], G, galway_places_summary_df1, galway_buildings_summary_df1, bus_stops_gdf)
            d_poi_x, d_poi_y = get_node_coords(shortest_journey_path_details['destination_poi_node_id'], G, galway_places_summary_df1, galway_buildings_summary_df1, bus_stops_gdf)
            if d_bs_x and d_poi_x:
                current_walk_label = 'Shortest Path (Walk)' if not walk_path_label_set else None
                ax.plot([d_bs_x, d_poi_x], [d_bs_y, d_poi_y], color=walk_color, linestyle=walk_linestyle, 
                        linewidth=walk_linewidth, zorder=path_zorder, label=current_walk_label, alpha=transit_alpha)
                
                
            # Update legend to ensure new labels are included and no duplicates
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) 
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
            print("Shortest journey path plotted (if details were available).")

        except Exception as e_plot:
            print(f"Error during shortest path plotting: {e_plot}")
            # import traceback # Already imported at top level of notebook usually
            traceback.print_exc()
    else:
        print("\nNo shortest journey path details available to plot.")


    # Final plot adjustments
    ax.set_title(f"Galway Map with Bus Stops (from gstops_df_v1)", color='black', fontsize=16)
    plt.legend(loc='upper right') # add a legend
    plt.axis('off')
    plt.tight_layout()
    plt.show()
 

except FileNotFoundError as e:
    print(f"\n--- File Error ---\n{e}\nPlease ensure file paths are correct.")
except ImportError as e:
    print(f"\n--- Import Error Occurred ---\nError: {e}\nPlease ensure required libraries are installed.")
except ValueError as e:
    print(f"\n--- Value Error ---\n{e}")
except Exception as e:
    print(f"\n--- An Unexpected Error Occurred ---\nError: {e}")
    import traceback
    traceback.print_exc()


# ### 14. Save Artifacts

# In[123]:


# import pickle

# Define file paths for all relevant artifacts
places_summary_path = os.path.join(artifact_dir, "galway_places_summary_df1.csv")
buildings_summary_path = os.path.join(artifact_dir, "galway_buildings_summary_df1.csv")
bus_stops_path = os.path.join(artifact_dir, "bus_stops_gdf.csv")
bus_timetables_path = os.path.join(artifact_dir, "bus_timetables.csv")
bus_routes_path = os.path.join(artifact_dir, "bus_routes_gdf.csv")
graph_pickle_path = os.path.join(artifact_dir, "G.gpickle")
place_to_building_connections_path = os.path.join(artifact_dir, "place_to_building_connections_df.csv")



if os.path.isdir(artifact_dir):
    # Save galway_places_summary_df1 (replaces if file exists)
    if galway_places_summary_df1 is not None:
        galway_places_summary_df1.to_csv(places_summary_path)
    # Save galway_buildings_summary_df1 (replaces if file exists)
    if galway_buildings_summary_df1 is not None:
        galway_buildings_summary_df1.to_csv(buildings_summary_path)
    # Save bus_stops_gdf (replaces if file exists)
    if bus_stops_gdf is not None:
        bus_stops_gdf.to_csv(bus_stops_path)
    # Save bus_timetables (replaces if file exists)
    if bus_timetables is not None:
        bus_timetables.to_csv(bus_timetables_path)
    # Save bus_routes_gdf (replaces if file exists)
    if bus_routes_gdf is not None:
        bus_routes_gdf.to_csv(bus_routes_path)
    # Save the graph G as a pickle file (replaces if file exists)
    if 'G' in locals() and G is not None:
        with open(graph_pickle_path, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    if place_to_building_connections_df is not None:
        place_to_building_connections_df.to_csv(place_to_building_connections_path)
    print("Artifacts saved successfully.")


# ### 15. Load the files (reuse the files when no changes in the data required)
# 
# Code commented out, run when necessary

# In[122]:


# # Define file paths for all relevant artifacts
# places_summary_path = os.path.join(artifact_dir, "galway_places_summary_df1.csv")
# buildings_summary_path = os.path.join(artifact_dir, "galway_buildings_summary_df1.csv")
# bus_stops_path = os.path.join(artifact_dir, "bus_stops_gdf.csv")
# bus_timetables_path = os.path.join(artifact_dir, "bus_timetables.csv")
# bus_routes_path = os.path.join(artifact_dir, "bus_routes_gdf.csv")
# graph_pickle_path = os.path.join(artifact_dir, "G.gpickle")
# place_to_building_connections_path = os.path.join(artifact_dir, "place_to_building_connections_df.csv")

# # Load the dataframes and graph from artifact_dir if available
# if os.path.isdir(artifact_dir):

#     # Read places summary, preserving index if present
#     if os.path.exists(places_summary_path):
#         galway_places_summary_df1 = pd.read_csv(places_summary_path, index_col=0)
#     else:
#         galway_places_summary_df1 = None

#     # Read buildings summary, preserving index (e.g., B1, B2, ...)
#     if os.path.exists(buildings_summary_path):
#         galway_buildings_summary_df1 = pd.read_csv(buildings_summary_path, index_col=0)
#     else:
#         galway_buildings_summary_df1 = None

#     # Read bus stops, preserving index if present
#     if os.path.exists(bus_stops_path):
#         bus_stops_gdf = pd.read_csv(bus_stops_path, index_col=0)
#     else:
#         bus_stops_gdf = None

#     # Read bus timetables, preserving index if present
#     if os.path.exists(bus_timetables_path):
#         bus_timetables = pd.read_csv(bus_timetables_path, index_col=0)
#     else:
#         bus_timetables = None

#     # Read bus routes, preserving index (e.g., BR1, BR2, ...)
#     if os.path.exists(bus_routes_path):
#         bus_routes_gdf = pd.read_csv(bus_routes_path, index_col=0)
#     else:
#         bus_routes_gdf = None

#     # Load the graph G from pickle if available
#     if os.path.exists(graph_pickle_path):
#         with open(graph_pickle_path, "rb") as f:
#             G = pickle.load(f)
#     else:
#         G = None
#     print("Artifacts loaded successfully.")
# else:
#     all_timetables_df = None
#     galway_places_summary_df1 = None
#     galway_buildings_summary_df1 = None
#     bus_stops_gdf = None
#     bus_timetables = None
#     bus_routes_gdf = None
#     G = None
#     print("No artifact directory found. Dataframes and graph not loaded.")

# if os.path.isdir(artifact_dir):
#     if os.path.exists(place_to_building_connections_path):
#         place_to_building_connections_df = pd.read_csv(place_to_building_connections_path, index_col=0)
#     else:
#         place_to_building_connections_df = None
# else:
#     place_to_building_connections_df = None

# print(f"Graph G loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
# print(f"DataFrame place_to_building_connections_df loaded: {place_to_building_connections_df.shape}")
# display(place_to_building_connections_df.head())


# ### 16. References

# https://libguides.ucd.ie/gisguide/findspatialdata 
# 
# 
# https://download.geofabrik.de/europe/ireland-and-northern-ireland.html
# 
# https://galway-bus.apis.ie/gstoptimes/#g-stop-time-schema
# 
# https://tilburgsciencehub.com/topics/visualization/data-visualization/graphs-charts/grammar-of-graphics-ggplot2/
# 
# https://python.langchain.com/docs/integrations/chat/ollama/
# 
# https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/
# 
# https://react-lm.github.io/ 
