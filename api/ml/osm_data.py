# Libraries
import pandas as pd
pd.set_option('display.max_colwidth', None) # display full column width
import os
import requests
from IPython.display import display
import zipfile
from selenium import webdriver # Used for scraping bus timetables from buseireann.ie
import time
import re # For regular expressions
from thefuzz import process, fuzz
import networkx as nx
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import logging
import warnings

'''
Set up graph so that get_nearby_stops() can be called
'''
def setup_graph():
    # TODO load dataframes from files

    # --- Helper Function for Haversine Distance ---
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        try:
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        except (ValueError, TypeError):
            return float('inf')
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c * 1000


    # --- 1. Initialize Graph ---
    G = nx.DiGraph()
    print("Graph initialized.")

    # --- Step 1a: Add Place Nodes ---
    print("\nAdding place nodes (general POIs)...")
    for index, row in galway_places_summary_df1.iterrows():
        place_node_id = row['place_name']
        G.add_node(place_node_id, type='place', name=row['place_name'], latitude=row['latitude'], longitude=row['longitude'])
    print(f"Nodes after general places: {G.number_of_nodes()}")

    # --- Step 1b: Add Building Nodes---
    print("\nAdding building nodes...")
    for index, row in galway_buildings_summary_df1.iterrows():
        building_node_id = row['building_name']
        G.add_node(building_node_id, type='building', name=row['building_name'], osm_id=row.get('osm_id'),
                building_type=row.get('building_type'), latitude=row['latitude'], longitude=row['longitude'])
    print(f"Nodes after adding buildings: {G.number_of_nodes()}")

    # --- Step 1c: Add Unique Bus Stop Nodes ---
    print("\nAdding unique bus stop nodes...")
    added_stop_ids = set()
    for index, row in bus_stops_gdf.iterrows():
        stop_id = row['stop_id']
        if stop_id not in added_stop_ids:
            G.add_node(stop_id, type='bus_stop', name=row['stop_name'], latitude=row['stop_lat'], longitude=row['stop_lon'],
                    direction=row.get('direction'), original_route_id=row.get('route_id'), geometry=row.get('geometry'),
                    norm_explicit=row.get('stop_name_norm_explicit'), norm_expanded=row.get('stop_name_norm_expanded'))
            added_stop_ids.add(stop_id)
    print(f"Total nodes after bus stops: {G.number_of_nodes()}")

    # --- Step 2: Add Access/Egress Edges
    print("\nAdding access/egress edges...")
    MAX_ACCESS_DISTANCE_METERS = 800

    access_edge_count = 0
    place_nodes_data = {node_id: data for node_id, data in G.nodes(data=True) if data.get('type') == 'place'}
    building_nodes_data = {node_id: data for node_id, data in G.nodes(data=True) if data.get('type') == 'building'}
    bus_stop_nodes_data = {node_id: data for node_id, data in G.nodes(data=True) if data.get('type') == 'bus_stop'}
    for place_node_id, place_data in place_nodes_data.items():
        place_lat = place_data.get('latitude'); place_lon = place_data.get('longitude')
        if place_lat is None or place_lon is None: continue
        for stop_node_id, stop_data in bus_stop_nodes_data.items():
            stop_lat = stop_data.get('latitude'); stop_lon = stop_data.get('longitude')
            if stop_lat is None or stop_lon is None: continue
            walking_distance_m = haversine(place_lat, place_lon, stop_lat, stop_lon)
            if walking_distance_m <= MAX_ACCESS_DISTANCE_METERS:
                edge_attrs = {'type':'access_egress', 'mode':'walk', 'distance_m': walking_distance_m}
                G.add_edge(place_node_id, stop_node_id, **edge_attrs); G.add_edge(stop_node_id, place_node_id, **edge_attrs)
                access_edge_count += 2;

    for building_node_id, building_data in building_nodes_data.items():
        building_lat = building_data.get('latitude'); building_lon = building_data.get('longitude')
        if building_lat is None or building_lon is None: continue
        for stop_node_id, stop_data in bus_stop_nodes_data.items():
            stop_lat = stop_data.get('latitude'); stop_lon = stop_data.get('longitude')
            if stop_lat is None or stop_lon is None: continue
            walking_distance_m = haversine(building_lat, building_lon, stop_lat, stop_lon)
            if walking_distance_m <= MAX_ACCESS_DISTANCE_METERS:
                edge_attrs = {'type':'access_egress', 'mode':'walk', 'distance_m': walking_distance_m}
                G.add_edge(building_node_id, stop_node_id, **edge_attrs); G.add_edge(stop_node_id, building_node_id, **edge_attrs)
                access_edge_count += 2;
    print(f"Added {access_edge_count} access/egress edges in total.")

    # --- Step 3: Add Directed Transit Edges ---
    print("\nAdding directed transit edges...")
    transit_edge_count = 0
    valid_graph_stop_node_ids = {node_id for node_id, data in G.nodes(data=True) if data.get('type') == 'bus_stop'}
    print(f"Debug: Found {len(valid_graph_stop_node_ids)} valid bus_stop nodes in the graph for transit edges.")
    #
    print(f"Debug: bus_timetables has {len(bus_timetables)} rows for transit edge creation.")

    for route_id_timetable, group in bus_timetables.groupby('route_id'): # Using bus_timetables
        route_stops = group.sort_values(by='stop_order_on_route')
        for i in range(len(route_stops) - 1):
            from_stop_id_mapped = route_stops.iloc[i]['stop_id_mapped']
            to_stop_id_mapped = route_stops.iloc[i+1]['stop_id_mapped']
            from_node_exists = from_stop_id_mapped in valid_graph_stop_node_ids
            to_node_exists = to_stop_id_mapped in valid_graph_stop_node_ids

            if from_node_exists and to_node_exists:
                from_stop_node_data = G.nodes[from_stop_id_mapped]
                to_stop_node_data = G.nodes[to_stop_id_mapped]
                from_lat, from_lon = from_stop_node_data.get('latitude'), from_stop_node_data.get('longitude')
                to_lat, to_lon = to_stop_node_data.get('latitude'), to_stop_node_data.get('longitude')

                if None not in [from_lat, from_lon, to_lat, to_lon]:
                    segment_distance_m = haversine(from_lat, from_lon, to_lat, to_lon)
                    edge_attrs = {'type':'transit', 'route_id':route_id_timetable, 'hop_count':1, 'distance_m':segment_distance_m}
                    G.add_edge(from_stop_id_mapped, to_stop_id_mapped, **edge_attrs)
                    transit_edge_count += 1
    print(f"Added {transit_edge_count} directed transit edges.")

    # --- Final Graph Summary ---
    print("\n--- Graph Construction Complete ---")
    print(f"Total nodes in graph: {G.number_of_nodes()}")
    print(f"Total edges in graph: {G.number_of_edges()}")
    node_types_list = [data.get('type', 'Unknown') for node, data in G.nodes(data=True)]
    print("\nNode type counts:\n", pd.Series(node_types_list).value_counts())
    edge_summary = []
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'Unknown')
        edge_summary.append(f"{edge_type}_{data.get('mode', '')}" if edge_type == 'access_egress' else edge_type)
    print("\nEdge type counts:\n", pd.Series(edge_summary).value_counts())

    # --- check edges with data ---
    print("\n--- Explicit check of G.edges(data=True) ---")
    all_edges_with_data = list(G.edges(data=True))
    if not all_edges_with_data: print("G.edges(data=True) is EMPTY.")
    else:
        print(f"Found {len(all_edges_with_data)} edges with data. Sample:")
        for i, edge_tuple in enumerate(all_edges_with_data): 
            print(edge_tuple)
            if i >= 2: 
                break 

"""
"""
def get_nearby_stops():
    # TODO: load graph from pickle or function rather than locals
    if 'G' in locals() and G.number_of_nodes() > 0:
        rahoon_node_id = "Rahoon"
        portershed_node_id = "Portershed a Dó" # the building_name used as node ID

        print(f"--- Checking for exact node ID: '{rahoon_node_id}' ---")
        if G.has_node(rahoon_node_id):
            print(f"Node '{rahoon_node_id}' FOUND.")
            print(f"Attributes: {G.nodes[rahoon_node_id]}")
        else:
            print(f"Node '{rahoon_node_id}' NOT FOUND by exact ID.")

        print(f"\n--- Checking for exact node ID: '{portershed_node_id}' ---")
        if G.has_node(portershed_node_id):
            print(f"Node '{portershed_node_id}' FOUND.")
            print(f"Attributes: {G.nodes[portershed_node_id]}")
        else:
            print(f"Node '{portershed_node_id}' NOT FOUND by exact ID.")
    else:
        print("Graph G is not defined or is empty.")



    # --- Step 3.1: Identify Relevant Stops (Nearby Stops) ---

    # Define the node IDs 
    place_of_interest_rahoon = rahoon_node_id
    place_of_interest_portershed = portershed_node_id

    # Define the proximity threshold (user input)
    PROXIMITY_THRESHOLD_METERS = MAX_ACCESS_DISTANCE_METERS 

    print(f"Using proximity threshold: {PROXIMITY_THRESHOLD_METERS} meters.")

    def __get_nearby_stops(graph, poi_node_id, max_distance):
        """
        Finds bus stops connected to a POI node via 'access_egress' edges
        within a specified maximum distance, and returns their IDs and distances.
        """
        nearby_stops_info = [] # list to store dictionaries

        if not graph.has_node(poi_node_id):
            print(f"Warning: POI node '{poi_node_id}' not found in the graph.")
            return nearby_stops_info # Return empty list

        # We are interested in edges FROM the POI TO a bus stop for "access"
        # The graph stores bi-directional access/egress, so out_edges from POI is sufficient
        # to find connected bus stops.
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
                
        return nearby_stops_info

    place_nearby_stops = __get_nearby_stops(G, place_of_interest_rahoon, PROXIMITY_THRESHOLD_METERS)
    building_nearby_stops = __get_nearby_stops(G, place_of_interest_portershed, PROXIMITY_THRESHOLD_METERS)
    place_nearby_stop_ids = [info['stop_id'] for info in place_nearby_stops]
    building_nearby_stop_ids = [info['stop_id'] for info in building_nearby_stops]
    print(f"Proximity Threshold: {PROXIMITY_THRESHOLD_METERS} meters.")
    print(f"place_nearby_stops : {place_nearby_stops}")
    print(f"building_nearby_stops: {building_nearby_stops}")
    print(f"place_nearby_stop_ids: {place_nearby_stop_ids}")
    print(f"building_nearby_stop_ids: {building_nearby_stop_ids}")

    return {
        'place_nearby_stops': place_nearby_stops, 
        'building_nearby_stops': building_nearby_stops, 
        'place_nearby_stop_ids': place_nearby_stop_ids, 
        'building_nearby_stop_ids': building_nearby_stop_ids
    }

# try:
#     from IPython.display import display
# except ImportError:
#     display = print # Fallback to simple print if not in IPython

def direct_transit_conn_between_places(G, bus_timetables, origin_poi_name, 
                                       origin_nearby_stops_info, destination_poi_name, 
                                       destination_nearby_stops_info):
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

    # --- Initialization and Input Preparation ---
    if not origin_nearby_stops_info or not destination_nearby_stops_info:
        print(f"Warning: Input list 'origin_nearby_stops_info' or 'destination_nearby_stops_info' is empty.")

    origin_target_stop_ids = [info['stop_id'] for info in origin_nearby_stops_info]
    destination_target_stop_ids = [info['stop_id'] for info in destination_nearby_stops_info]

    print(f"Origin ({origin_poi_name}) Target Stops: {origin_target_stop_ids}")
    print(f"Destination ({destination_poi_name}) Target Stops: {destination_target_stop_ids}")

    direct_transit_connections = []
    origin_routes_no_destination_connection = []

    # --- Main Analysis Logic ---
    required_cols = ['route_id', 'stop_id_mapped', 'stop_order_on_route']
    if not isinstance(bus_timetables, pd.DataFrame) or not all(col in bus_timetables.columns for col in required_cols):
        print(f"Error: 'bus_timetables' DataFrame is not valid or is missing required columns: {required_cols}")
        return None, None 
    if not hasattr(G, 'edges'):
        print("Error: NetworkX graph 'G' is not valid.")
        return None, None
    if not origin_target_stop_ids or not destination_target_stop_ids:
        print(f"Error: Either Origin ({origin_poi_name}) or Destination ({destination_poi_name}) target stop lists are empty. Cannot proceed.")
        # Return empty DataFrames as per function definition
        return pd.DataFrame(columns=['origin_poi', 'destination_poi', 'route_id', 'origin_stop_id', 
                                     'destination_stop_id', 'origin_stop_order', 'destination_stop_order', 
                                     'hops', 'transit_distance_m']), \
               pd.DataFrame(columns=['origin_stop_id', 'route_id', 'message'])


    print(f"\nAnalyzing routes from {origin_poi_name} area stops towards {destination_poi_name} area stops...")

    for r_stop_id in origin_target_stop_ids:
        routes_serving_r_stop_df = bus_timetables[bus_timetables['stop_id_mapped'] == r_stop_id]
        
        if routes_serving_r_stop_df.empty:
            continue

        unique_routes_for_this_r_stop = routes_serving_r_stop_df['route_id'].unique()

        for route_id_val in unique_routes_for_this_r_stop:
            route_sequence_df = bus_timetables[bus_timetables['route_id'] == route_id_val].sort_values(by='stop_order_on_route')
            
            if route_sequence_df.empty: 
                continue 

            stop_to_order_map = pd.Series(route_sequence_df['stop_order_on_route'].values, index=route_sequence_df['stop_id_mapped']).to_dict()
            
            if r_stop_id not in stop_to_order_map:
                continue
            r_stop_order = stop_to_order_map[r_stop_id]
            
            found_connection_on_this_route_for_r_stop = False

            for p_stop_id in destination_target_stop_ids:
                if p_stop_id in stop_to_order_map:
                    p_stop_order = stop_to_order_map[p_stop_id]
                    
                    if r_stop_order < p_stop_order:
                        found_connection_on_this_route_for_r_stop = True
                        hops = p_stop_order - r_stop_order
                        current_distance_m = 0.0
                        path_found_in_graph = True
                        
                        path_segment_df = route_sequence_df[
                            (route_sequence_df['stop_order_on_route'] >= r_stop_order) &
                            (route_sequence_df['stop_order_on_route'] <= p_stop_order)
                        ]
                        actual_stops_in_path_sequence = path_segment_df['stop_id_mapped'].tolist()
                        
                        if len(actual_stops_in_path_sequence) < 2:
                            if r_stop_id == p_stop_id:
                                current_distance_m = 0.0
                            else: 
                                path_found_in_graph = False
                        else:
                            for i in range(len(actual_stops_in_path_sequence) - 1):
                                from_s = actual_stops_in_path_sequence[i]
                                to_s = actual_stops_in_path_sequence[i+1]
                                
                                if G.has_edge(from_s, to_s):
                                    edge_data = G.get_edge_data(from_s, to_s)
                                    if edge_data.get('type') == 'transit' and edge_data.get('route_id') == route_id_val:
                                        current_distance_m += edge_data.get('distance_m', 0.0)
                                    else:
                                        path_found_in_graph = False; break
                                else:
                                    path_found_in_graph = False; break
                        
                        if not path_found_in_graph:
                            current_distance_m = None 
                            
                        connection_details = {
                            'origin_poi': origin_poi_name, 
                            'destination_poi': destination_poi_name,
                            'route_id': route_id_val,
                            'origin_stop_id': r_stop_id,
                            'destination_stop_id': p_stop_id,
                            'origin_stop_order': r_stop_order,
                            'destination_stop_order': p_stop_order,
                            'hops': hops,
                            'transit_distance_m': current_distance_m 
                        }
                        direct_transit_connections.append(connection_details)

            if not found_connection_on_this_route_for_r_stop and r_stop_id in stop_to_order_map :
                 origin_routes_no_destination_connection.append({
                     'origin_stop_id': r_stop_id,
                     'route_id': route_id_val,
                     'message': f"Route {route_id_val} serves {origin_poi_name} area stop {r_stop_id} but does not connect to any target {destination_poi_name} area stops in sequence."
                 })

    # --- Prepare Output DataFrames ---
    connections_df = pd.DataFrame()
    if direct_transit_connections:
        connections_df = pd.DataFrame(direct_transit_connections)
        connections_df = connections_df.sort_values(by=['route_id', 'origin_stop_order', 'hops'])

    no_connection_df = pd.DataFrame()
    if origin_routes_no_destination_connection:
        no_connection_df = pd.DataFrame(origin_routes_no_destination_connection).drop_duplicates()
        
    return connections_df, no_connection_df


## TODO see where the code below should live

place_to_building_connections_df, place_to_building_no_connection_df = direct_transit_conn_between_places(G, bus_timetables, place_of_interest_rahoon, place_nearby_stops, 
                                   place_of_interest_portershed, building_nearby_stops)
display(place_to_building_connections_df)


#####################################################################################
# --- Calculate Walking Distance from Origin POI (Rahoon) to its nearby bus stops ---
unique_origin_stops = place_to_building_connections_df['origin_stop_id'].unique()
origin_walking_distances_map = {}

for stop_id in unique_origin_stops:
    walking_distance = np.nan
    if G.has_node(stop_id):
        # Check for a direct access_egress edge FROM the Rahoon POI node TO the origin bus stop
        if G.has_edge(rahoon_node_id, stop_id):
            edge_data = G.get_edge_data(rahoon_node_id, stop_id)
            if edge_data.get('type') == 'access_egress':
                walking_distance = edge_data.get('distance_m', np.nan)
    else:
        print(f"Warning: Origin stop ID '{stop_id}' from place_to_building_connections_df not found in graph G.")
    origin_walking_distances_map[stop_id] = walking_distance

# place_to_building_connections_df['walking_distance_from_origin_poi_m'] = place_to_building_connections_df['origin_stop_id'].map(origin_walking_distances_map)

# unique_dest_stops = place_to_building_connections_df['destination_stop_id'].unique()
# dest_walking_distances_map = {}  # To store {dest_stop_id: walking_distance_m}

# for stop_id in unique_dest_stops:
#     walking_distance = np.nan # Default to NaN if no direct walking edge found
#     if G.has_node(stop_id): # 
#         # Check for a direct access_egress edge from the bus stop TO the Portershed building node
#         if G.has_edge(stop_id, portershed_node_id):
#             edge_data = G.get_edge_data(stop_id, portershed_node_id)
#             if edge_data.get('type') == 'access_egress': 
#                 walking_distance = edge_data.get('distance_m', np.nan)
#     else:
#         print(f"Warning: Destination stop ID '{stop_id}' from connections_df not found in graph G.")
        
#     dest_walking_distances_map[stop_id] = walking_distance
# place_to_building_connections_df['walking_distance_to_dest_poi_m'] = place_to_building_connections_df['destination_stop_id'].map(dest_walking_distances_map)

# # create a total travel cost (transit_distance + walking_distance)
# if 'transit_distance_m' in place_to_building_connections_df.columns:
#     # Convert to numeric
#     place_to_building_connections_df['numeric_origin_walk_dist'] = pd.to_numeric(place_to_building_connections_df['walking_distance_from_origin_poi_m'], errors='coerce')
#     place_to_building_connections_df['numeric_transit_dist'] = pd.to_numeric(place_to_building_connections_df['transit_distance_m'], errors='coerce')
#     place_to_building_connections_df['numeric_dest_walk_dist'] = pd.to_numeric(place_to_building_connections_df['walking_distance_to_dest_poi_m'], errors='coerce')

    
#     # Calculate total distance only if all three components are available
#     place_to_building_connections_df['total_journey_distance_m'] = place_to_building_connections_df[
#         ['numeric_origin_walk_dist', 'numeric_transit_dist', 'numeric_dest_walk_dist']
#     ].sum(axis=1, min_count=3) # min_count=3 ensures all parts are present 
        
#     print("\n--- place_to_building_connections_df with updated total journey distance (Origin Walk + Transit + Destination Walk) ---")
#     place_to_building_connections_df = place_to_building_connections_df.drop(columns=['numeric_origin_walk_dist', 'numeric_transit_dist', 'numeric_dest_walk_dist'])
#     display(place_to_building_connections_df)