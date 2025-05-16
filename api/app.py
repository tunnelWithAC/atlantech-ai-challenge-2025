import os

from flask import Flask, request, jsonify
import networkx as nx
import pandas as pd
import pickle

from ml.enriched_transit_connections import get_enriched_transit_connections
from ml.accessibility_score import calculate_and_display_accessibility_score
from ml.get_nearby_stops import get_nearby_stops

app = Flask(__name__)

# Load artifacts and create graph on start up
artifact_dir = '../artifacts'
places_summary_path = os.path.join(artifact_dir, "galway_places_summary_df1.csv")
buildings_summary_path = os.path.join(artifact_dir, "galway_buildings_summary_df1.csv")
bus_stops_path = os.path.join(artifact_dir, "bus_stops_gdf.csv")
bus_timetables_path = os.path.join(artifact_dir, "bus_timetables.csv")
bus_routes_path = os.path.join(artifact_dir, "bus_routes_gdf.csv")
graph_pickle_path = os.path.join(artifact_dir, "G.gpickle")
place_to_building_connections_path = os.path.join(artifact_dir, "place_to_building_connections_df.csv")

# if os.path.isdir(artifact_dir):
#     raise Exception("Unable to find artifacts directory")

# Save galway_places_summary_df1 (replaces if file exists)
galway_places_summary_df1 = pd.read_csv(places_summary_path)
# Save galway_buildings_summary_df1 (replaces if file exists)

galway_buildings_summary_df1 = pd.read_csv(buildings_summary_path)

bus_stops_gdf = pd.read_csv(bus_stops_path)

bus_timetables = pd.read_csv(bus_timetables_path)
bus_routes_gdf = pd.read_csv(bus_routes_path)

# Save the graph G as a pickle file (replaces if file exists)
# if 'G' in locals() and G is not None:
#     with open(graph_pickle_path, "wb") as f:
#         pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

with open(graph_pickle_path, "rb") as f:
    G = pickle.load(f)

place_to_building_connections_df = pd.read_csv(place_to_building_connections_path)

offices = {
    "platform94": {
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    },
    "portershed": {
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    },
    "parkmore": {
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    }
}


@app.route('/score', methods=['GET'])
def get_score():
    office_name = request.args.get('office_name', '').lower()
    
    if not office_name:
        return jsonify({"error": "office_name parameter is required"}), 400
    
    if office_name not in offices.keys():
        return jsonify({"error": f"No data found for office: {office_name}"}), 404

    MAX_ACCESS_DISTANCE_METERS = 800
    # TODO add as params
    place_node_id = "NUIG Education"  # This is the specific node ID for Rahoon
    building_node_id = "Ballybrit" # This is the specific node ID for Portershed

    # --- Call get_nearby_stops for Rahoon ---
    place_node_id, place_nearby_stops = get_nearby_stops(G, place_node_id, MAX_ACCESS_DISTANCE_METERS)

    # --- Call get_nearby_stops for Portershed ---
    building_node_id, building_nearby_stops = get_nearby_stops(G, building_node_id, MAX_ACCESS_DISTANCE_METERS)

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
    else:
        print("No enriched connections found or DataFrame is None.")

    # Generate accessibility score
    accessibility_score = calculate_and_display_accessibility_score(place_to_building_connections_df)

    results = {
        'scores': offices[office_name],
        'accessibility_score': accessibility_score
    }
    return jsonify(results)


if __name__ == '__main__':
    app.run(port=8080, debug=True) 