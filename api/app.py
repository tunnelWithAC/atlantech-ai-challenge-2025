import os

from flask import Flask, request, jsonify
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

# Save galway_places_summary_df1 (replaces if file exists)
galway_places_summary_df1 = pd.read_csv(places_summary_path)
galway_buildings_summary_df1 = pd.read_csv(buildings_summary_path)

bus_stops_gdf = pd.read_csv(bus_stops_path)
bus_timetables = pd.read_csv(bus_timetables_path)
bus_routes_gdf = pd.read_csv(bus_routes_path)


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
    ### SETUP
    MAX_ACCESS_DISTANCE_METERS = 800

    """
    Barna -> Doughiska
    Knocknacarra -> Knocknacarragh
    Oranmore -> Ballybrit
    """
    ###

    building_ids = {
        '1': 'Salthill Lodge',
        '2': 'Eyre Square Centre',
        '3': 'Galway Cathedral',
        '4': 'Merchants Quay',
        '5': 'Spanish Arch Car Park',
        '6': 'Merlin Park',
        '7': 'Portershed'
    }

    building_id = request.args.get('building_id', '').lower()
    
    if not building_id:
        return jsonify({"error": "building_id parameter is required"}), 400
    
    if building_id not in building_ids.keys():
        return jsonify({"error": f"No data found for building id: {building_ids}"}), 404
    place_node_id = building_ids[building_id]  # This is the specific node ID for Rahoon

    def _get_place_accessibility_score(__place_node_id, building_node_id):
        # --- Call get_nearby_stops for Rahoon ---
        __place_node_id, place_nearby_stops = get_nearby_stops(G, __place_node_id, MAX_ACCESS_DISTANCE_METERS)

        # --- Call get_nearby_stops for Portershed ---
        building_node_id, building_nearby_stops = get_nearby_stops(G, building_node_id, MAX_ACCESS_DISTANCE_METERS)

        place_to_building_connections_df, no_conn_df = get_enriched_transit_connections(
            G_input=G,
            bus_timetables_input=bus_timetables,
            origin_poi_name_param=__place_node_id,
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
        return calculate_and_display_accessibility_score(place_to_building_connections_df)

    renmore_accessibility_score = _get_place_accessibility_score(place_node_id, "Renmore")
    knocknacrra_accessibility_score = _get_place_accessibility_score(place_node_id, "Knocknacarragh")
    ballybrit_accessibility_score = _get_place_accessibility_score(place_node_id, "Ballybrit")

    accessibility_score = (renmore_accessibility_score + knocknacrra_accessibility_score + ballybrit_accessibility_score) / 3

    results = {
        'scores': {
            "barna": renmore_accessibility_score,
            "knocknacarra": knocknacrra_accessibility_score,
            "oranmore": ballybrit_accessibility_score
        },
        'accessibility_score': accessibility_score
    }
    return jsonify(results)


if __name__ == '__main__':
    app.run(port=8080, debug=True) 