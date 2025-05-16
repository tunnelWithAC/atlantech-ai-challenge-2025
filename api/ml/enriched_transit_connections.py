import networkx as nx
import pandas as pd
import numpy as np

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

    G = G_input  # Graph reference
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
            print(
                f"Warning: Input list 'origin_nearby_stops_info_inner' or 'destination_nearby_stops_info_inner' is empty.")

        origin_target_stop_ids = [info['stop_id'] for info in origin_nearby_stops_info_inner]
        destination_target_stop_ids = [info['stop_id'] for info in destination_nearby_stops_info_inner]

        print(f"Origin ({origin_poi_name_inner}) Target Stops: {origin_target_stop_ids}")
        print(f"Destination ({destination_poi_name_inner}) Target Stops: {destination_target_stop_ids}")

        direct_transit_connections = []
        origin_routes_no_destination_connection = []

        required_cols = ['route_id', 'stop_id_mapped', 'stop_order_on_route']
        if not isinstance(bus_timetables_inner, pd.DataFrame) or not all(
                col in bus_timetables_inner.columns for col in required_cols):
            print(
                f"Error: 'bus_timetables_inner' DataFrame is not valid or is missing required columns: {required_cols}")
            return pd.DataFrame(), pd.DataFrame()  # Return empty DFs for robustness
        if not hasattr(G_inner, 'edges'):
            print("Error: NetworkX graph 'G_inner' is not valid.")
            return pd.DataFrame(), pd.DataFrame()
        if not origin_target_stop_ids or not destination_target_stop_ids:
            print(
                f"Error: Either Origin ({origin_poi_name_inner}) or Destination ({destination_poi_name_inner}) target stop lists are empty. Cannot proceed.")
            return pd.DataFrame(columns=['origin_poi', 'destination_poi', 'route_id', 'origin_stop_id',
                                         'destination_stop_id', 'origin_stop_order', 'destination_stop_order',
                                         'hops', 'transit_distance_m']), \
                pd.DataFrame(columns=['origin_stop_id', 'route_id', 'message'])

        print(
            f"\nAnalyzing routes from {origin_poi_name_inner} area stops towards {destination_poi_name_inner} area stops...")

        for r_stop_id in origin_target_stop_ids:
            routes_serving_r_stop_df = bus_timetables_inner[bus_timetables_inner['stop_id_mapped'] == r_stop_id]
            if routes_serving_r_stop_df.empty: continue
            unique_routes_for_this_r_stop = routes_serving_r_stop_df['route_id'].unique()

            for route_id_val in unique_routes_for_this_r_stop:
                route_sequence_df = bus_timetables_inner[bus_timetables_inner['route_id'] == route_id_val].sort_values(
                    by='stop_order_on_route')
                if route_sequence_df.empty: continue
                stop_to_order_map = pd.Series(route_sequence_df['stop_order_on_route'].values,
                                              index=route_sequence_df['stop_id_mapped']).to_dict()
                if r_stop_id not in stop_to_order_map: continue
                r_stop_order = stop_to_order_map[r_stop_id]
                found_connection_on_this_route_for_r_stop = False

                for p_stop_id in destination_target_stop_ids:
                    if p_stop_id in stop_to_order_map:
                        p_stop_order = stop_to_order_map[p_stop_id]
                        if r_stop_order < p_stop_order:
                            found_connection_on_this_route_for_r_stop = True;
                            hops = p_stop_order - r_stop_order
                            current_distance_m = 0.0;
                            path_found_in_graph = True
                            path_segment_df = route_sequence_df[
                                (route_sequence_df['stop_order_on_route'] >= r_stop_order) & (
                                            route_sequence_df['stop_order_on_route'] <= p_stop_order)]
                            actual_stops_in_path_sequence = path_segment_df['stop_id_mapped'].tolist()

                            if len(actual_stops_in_path_sequence) < 2:
                                if r_stop_id == p_stop_id:
                                    current_distance_m = 0.0
                                else:
                                    path_found_in_graph = False
                            else:
                                for i in range(len(actual_stops_in_path_sequence) - 1):
                                    from_s, to_s = actual_stops_in_path_sequence[i], actual_stops_in_path_sequence[
                                        i + 1]
                                    if G_inner.has_edge(from_s, to_s):
                                        edge_data = G_inner.get_edge_data(from_s, to_s)
                                        if edge_data.get('type') == 'transit' and edge_data.get(
                                                'route_id') == route_id_val:
                                            current_distance_m += edge_data.get('distance_m', 0.0)
                                        else:
                                            path_found_in_graph = False; break
                                    else:
                                        path_found_in_graph = False; break
                            if not path_found_in_graph: current_distance_m = None
                            direct_transit_connections.append({
                                'origin_poi': origin_poi_name_inner, 'destination_poi': destination_poi_name_inner,
                                'route_id': route_id_val, 'origin_stop_id': r_stop_id, 'destination_stop_id': p_stop_id,
                                'origin_stop_order': r_stop_order, 'destination_stop_order': p_stop_order,
                                'hops': hops, 'transit_distance_m': current_distance_m})
                if not found_connection_on_this_route_for_r_stop and r_stop_id in stop_to_order_map:
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
            walking_distance = np.nan  # Default
            if G.has_node(stop_id):
                # Use origin_poi_name
                if G.has_edge(origin_poi_name, stop_id):
                    edge_data = G.get_edge_data(origin_poi_name, stop_id)
                    if edge_data.get('type') == 'access_egress':
                        walking_distance = edge_data.get('distance_m', np.nan)
            else:
                print(f"Warning: Origin stop ID '{stop_id}' from connections_df not found in graph G.")
            origin_walking_distances_map[stop_id] = walking_distance
        place_to_building_connections_df['walking_distance_from_origin_poi_m'] = place_to_building_connections_df[
            'origin_stop_id'].map(origin_walking_distances_map)

        # --- Calculate Walking Distance from Destination bus stops to Destination POI ---
        unique_dest_stops = place_to_building_connections_df['destination_stop_id'].unique()
        dest_walking_distances_map = {}
        for stop_id in unique_dest_stops:
            walking_distance = np.nan  # Default
            if G.has_node(stop_id):
                # Use destination_poi_name
                if G.has_edge(stop_id, destination_poi_name):
                    edge_data = G.get_edge_data(stop_id, destination_poi_name)
                    if edge_data.get('type') == 'access_egress':
                        walking_distance = edge_data.get('distance_m', np.nan)
            else:
                print(f"Warning: Destination stop ID '{stop_id}' from connections_df not found in graph G.")
            dest_walking_distances_map[stop_id] = walking_distance
        place_to_building_connections_df['walking_distance_to_dest_poi_m'] = place_to_building_connections_df[
            'destination_stop_id'].map(dest_walking_distances_map)

        # --- Create total travel cost ---
        if 'transit_distance_m' in place_to_building_connections_df.columns:
            place_to_building_connections_df['numeric_origin_walk_dist'] = pd.to_numeric(
                place_to_building_connections_df['walking_distance_from_origin_poi_m'], errors='coerce')
            place_to_building_connections_df['numeric_transit_dist'] = pd.to_numeric(
                place_to_building_connections_df['transit_distance_m'], errors='coerce')
            place_to_building_connections_df['numeric_dest_walk_dist'] = pd.to_numeric(
                place_to_building_connections_df['walking_distance_to_dest_poi_m'], errors='coerce')

            place_to_building_connections_df['total_journey_distance_m'] = place_to_building_connections_df[
                ['numeric_origin_walk_dist', 'numeric_transit_dist', 'numeric_dest_walk_dist']
            ].sum(axis=1, min_count=3)

            # Drop temporary numeric columns
            cols_to_drop_temp = ['numeric_origin_walk_dist', 'numeric_transit_dist', 'numeric_dest_walk_dist']
            place_to_building_connections_df = place_to_building_connections_df.drop(
                columns=[col for col in cols_to_drop_temp if col in place_to_building_connections_df.columns])

        else:
            print("Warning: 'transit_distance_m' not found. Cannot calculate total_journey_distance_m.")
    else:
        print("No connections found (place_to_building_connections_df is None or empty). Skipping enrichment.")

    print("--- Enriched Transit Connection Analysis Finished ---")
    return place_to_building_connections_df, place_to_building_no_connection_df
