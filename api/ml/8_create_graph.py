def create_galway_transport_graph(
        galway_places_summary_df1_input: pd.DataFrame,
        galway_buildings_summary_df1_input: pd.DataFrame,
        bus_stops_gdf_input: pd.DataFrame,  # GeoDataFrame
        bus_timetables_input: pd.DataFrame,
        MAX_ACCESS_DISTANCE_METERS_input: int = 800  # Default value
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
    MAX_ACCESS_DISTANCE_METERS = MAX_ACCESS_DISTANCE_METERS_input  # Use the input parameter

    # --- Helper Function for Haversine Distance ---
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers
        try:
            # It is to ensure inputs are explicitly converted to float before radians
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        except (ValueError, TypeError):  # Handle cases where conversion to float might fail
            return float('inf')  # Return infinity if coordinates are invalid

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance_km = R * c
        return distance_km * 1000  # Convert to meters

    # --- 1. Initialize Graph ---
    G = nx.DiGraph()
    print("Graph initialized.")

    # --- Step 1a: Add Place Nodes ---
    print("\nAdding place nodes (general POIs)...")

    for index, row in galway_places_summary_df1.iterrows():
        place_node_id = row['place_name']  # place_name is an attribute in galway_places_summary_df1
        if pd.notna(place_node_id):
            G.add_node(place_node_id, type='place', name=row['place_name'],
                       latitude=row.get('latitude'), longitude=row.get('longitude'))
    print(f"Nodes after general places: {G.number_of_nodes()}")

    # --- Step 1b: Add Building Nodes---
    print("\nAdding building nodes...")
    for index, row in galway_buildings_summary_df1.iterrows():
        building_node_id = row[
            'building_name']  # building_name is unique and an attribute in galway_buildings_summary_df1
        if pd.notna(building_node_id):
            G.add_node(building_node_id, type='building', name=row['building_name'],
                       osm_id=row.get('osm_id'), building_type=row.get('building_type'),
                       latitude=row.get('latitude'), longitude=row.get('longitude'))
    print(f"Nodes after adding buildings: {G.number_of_nodes()}")

    # --- Step 1c: Add Unique Bus Stop Nodes ---
    print("\nAdding unique bus stop nodes...")
    added_stop_ids = set()  # To keep track of added stop_ids
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
    bus_stop_nodes_data_for_access = {node_id: data for node_id, data in G.nodes(data=True) if
                                      data.get('type') == 'bus_stop'}  # Renamed to avoid conflict

    for place_node_id, place_data in place_nodes_data.items():
        place_lat = place_data.get('latitude');
        place_lon = place_data.get('longitude')
        if place_lat is None or place_lon is None or pd.isna(place_lat) or pd.isna(place_lon): continue
        for stop_node_id, stop_data in bus_stop_nodes_data_for_access.items():
            stop_lat = stop_data.get('latitude');
            stop_lon = stop_data.get('longitude')
            if stop_lat is None or stop_lon is None or pd.isna(stop_lat) or pd.isna(stop_lon): continue
            walking_distance_m = haversine(place_lat, place_lon, stop_lat, stop_lon)
            if walking_distance_m <= MAX_ACCESS_DISTANCE_METERS:  # Using the input parameter
                edge_attrs = {'type': 'access_egress', 'mode': 'walk', 'distance_m': walking_distance_m}
                G.add_edge(place_node_id, stop_node_id, **edge_attrs)
                G.add_edge(stop_node_id, place_node_id, **edge_attrs)
                access_edge_count += 2

    for building_node_id, building_data in building_nodes_data.items():
        building_lat = building_data.get('latitude');
        building_lon = building_data.get('longitude')
        if building_lat is None or building_lon is None or pd.isna(building_lat) or pd.isna(building_lon): continue
        for stop_node_id, stop_data in bus_stop_nodes_data_for_access.items():
            stop_lat = stop_data.get('latitude');
            stop_lon = stop_data.get('longitude')
            if stop_lat is None or stop_lon is None or pd.isna(stop_lat) or pd.isna(stop_lon): continue
            walking_distance_m = haversine(building_lat, building_lon, stop_lat, stop_lon)
            if walking_distance_m <= MAX_ACCESS_DISTANCE_METERS:  # Using the input parameter
                edge_attrs = {'type': 'access_egress', 'mode': 'walk', 'distance_m': walking_distance_m}
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
        print(
            f"Warning: Missing required columns in bus_timetables for transit edges: {missing_cols}. Skipping transit edge creation.")
    else:
        for route_id_timetable, group in bus_timetables.groupby('route_id'):
            route_stops = group.sort_values(by='stop_order_on_route')
            for i in range(len(route_stops) - 1):
                from_stop_id_mapped = route_stops.iloc[i]['stop_id_mapped']
                to_stop_id_mapped = route_stops.iloc[i + 1]['stop_id_mapped']

                # check if mapped stop_ids actually exist as nodes in our graph
                from_node_exists = from_stop_id_mapped in valid_graph_stop_node_ids
                to_node_exists = to_stop_id_mapped in valid_graph_stop_node_ids

                if from_node_exists and to_node_exists:
                    from_stop_node_data = G.nodes[from_stop_id_mapped]
                    to_stop_node_data = G.nodes[to_stop_id_mapped]
                    from_lat = from_stop_node_data.get('latitude');
                    from_lon = from_stop_node_data.get('longitude')
                    to_lat = to_stop_node_data.get('latitude');
                    to_lon = to_stop_node_data.get('longitude')

                    if None not in [from_lat, from_lon, to_lat, to_lon] and \
                            all(pd.notna(coord) for coord in [from_lat, from_lon, to_lat, to_lon]):
                        segment_distance_m = haversine(from_lat, from_lon, to_lat, to_lon)
                        edge_attrs = {'type': 'transit', 'route_id': route_id_timetable,
                                      'hop_count': 1, 'distance_m': segment_distance_m}
                        G.add_edge(from_stop_id_mapped, to_stop_id_mapped, **edge_attrs)
                        transit_edge_count += 1
    print(f"Added {transit_edge_count} directed transit edges.")

    # --- Node Relabeling ---
    old_name = "Portershed a Dó"  # old name
    new_name = "Portershed"  # new name
    mapping = {old_name: new_name}

    if G.has_node(old_name):
        print(f"Node '{old_name}' exists in the graph before relabeling.")
        G = nx.relabel_nodes(G, mapping, copy=False)  # copy=False modifies in place
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
        if all_edges_with_data:  # Check if list is not empty before sampling
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
