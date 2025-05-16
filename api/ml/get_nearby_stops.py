def get_nearby_stops(graph, poi_node_id, max_distance):
    """
    Finds bus stops connected to a POI node(place or building) via 'access_egress' edges
    within a specified maximum distance, and returns their IDs and distances.
    """
    nearby_stops_info = []  # list to store dictionaries

    if not graph.has_node(poi_node_id):
        print(f"Warning: POI node '{poi_node_id}' not found in the graph.")
        return nearby_stops_info  # Return empty list

    for u, v, data in graph.out_edges(poi_node_id, data=True):
        edge_type = data.get('type')
        edge_distance = data.get('distance_m', float('inf'))  # infinity if no distance

        # Check if the edge is an access/egress edge
        if edge_type == 'access_egress':
            # Check if the connected node 'v' is a bus stop
            if graph.has_node(v) and graph.nodes[v].get('type') == 'bus_stop':
                # Check if the distance is within the threshold
                if edge_distance <= max_distance:
                    # Add a dictionary with stop_id and distance
                    nearby_stops_info.append({'stop_id': v, 'distance_m': edge_distance})

    return poi_node_id, nearby_stops_info
