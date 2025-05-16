import pandas as pd
import numpy as np


# called from step 10
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
    
    # required_poi_cols = ['origin_poi', 'destination_poi']
    # if all(col in place_to_building_connections_df.columns for col in required_poi_cols) and \
    #    not place_to_building_connections_df.empty:
    #     try:
    #         origin = place_to_building_connections_df['origin_poi'].iloc[0]
    #         destination = place_to_building_connections_df['destination_poi'].iloc[0]
    #     except IndexError:
    #         print("Warning: Could not extract origin/destination POI names from DataFrame")
    # else:
    #     print(f"Warning: Missing '{required_poi_cols[0]}' or '{required_poi_cols[1]}' columns, or DataFrame is empty.")

    return accessibility_score


# if __name__ == '__main__':


#     print("--- Calling calculate_and_display_accessibility_score ---")
#     if 'place_to_building_connections_df' in locals() and isinstance(place_to_building_connections_df, pd.DataFrame):
#         calculate_and_display_accessibility_score(place_to_building_connections_df)
 
#     else:
#         print("Dummy 'place_to_building_connections_df' not defined for example usage.")
