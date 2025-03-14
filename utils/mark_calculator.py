def calculate_weighted_final_marks(data, weights=None, max_scores=None):
    """
    Calculate the final mark based on custom component weights and user-provided maximum scores
    
    Parameters:
    - data: DataFrame containing student data
    - weights: Dictionary mapping component names to their percentage weights
    - max_scores: Dictionary mapping component names to their maximum possible scores
    
    Returns:
    - DataFrame with added 'Calculated_Final_Mark' column
    """
    # Make a copy of the input data
    result_data = data.copy()
    
    # Get assessment components
    exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster', 'Calculated_Final_Mark']
    assessment_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Essay')]
    
    # If weights are not provided, use equal weighting
    if weights is None:
        weight_per_component = 100 / len(assessment_cols)
        weights = {col: weight_per_component for col in assessment_cols}
    
    # Ensure max_scores is a dictionary even if None was passed
    if max_scores is None:
        max_scores = {}
    
    # Print debug information about weights and max scores being used
    print("Using the following weights and max scores:")
    for col in assessment_cols:
        if col in weights:
            print(f"  {col}: Weight = {weights.get(col, 0)}%, Max Score = {max_scores.get(col, 'Not specified')}")
    
    # Initialize the weighted sum column
    result_data['Calculated_Final_Mark'] = 0
    
    # Add the weighted contribution of each component
    for col in assessment_cols:
        if col in weights and col in data.columns:
            # Get the component's weight (as a proportion of 1)
            weight = weights[col] / 100
            
            # Get the component scores, handling missing values
            component_scores = data[col].fillna(data[col].mean())
            
            # Normalize by user-provided max score
            if col in max_scores and max_scores[col] > 0:
                # Convert raw score to 0-100 scale using user-provided max score
                normalized_scores = (component_scores / max_scores[col])
                print(f"Normalized {col} using max score {max_scores[col]}")
            else:
                # If no max score provided for this component, assume scores are already on 0-100 scale
                normalized_scores = component_scores
                print(f"No max score provided for {col}, using raw values")
            
            # Apply weight and add to final mark
            weighted_contribution = normalized_scores * weight
            result_data['Calculated_Final_Mark'] += weighted_contribution
            
            # Print contribution for first few students to aid debugging
            print(f"{col} contribution to final mark (first 3 students):")
            for i in range(min(3, len(data))):
                print(f"  Student {data.iloc[i]['Student_ID']}: Raw={component_scores.iloc[i]:.2f}, "
                      f"Normalized={normalized_scores.iloc[i]:.2f}, "
                      f"Weighted={weighted_contribution.iloc[i]:.2f}")
    
    # Print final calculated marks for first few students
    print("\nCalculated Final Marks (first 3 students):")
    for i in range(min(3, len(data))):
        print(f"  Student {data.iloc[i]['Student_ID']}: {result_data.iloc[i]['Calculated_Final_Mark']:.2f}")
    
    return result_data