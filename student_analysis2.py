import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Assessment component structure:
# Original structure:
# 1. total (50%) = OBA (25%) + EMI (25%)
# 2. total essay (50%) = (Essay 1 + Essay 2 + Essay 3)/30*50
# 3. theory (60%) = (total + total essay) * 0.6
# 4. Final Mark = theory + OSPE + PAM + PBL

# Load the dataset and calculate derived columns
def load_data(file_path=None, data_df=None, course_name=None):
    """
    Load and preprocess student data.
    
    Parameters:
    - file_path: Path to CSV file (if loading from file)
    - data_df: Pandas DataFrame (if data is already loaded)
    - course_name: Name of the course for the data
    
    Returns:
    - Processed DataFrame with derived columns
    """
    if data_df is not None:
        # Use the provided DataFrame
        data = data_df.copy()
    elif file_path:
        # Read CSV with proper column handling
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Either file_path or data_df must be provided")
    
    # Add course name if provided
    if course_name:
        data['Course'] = course_name
    
    # Clean column names (remove whitespace)
    data.columns = data.columns.str.strip()
    
    # Handle null columns - remove columns where all values are null
    null_columns = data.columns[data.isna().all()].tolist()
    if null_columns:
        print(f"Removing completely null columns: {null_columns}")
        data = data.drop(columns=null_columns)
    
    # Additionally, identify columns with majority null values (e.g., >90% null)
    mostly_null_threshold = 0.9
    mostly_null_columns = [col for col in data.columns if data[col].isna().mean() > mostly_null_threshold]
    if mostly_null_columns:
        print(f"Warning: The following columns have >{mostly_null_threshold*100}% null values: {mostly_null_columns}")
        # Optionally, you could drop these columns too:
        # data = data.drop(columns=mostly_null_columns)
    
    return data

# 1. Finding students who need early intervention
def identify_students_needing_intervention(data, percentile_threshold=25):
    """Identify students who may need early intervention based on their performance."""
    # Get threshold value from percentile
    threshold = np.percentile(data['Final Mark'], percentile_threshold)
    print(f"Intervention threshold (bottom {percentile_threshold}%): {threshold:.2f}")
    
    # Identify at-risk students
    at_risk_students = data[data['Final Mark'] <= threshold].copy()
    
    # Define primary components (non-derived) - dynamically identify assessment components
    # Exclude columns that are likely not assessment components
    exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster']
    primary_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Essay')]
    
    print(f"Identified assessment components: {primary_cols}")
    
    # Calculate average for each component
    component_avgs = {col: data[col].mean() for col in primary_cols if col in data.columns}
    
    # For each student, find their weakest areas
    student_weaknesses = []
    for _, student in at_risk_students.iterrows():
        # Calculate relative performance (how far below average) for each component
        weaknesses = []
        for col in primary_cols:
            if col in data.columns and not pd.isna(student[col]) and component_avgs.get(col, 0) > 0:
                relative_performance = (student[col] - component_avgs[col]) / component_avgs[col]
                weaknesses.append((col, relative_performance))
        
        # Sort by relative performance (ascending)
        weaknesses.sort(key=lambda x: x[1])
        
        # Store the student with their top 3 weakest areas (or fewer if less than 3 components)
        max_weak_areas = min(3, len(weaknesses))
        student_info = {
            'Student_ID': student['Student_ID'],
            'Final_Mark': student['Final Mark'],
            'Weakest_Areas': [w[0] for w in weaknesses[:max_weak_areas]],
            'Relative_Performance': [f"{w[1]:.2%}" for w in weaknesses[:max_weak_areas]]
        }
        
        # Add course name if available
        if 'Course' in student:
            student_info['Course'] = student['Course']
            
        student_weaknesses.append(student_info)
    
    # Add intervention recommendations based on weak areas
    for student in student_weaknesses:
        interventions = []
        for area in student['Weakest_Areas']:
            if 'MCQ' in area:
                interventions.append(f"Additional objective assessment practice in {area}")
            elif 'MEQ' in area:
                interventions.append(f"Focus on concept understanding on {area}")
            elif 'OSPE' in area:
                interventions.append("Revise skills lab sessions")
            elif 'PAM' in area:
                interventions.append("Revisit End of Module Assessment")
            elif 'PBL' in area:
                interventions.append("Problem-based learning study group")
            else:
                interventions.append(f"Additional support for {area}")
        student['Recommended_Interventions'] = interventions
    
    return student_weaknesses, threshold

# 2. Discovering natural groupings of student performance patterns
def discover_student_groupings(data, n_clusters=4, threshold=None):
    """Use clustering to discover natural groupings of student performance patterns."""
    # Dynamically identify primary components - exclude non-assessment columns
    exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster']
    primary_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Essay')]
    
    print(f"Using assessment components for clustering: {primary_cols}")
    
    # Filter to only include columns that exist in the data
    primary_cols = [col for col in primary_cols if col in data.columns]
    
    # Ensure we have at least 2 columns for clustering
    if len(primary_cols) < 2:
        print("Warning: Not enough assessment components for meaningful clustering. Using all available numeric columns.")
        # Use all numeric columns except Student_ID, Final Mark, and Cluster
        primary_cols = [col for col in data.select_dtypes(include=['number']).columns 
                        if col not in ['Student_ID', 'Final Mark', 'Cluster']]
    
    # Prepare data for clustering
    X = data[primary_cols]
    
    # Handle any remaining missing values by filling with means
    for col in X.columns:
        if X[col].isna().any():
            col_mean = X[col].mean()
            X[col] = X[col].fillna(col_mean)
            print(f"Filled {X[col].isna().sum()} missing values in {col} with mean ({col_mean:.2f})")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters using the Elbow method
    inertia = []
    K_range = range(1, min(10, len(data) // 5 + 1))  # Ensure K is reasonable for dataset size
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Determine optimal K from elbow plot
    if len(K_range) > 2:
        inertia_diff = np.diff(inertia)
        if len(inertia_diff) > 1:
            inertia_diff2 = np.diff(inertia_diff)
            optimal_k = K_range[np.argmax(inertia_diff2) + 1]
            print(f"Optimal number of clusters detected: {optimal_k}")
        else:
            optimal_k = 2
            print("Not enough data points to calculate second derivative. Using K=2.")
    else:
        optimal_k = 2
        print("Not enough K values to calculate optimal K. Using K=2.")
    
    # Use the specified number of clusters if provided, otherwise use the detected optimal
    n_clusters = n_clusters if n_clusters else optimal_k
    
    # Ensure n_clusters is valid (not more than the number of samples)
    n_clusters = min(n_clusters, len(data) - 1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze each cluster - use numeric_only to prevent errors with Student_ID
    cluster_profiles = data.groupby('Cluster').mean(numeric_only=True)
    
    # Get size of each cluster
    cluster_sizes = data['Cluster'].value_counts().sort_index()
    
    # If no threshold is provided, calculate one based on the 25th percentile
    if threshold is None:
        threshold = np.percentile(data['Final Mark'], 25)
    
    # Calculate performance distribution within each cluster
    cluster_distributions = {}
    for cluster in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        cluster_distributions[cluster] = {
            'size': len(cluster_data),
            'min_mark': cluster_data['Final Mark'].min(),
            'max_mark': cluster_data['Final Mark'].max(),
            'mean_mark': cluster_data['Final Mark'].mean(),
            'percentile_25': cluster_data['Final Mark'].quantile(0.25),
            'median_mark': cluster_data['Final Mark'].median(),
            'percentile_75': cluster_data['Final Mark'].quantile(0.75),
            'at_risk_percentage': 
                (cluster_data['Final Mark'] <= threshold).mean() * 100
        }
    
    # Identify key characteristics of each cluster
    cluster_characteristics = {}
    for cluster in range(n_clusters):
        # Compare this cluster's average to the overall average
        comparison = {}
        for col in primary_cols:
            overall_avg = data[col].mean()
            cluster_avg = cluster_profiles.loc[cluster, col]
            difference = (cluster_avg - overall_avg) / overall_avg
            comparison[col] = {
                'cluster_avg': cluster_avg,
                'overall_avg': overall_avg,
                'difference': difference
            }
        
        # Sort by absolute difference to find most distinctive characteristics
        sorted_comparison = sorted(
            comparison.items(),
            key=lambda x: abs(x[1]['difference']),
            reverse=True
        )
        
        # Store top 3 distinctive features (or fewer if less than 3 components)
        max_features = min(3, len(sorted_comparison))
        distinctive_features = []
        for col, values in sorted_comparison[:max_features]:
            if values['difference'] > 0:
                status = "strong"
            else:
                status = "weak"
                
            distinctive_features.append({
                'component': col,
                'status': status,
                'difference': f"{values['difference']:.2%}"
            })
            
        cluster_characteristics[cluster] = distinctive_features
    
    # For visualization (if needed), reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = data['Cluster']
    pca_df['Final Mark'] = data['Final Mark']
    
    # Add course information if available
    if 'Course' in data.columns:
        pca_df['Course'] = data['Course'].values
    
    return {
        'cluster_profiles': cluster_profiles,
        'cluster_sizes': cluster_sizes,
        'cluster_distributions': cluster_distributions,
        'cluster_characteristics': cluster_characteristics,
        'pca_data': pca_df
    }

# 3. Streamlining assessment structure by finding the most important components
def identify_important_components(data):
    """Identify the most important assessment components using multiple methods."""
    # Dynamically identify primary components
    exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster']
    primary_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Essay')]
    
    print(f"Analyzing importance of components: {primary_cols}")
    
    # Ensure all selected columns exist in data
    primary_cols = [col for col in primary_cols if col in data.columns]
    
    # Handle any missing values in the data
    X = data[primary_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())
    
    y = data['Final Mark']
    
    # Method 1: Feature importance using Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Method 2: Feature importance using Gradient Boosting
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X, y)
    gb_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Method 3: Correlation with Final Mark
    correlations = data[primary_cols + ['Final Mark']].corr()['Final Mark'].drop('Final Mark')
    corr_importance = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    }).sort_values('Correlation', ascending=False)
    
    # Method 4: Principal Component Analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    
    # Get loadings of first principal component
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(min(len(X.columns), len(pca.components_)))],
        index=X.columns
    )
    
    # Explained variance by each component
    explained_variance = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained_Variance': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    
    # Find redundant components using correlation
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs with correlation above 0.7 (only if we have more than one component)
    redundant_pairs = []
    if len(primary_cols) > 1:
        for col in upper_tri.columns:
            high_corr = upper_tri[col][upper_tri[col] > 0.7].index.tolist()
            for idx in high_corr:
                redundant_pairs.append({
                    'Component1': col,
                    'Component2': idx,
                    'Correlation': corr_matrix.loc[col, idx]
                })
    
    # Sort redundant pairs by correlation strength
    redundant_pairs = sorted(redundant_pairs, key=lambda x: x['Correlation'], reverse=True)
    
    # Determine number of components to retain
    # Based on explained variance (e.g., 90% threshold)
    n_components_90 = 1  # Default to 1
    if not explained_variance.empty:
        n_components_90 = np.argmax(explained_variance['Cumulative_Variance'] >= 0.9) + 1
        # Ensure it's not more than the available components
        n_components_90 = min(n_components_90, len(primary_cols))
    
    # Create a simplified assessment recommendation
    # Combine results from different methods
    important_components = set()
    
    # Top components from Random Forest (if available)
    if not rf_importance.empty:
        for feature in rf_importance['Feature'][:min(3, len(rf_importance))]:
            important_components.add(feature)
    
    # Top components from correlation (if available)
    if not corr_importance.empty:
        for feature in corr_importance['Feature'][:min(3, len(corr_importance))]:
            important_components.add(feature)
    
    # Create a set of potentially redundant components
    redundant_components = set()
    for pair in redundant_pairs:
        # If one component is important and the other is correlated
        # then the other is redundant
        if pair['Component1'] in important_components:
            redundant_components.add(pair['Component2'])
        elif pair['Component2'] in important_components:
            redundant_components.add(pair['Component1'])
    
    # Remove redundant components from important components list
    important_components = important_components - redundant_components
    
    # Generate recommendations for streamlining
    streamlined_structure = {
        'retained_components': list(important_components),
        'potentially_redundant': list(redundant_components),
        'pca_recommended_components': n_components_90
    }
    
    return {
        'rf_importance': rf_importance,
        'gb_importance': gb_importance,
        'correlations': corr_importance,
        'pca_loadings': loadings,
        'explained_variance': explained_variance,
        'redundant_pairs': redundant_pairs,
        'streamlined_structure': streamlined_structure
    }

# 4. Generate student e-portfolio
def generate_student_portfolio(data, student_id):
    """Generate a comprehensive e-portfolio for an individual student."""
    # Find the student
    if student_id not in data['Student_ID'].values:
        return None
    
    student_data = data[data['Student_ID'] == student_id].iloc[0]
    
    # Dynamically define primary components
    exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster']
    primary_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Essay')]
    derived_cols = ['Final Mark']
    
    print(f"Generating portfolio with components: {primary_cols}")
    
    # Calculate percentile ranks for each component
    percentiles = {}
    for col in primary_cols + derived_cols:
        if col in data.columns and not pd.isna(student_data[col]):
            percentiles[col] = round(100 * (data[col] <= student_data[col]).mean())
    
    # Calculate comparison to class average
    comparisons = {}
    for col in primary_cols + derived_cols:
        if col in data.columns and not pd.isna(student_data[col]):
            class_avg = data[col].mean()
            student_val = student_data[col]
            diff_pct = (student_val - class_avg) / class_avg * 100 if class_avg != 0 else 0
            comparisons[col] = {
                'student_value': student_val,
                'class_average': class_avg,
                'difference': student_val - class_avg,
                'difference_percent': diff_pct,
                'status': 'above average' if diff_pct > 0 else 'below average'
            }
    
    # Identify strengths and weaknesses (only for components that exist)
    valid_comparisons = [(col, comparisons[col]['difference_percent']) 
                         for col in primary_cols 
                         if col in comparisons]
    
    sorted_components = sorted(valid_comparisons, key=lambda x: x[1], reverse=True)
    
    # Get top and bottom strengths and weaknesses (or fewer if limited components)
    max_items = min(3, len(sorted_components))
    strengths = sorted_components[:max_items]
    weaknesses = sorted_components[-max_items:] if len(sorted_components) >= max_items else []
    
    # Determine which cluster the student belongs to
    cluster = student_data.get('Cluster', None)
    
    # Add course information if available
    course = student_data.get('Course', None)
    
    # Generate recommendations based on weaknesses
    recommendations = []
    for component, _ in weaknesses:
        if 'MCQ' in component:
            recommendations.append(f"Focus on improving objective assessment skills in {component}")
        elif 'MEQ' in component:
            recommendations.append(f"Work on writing and elaborating skills for {component}")
        elif 'OSPE' in component:
            recommendations.append("Revisit lab sessions or discuss more on missed details")
        elif 'PAM' in component:
            recommendations.append("Revisit End of Module Assessment")
        elif 'PBL' in component:
            recommendations.append("Enhance problem-solving approach through group study")
        else:
            # Generic recommendation for unknown components
            recommendations.append(f"Dedicate additional study time to improve performance in {component}")
    
    # Compile portfolio
    portfolio = {
        'student_id': student_id,
        'student_data': student_data,
        'percentiles': percentiles,
        'comparisons': comparisons,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommendations': recommendations,
        'cluster': cluster,
        'course': course
    }
    
    return portfolio

# Main function to run all analyses
def analyze_student_performance(file_path=None, data_df=None, course_name=None):
    """
    Run complete analysis on student performance data.
    
    Parameters:
    - file_path: Path to CSV file (if loading from file)
    - data_df: Pandas DataFrame (if data is already loaded)
    - course_name: Name of the course for this data
    
    Returns:
    - Dictionary containing all analysis results
    """
    # Load data
    data = load_data(file_path=file_path, data_df=data_df, course_name=course_name)
    print(f"Loaded dataset with {len(data)} students and {len(data.columns)} columns")
    
    # 1. Identify students needing intervention
    intervention_results, threshold = identify_students_needing_intervention(data)
    print(f"Identified {len(intervention_results)} students needing intervention")
    
    # 2. Discover natural groupings
    cluster_results = discover_student_groupings(data, threshold=threshold)
    print(f"Identified {len(cluster_results['cluster_sizes'])} natural student groupings")
    
    # 3. Identify important components for streamlining
    component_results = identify_important_components(data)
    print("Completed component importance analysis")
    
    # Analyze the assessment structure and its impact
    assessment_structure = {
        'component_weights': {
            'MCQ': 'Approximately 20% of final mark',
            'MEQ': 'Approximately 20% of final mark',
            'OSPE': 'Approximately 20% of final mark',
            'PAM': 'Approximately 20% of final mark',
            'PBL': 'Approximately 20% of final mark'
        },
        'pathway_analysis': {
            'all_components': {
                'description': 'All components contribute directly to final mark',
                'pathway': 'All components → Final Mark',
                'formula': 'Final Mark = MCQ + MEQ + OSPE + PAM + PBL'
            }
        }
    }
    
    # Return all results
    return {
        'data': data,
        'intervention_results': intervention_results,
        'cluster_results': cluster_results,
        'component_results': component_results,
        'assessment_structure': assessment_structure,
        'student_ids': data['Student_ID'].tolist(),  # Add list of student IDs for the dropdown
        'course_name': course_name if course_name else "Unnamed Course"
    }

# Multi-course analysis
def analyze_multiple_courses(course_files):
    """
    Analyze multiple courses from different CSV files.
    
    Parameters:
    - course_files: Dictionary mapping course names to file paths
    
    Returns:
    - Dictionary containing results for each course and combined analysis
    """
    all_courses_data = pd.DataFrame()
    course_results = {}
    
    # Analyze each course separately
    for course_name, file_path in course_files.items():
        print(f"Analyzing course: {course_name}")
        # Run analysis for this course
        results = analyze_student_performance(file_path=file_path, course_name=course_name)
        # Store the results
        course_results[course_name] = results
        # Append the data to all courses
        all_courses_data = pd.concat([all_courses_data, results['data']], ignore_index=True)
    
    # Generate combined analysis if we have multiple courses
    if len(course_files) > 1:
        print("Generating combined analysis for all courses")
        combined_results = analyze_student_performance(data_df=all_courses_data, course_name="All Courses")
        course_results["All Courses"] = combined_results
    
    return course_results

# If this script is run directly
if __name__ == "__main__":
    # Example of running analysis on a single file
    results = analyze_student_performance("test2.csv")
    
    # Example output
    print("\n===== EARLY INTERVENTION STUDENTS =====")
    for i, student in enumerate(results['intervention_results'][:5]):
        print(f"\nStudent {i+1}: {student['Student_ID']} (Final Mark: {student['Final_Mark']:.2f})")
        print("Weakest areas:")
        for area, perf in zip(student['Weakest_Areas'], student['Relative_Performance']):
            print(f"  - {area}: {perf} below average")
        print("Recommended interventions:")
        for intervention in student['Recommended_Interventions']:
            print(f"  - {intervention}")
    
    print("\n===== STUDENT GROUPINGS =====")
    for cluster in range(len(results['cluster_results']['cluster_sizes'])):
        dist = results['cluster_results']['cluster_distributions'][cluster]
        print(f"\nCluster {cluster}: {dist['size']} students")
        print(f"Performance: range {dist['min_mark']:.2f}-{dist['max_mark']:.2f}, mean {dist['mean_mark']:.2f}")
        print(f"At-risk percentage: {dist['at_risk_percentage']:.2f}%")
        print("Distinctive characteristics:")
        for feature in results['cluster_results']['cluster_characteristics'][cluster]:
            print(f"  - {feature['component']}: {feature['status']} ({feature['difference']} from average)")
