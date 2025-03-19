import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from student_analysis2 import discover_student_groupings

def render_clusters_tab(results, all_course_names, section_separator, get_assessment_components):
    """
    Render the Student Clusters tab with clustering visualizations
    
    Parameters:
    - results: Dictionary containing analysis results for all courses
    - all_course_names: List of course names
    - section_separator: Function to create visual separation between sections
    - get_assessment_components: Function to identify assessment components
    """
    st.header("Student Performance Clusters")
    
    # Course selector for clusters tab
    selected_course = st.selectbox(
        "Select Course:", 
        options=all_course_names,
        index=all_course_names.index("All Courses") if "All Courses" in all_course_names else 0,
        key='cluster_course_selector'
    )
    
    # Get results for the selected course
    course_results = results[selected_course]
    course_data = course_results['data']
    
    section_separator()
    
    # Get dynamic assessment components
    primary_cols = get_assessment_components(course_data)
    
    # Determine max clusters based on data size and component count
    max_clusters = min(6, len(course_data) // 5) if len(course_data) > 10 else 2
    max_clusters = max(2, max_clusters)  # Ensure at least 2
    
    # Cluster Selection
    n_clusters = st.slider(
        "Number of Clusters:",
        min_value=2,
        max_value=max_clusters,
        value=min(4, max_clusters),
        step=1,
        key="cluster_slider"
    )
    
    # Get threshold from the original analysis
    threshold = np.percentile(course_data['Final Mark'], 25)
    
    # Run clustering with updated cluster count
    if len(primary_cols) >= 2:  # Need at least 2 components for clustering
        new_clusters = discover_student_groupings(course_data, n_clusters=n_clusters, threshold=threshold)
        
        # Cluster visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # PCA plot of clusters
            fig = px.scatter(
                new_clusters['pca_data'],
                x="PC1", y="PC2",
                color="Cluster",
                size="Final Mark",
                title=f"Student Clusters (n={n_clusters}) - {selected_course}",
                hover_data=["Cluster", "Final Mark"],
                color_continuous_scale=px.colors.qualitative.G10
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Only show radar chart if we have 3+ components
            if len(primary_cols) >= 3:
                # Create radar chart for cluster profiles
                # Normalize the cluster profiles for radar chart
                profiles_normalized = new_clusters['cluster_profiles'][primary_cols].copy()
                
                # Handle normalization properly to avoid division by zero
                for col in primary_cols:
                    range_val = course_data[col].max() - course_data[col].min()
                    if range_val > 0:
                        profiles_normalized[col] = (profiles_normalized[col] - course_data[col].min()) / range_val
                    else:
                        profiles_normalized[col] = 0.5  # Default to middle if no range
                
                # Create radar chart
                radar_fig = go.Figure()
                
                for cluster in range(n_clusters):
                    if cluster in profiles_normalized.index:
                        radar_fig.add_trace(go.Scatterpolar(
                            r=profiles_normalized.loc[cluster].values,
                            theta=primary_cols,
                            fill='toself',
                            name=f'Cluster {cluster}'
                        ))
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title=f"Cluster Profiles - {selected_course}",
                    showlegend=True
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.warning("At least 3 assessment components are needed for radar chart visualization")
        
        section_separator()
        
        # Cluster profiles
        st.subheader("Cluster Profiles")
        
        # Create rows of columns for cluster profiles
        # Use at most 3 columns per row
        clusters_per_row = min(3, n_clusters)
        
        # Calculate number of rows needed
        num_rows = (n_clusters + clusters_per_row - 1) // clusters_per_row
        
        # Display clusters in rows of 3
        for row in range(num_rows):
            # Create columns for this row
            cols = st.columns(clusters_per_row)
            
            # Fill columns with clusters
            for i in range(clusters_per_row):
                cluster_idx = row * clusters_per_row + i
                
                # Break if we've displayed all clusters
                if cluster_idx >= n_clusters:
                    break
                
                # Display cluster in this column
                with cols[i]:
                    if cluster_idx in new_clusters['cluster_distributions']:
                        dist = new_clusters['cluster_distributions'][cluster_idx]
                        
                        st.write(f"**Cluster {cluster_idx} ({dist['size']} students)**")
                        st.write(f"Performance: range {dist['min_mark']:.2f}-{dist['max_mark']:.2f}")
                        st.write(f"Mean: {dist['mean_mark']:.2f}, Median: {dist['median_mark']:.2f}")
                        st.write(f"At-risk percentage: {dist['at_risk_percentage']:.2f}%")
                        
                        # Show distinctive characteristics if available
                        if cluster_idx in new_clusters['cluster_characteristics']:
                            characteristics = new_clusters['cluster_characteristics'][cluster_idx]
                            if characteristics:
                                st.write("Distinctive characteristics:")
                                for feature in characteristics:
                                    st.write(f"- {feature['component']}: {feature['status']} ({feature['difference']} from average)")
                        
                        st.write("---")
    else:
        st.warning("At least 2 assessment components are needed for cluster analysis")
