import streamlit as st
import plotly.express as px
import pandas as pd

def render_components_tab(results, all_course_names, section_separator):
    """
    Render the Component Analysis tab with feature importance and redundancy analysis
    
    Parameters:
    - results: Dictionary containing analysis results for all courses
    - all_course_names: List of course names
    - section_separator: Function to create visual separation between sections
    """
    st.header("Assessment Component Analysis")
    
    # Course selector for components tab
    selected_course = st.selectbox(
        "Select Course:", 
        options=all_course_names,
        index=all_course_names.index("All Courses") if "All Courses" in all_course_names else 0,
        key='component_course_selector'
    )
    
    # Get results for the selected course
    course_results = results[selected_course]
    component_results = course_results['component_results']
    
    # Dynamically get assessment components
    exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster']
    primary_cols = [col for col in course_results['data'].columns 
                  if col not in exclude_cols and not col.startswith('Essay')]
    
    if len(primary_cols) >= 2:  # Need at least 2 components for meaningful analysis
        section_separator()
        
        # Feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            # Random Forest importance
            if not component_results['rf_importance'].empty:
                fig = px.bar(
                    component_results['rf_importance'],
                    x='Feature', y='Importance',
                    title=f"Component Importance (Random Forest) - {selected_course}",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for Random Forest importance visualization")
        
        with col2:
            # Correlation importance
            if not component_results['correlations'].empty:
                fig = px.bar(
                    component_results['correlations'],
                    x='Feature', y='Correlation',
                    title=f"Component Correlation with Final Mark - {selected_course}",
                    color='Correlation',
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for correlation visualization")
        
        section_separator()
        
        # Redundant components
        st.subheader("Potentially Redundant Components")
        
        # Create dataframe for redundant pairs
        if component_results['redundant_pairs']:
            redundant_df = pd.DataFrame([
                {
                    'Component1': pair['Component1'],
                    'Component2': pair['Component2'],
                    'Correlation': f"{pair['Correlation']:.4f}"
                } for pair in component_results['redundant_pairs'][:5]
            ])
            
            if not redundant_df.empty:
                st.dataframe(redundant_df, use_container_width=True)
            else:
                st.write("No redundant components identified.")
        else:
            st.write("No redundant components identified.")
        
        section_separator()
        
        # PCA Analysis
        st.subheader("Principal Component Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Explained variance
            if not component_results['explained_variance'].empty:
                fig = px.bar(
                    component_results['explained_variance'],
                    x='Component', y='Explained_Variance',
                    title=f"Explained Variance by Principal Component - {selected_course}",
                    text='Explained_Variance'
                ).update_traces(texttemplate='%{text:.1%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for explained variance visualization")
        
        with col2:
            # PCA loadings heatmap - only show first 2 PCs if available
            if not component_results['pca_loadings'].empty and component_results['pca_loadings'].shape[1] >= 2:
                fig = px.imshow(
                    component_results['pca_loadings'].iloc[:, :2],
                    title=f"PCA Component Loadings (First 2 PCs) - {selected_course}",
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="Principal Component", y="Assessment Component", color="Loading")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data for PCA loadings visualization")
        
        section_separator()
        
        # Recommended assessment structure
        st.subheader("Recommended Assessment Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Key Components to Retain:**")
            if component_results['streamlined_structure']['retained_components']:
                st.write(", ".join(component_results['streamlined_structure']['retained_components']))
            else:
                st.write("No specific components identified to retain")
        
        with col2:
            st.write("**Consider Removing or Combining:**")
            if component_results['streamlined_structure']['potentially_redundant']:
                st.write(", ".join(component_results['streamlined_structure']['potentially_redundant']))
            else:
                st.write("No redundant components identified.")
    else:
        st.warning("At least 2 assessment components are needed for component analysis")