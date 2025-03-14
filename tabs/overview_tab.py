import streamlit as st
import plotly.express as px
import pandas as pd

def render_overview_tab(results, all_course_names, section_separator, get_assessment_components):
    """
    Render the Overview tab with summary statistics and visualizations
    
    Parameters:
    - results: Dictionary containing analysis results for all courses
    - all_course_names: List of course names
    - section_separator: Function to create visual separation between sections
    - get_assessment_components: Function to identify assessment components
    """
    st.header("Student Performance Overview")
    
    # Course selector for overview tab
    selected_course = st.selectbox(
        "Select Course:", 
        options=all_course_names,
        index=all_course_names.index("All Courses") if "All Courses" in all_course_names else 0,
        key='overview_course_selector'
    )
    
    # Get results for the selected course
    course_results = results[selected_course]
    course_data = course_results['data']
    
    section_separator()
    
    # Summary statistics and histogram in two columns
    col1, col2 = st.columns([3, 7])
    
    with col1:
        st.subheader("Key Statistics")
        st.write(f"Total Students: {len(course_data)}")
        st.write(f"Average Final Mark: {course_data['Final Mark'].mean():.2f}")
        st.write(f"Highest Mark: {course_data['Final Mark'].max():.2f}")
        st.write(f"Lowest Mark: {course_data['Final Mark'].min():.2f}")
        st.write(f"Students Needing Intervention: {len(course_results['intervention_results'])}")
    
    with col2:
        fig = px.histogram(
            course_data, x="Final Mark",
            title=f"Distribution of Final Marks - {selected_course}",
            color_discrete_sequence=['lightblue'],
            histnorm='percent',
            marginal="box"
        ).update_layout(xaxis_title="Final Mark", yaxis_title="Percentage of Students")
        st.plotly_chart(fig, use_container_width=True)
    
    section_separator()
    
    # Component contribution
    st.subheader("Assessment Component Structure")
    
    # Get dynamic assessment components
    primary_cols = get_assessment_components(course_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sunburst chart showing component contributions - dynamically created
        if primary_cols:
            # Define custom weights for each component (must sum to 100)
            component_weights = {
                'MCQ': 30,    # Adjusted from 20 to 30
                'MEQ': 25,    # Adjusted from 20 to 25
                'OSPE': 15,   # Adjusted from 20 to 15
                'PAM': 20,    # Remains at 20
                'PBL': 10     # Adjusted from 20 to 10
            }

            # Create data for the sunburst chart - with custom weights
            sunburst_data = [{'id': 'Final Mark', 'parent': '', 'value': 100}]

            # Add each component with custom weight
            for col in primary_cols:
                if col in component_weights:
                    weight = component_weights[col]
                else:
                    # For any components not explicitly defined, use a default weight
                    # You could divide the remaining weight among undefined components
                    weight = 5  # Default weight for any unspecified component
                
                sunburst_data.append({
                    'id': col, 
                    'parent': 'Final Mark', 
                    'value': weight
                })
                
            # Create the sunburst chart
            fig = px.sunburst(
                pd.DataFrame(sunburst_data),
                ids='id',
                parents='parent',
                values='value',
                title="Assessment Component Contribution",
                branchvalues="total"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No assessment components detected for visualization")
    
    with col2:
        # Component Correlation Heatmap - dynamically use available components
        if primary_cols:
            # Include Final Mark in correlation matrix
            corr_cols = primary_cols + ['Final Mark']
            
            # Create correlation matrix
            fig = px.imshow(
                course_data[corr_cols].corr(),
                title=f"Component Correlation Matrix - {selected_course}",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No assessment components detected for correlation matrix")
    
    # Course comparison if we have multiple courses
    if len([k for k in results.keys() if k != "All Courses"]) > 1:
        section_separator()
        st.subheader("Course Comparison")
        
        # Create comparison data for key metrics
        comparison_data = []
        for course_name, course_results in results.items():
            if course_name != "All Courses":  # Exclude the combined analysis
                course_data = course_results['data']
                comparison_data.append({
                    'Course': course_name,
                    'Average Mark': course_data['Final Mark'].mean(),
                    'Pass Rate (%)': (course_data['Final Mark'] >= 60).mean() * 100,
                    'Student Count': len(course_data)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Show comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                comparison_df,
                x='Course',
                y='Average Mark',
                title="Average Final Mark by Course",
                color='Course'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                comparison_df,
                x='Course',
                y='Pass Rate (%)',
                title="Pass Rate by Course (≥60)",
                color='Course'
            )
            st.plotly_chart(fig, use_container_width=True)