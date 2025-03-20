import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from student_analysis2 import generate_student_portfolio
from utils.grade_calculator import grade_calculate

def render_portfolio_tab(results, all_course_names, section_separator, get_assessment_components):
    """
    Render the Student e-Portfolio tab with personalized student analytics
    
    Parameters:
    - results: Dictionary containing analysis results for all courses
    - all_course_names: List of course names
    - section_separator: Function to create visual separation between sections
    - get_assessment_components: Function to identify assessment components
    """
    st.header("Student e-Portfolio")
    st.write("View individual student performance compared to class averages")
    
    # Course selector for e-Portfolio tab
    selected_course = st.selectbox(
        "Select Course:", 
        options=all_course_names,
        index=all_course_names.index("All Courses") if "All Courses" in all_course_names else 0,
        key='portfolio_course_selector'
    )
    
    # Get results for the selected course
    course_results = results[selected_course]
    course_data = course_results['data']
    
    section_separator()
    
    # First, let's add a text input for searching
    search_term = st.text_input("Search for a student:")
    
    # Filter the student options based on the search term
    filtered_student_ids = course_results['student_ids']
    if search_term:
        filtered_student_ids = [
            student_id for student_id in course_results['student_ids'] 
            if search_term.lower() in str(student_id).lower()
        ]
    
    # Then use the filtered list in the selectbox
    student_id = st.selectbox(
        "Select a student:",
        options=filtered_student_ids,
        format_func=lambda x: f"{x}"
    )

    if student_id:
        # Your code to display student portfolio
        st.write(f"Showing portfolio for student: {student_id}")
        
    if student_id:
        # Generate portfolio for selected student
        portfolio = generate_student_portfolio(course_data, student_id)
        
        if portfolio:
            # Display overall performance
            st.subheader("Overall Performance")
            
            # Create metrics for key performance indicators
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Final Mark", 
                    f"{portfolio['student_data']['Final Mark']:.2f}"
                )
            
            with col2:
                if 'Final Mark' in portfolio['percentiles']:
                    st.metric(
                        "Percentile Rank", 
                        f"{portfolio['percentiles']['Final Mark']}%"
                    )
                else:
                    st.metric("Percentile Rank", "N/A")
            
            with col3:
                if 'Final Mark' in portfolio['comparisons']:
                    st.metric(
                        "Class Average", 
                        f"{portfolio['comparisons']['Final Mark']['class_average']:.2f}"
                    )
                else:
                    st.metric("Class Average", "N/A")
            
            with col4:
                if 'Cluster' in portfolio['student_data'] and portfolio['student_data']['Cluster'] is not None:
                    st.metric("Cluster", f"{int(portfolio['student_data']['Cluster'])}")
                else:
                    st.metric("Cluster", "N/A")
            
            with col5:
                if 'Final Mark' in portfolio['student_data']:
                    st.metric(
                        "Grade Status",
                        grade_calculate(portfolio['student_data']['Final Mark'])
                    )
                else:
                    st.metric("Grade Status", "N/A")
            
            section_separator()
            
            # Get dynamic assessment components for this student
            primary_cols = get_assessment_components(course_data)
            
            # Component performance visualization
            st.subheader("Component Performance")
            
            # Create two columns for visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart comparing student to class average - need at least 3 components
                if len(primary_cols) >= 3:
                    # Get student values and class averages
                    student_values = []
                    class_averages = []
                    radar_cols = []
                    
                    for col in primary_cols:
                        if col in portfolio['student_data'] and col in portfolio['comparisons']:
                            student_val = portfolio['student_data'][col]
                            class_avg = portfolio['comparisons'][col]['class_average']
                            
                            # Only include if we have valid values
                            if not pd.isna(student_val) and not pd.isna(class_avg):
                                student_values.append(student_val)
                                class_averages.append(class_avg)
                                radar_cols.append(col)
                    
                    # Only draw radar chart if we have at least 3 valid components
                    if len(radar_cols) >= 3:
                        radar_fig = go.Figure()
                        
                        # Add student data
                        radar_fig.add_trace(go.Scatterpolar(
                            r=student_values,
                            theta=radar_cols,
                            fill='toself',
                            name=f'Student {student_id}'
                        ))
                        
                        # Add class average
                        radar_fig.add_trace(go.Scatterpolar(
                            r=class_averages,
                            theta=radar_cols,
                            fill='toself',
                            name='Class Average'
                        ))
                        
                        radar_fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True
                                )
                            ),
                            title="Student vs. Class Average",
                            showlegend=True
                        )
                        
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.warning("Not enough valid assessment components for radar chart")
                else:
                    st.warning("At least 3 assessment components are needed for radar chart visualization")
            
            with col2:
                # Bar chart showing percentile ranks
                percentile_data = []
                for col in primary_cols:
                    if col in portfolio['percentiles']:
                        percentile_data.append({
                            'Component': col,
                            'Percentile': portfolio['percentiles'][col]
                        })
                
                if percentile_data:
                    percentile_df = pd.DataFrame(percentile_data)
                    
                    fig = px.bar(
                        percentile_df,
                        x='Component',
                        y='Percentile',
                        title="Percentile Ranks by Component",
                        color='Percentile',
                        color_continuous_scale='Viridis',
                        range_y=[0, 100]
                    )
                    
                    fig.add_hline(
                        y=50, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text="50th Percentile", 
                        annotation_position="bottom right"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No percentile data available for visualization")
            
            section_separator()
            
            # Detailed component analysis
            st.subheader("Detailed Component Analysis")
            
            # Create a table with component details
            component_details = []
            for col in primary_cols:
                if col in portfolio['student_data'] and col in portfolio['comparisons']:
                    component_details.append({
                        'Component': col,
                        'Student Score': f"{portfolio['student_data'][col]:.2f}",
                        'Class Average': f"{portfolio['comparisons'][col]['class_average']:.2f}",
                        'Difference': f"{portfolio['comparisons'][col]['difference']:.2f}",
                        'Percentile': f"{portfolio['percentiles'].get(col, 'N/A')}",
                        'Status': portfolio['comparisons'][col]['status'].title()
                    })
            
            if component_details:
                st.dataframe(pd.DataFrame(component_details), use_container_width=True)
                
                # Strengths and weaknesses
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Strengths")
                    if portfolio['strengths']:
                        for component, diff_pct in portfolio['strengths']:
                            st.write(f"- **{component}**: {diff_pct:.2f}% above average")
                    else:
                        st.write("No specific strengths identified")
                
                with col2:
                    st.subheader("Areas for Improvement")
                    if portfolio['weaknesses']:
                        for component, diff_pct in portfolio['weaknesses']:
                            st.write(f"- **{component}**: {abs(diff_pct):.2f}% below average")
                    else:
                        st.write("No specific weaknesses identified")
                
                # Recommendations
                st.subheader("Personalized Recommendations")
                if portfolio['recommendations']:
                    for recommendation in portfolio['recommendations']:
                        st.write(f"- {recommendation}")
                else:
                    st.write("No specific recommendations available")
            else:
                st.warning("No component data available for detailed analysis")
        else:
            st.error("Could not generate portfolio for the selected student.")
