import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def render_intervention_tab(results, all_course_names, section_separator, get_assessment_components):
    """
    Render the Early Intervention tab with at-risk students and related visualizations
    
    Parameters:
    - results: Dictionary containing analysis results for all courses
    - all_course_names: List of course names
    - section_separator: Function to create visual separation between sections
    - get_assessment_components: Function to identify assessment components
    """
    st.header("Students Needing Intervention")
    
    # Course selector for intervention tab
    selected_course = st.selectbox(
        "Select Course:", 
        options=all_course_names,
        index=all_course_names.index("All Courses") if "All Courses" in all_course_names else 0,
        key='intervention_course_selector'
    )
    
    # Get results for the selected course
    course_results = results[selected_course]
    course_data = course_results['data']
    
    section_separator()
    
    # Intervention threshold slider
    percentile = st.slider(
        "Intervention Threshold (Final Mark Percentile):",
        min_value=5,
        max_value=50,
        value=25,
        step=5,
        key="intervention_slider"
    )
    
    threshold = np.percentile(course_data['Final Mark'], percentile)
    st.write(f"Intervention threshold: {threshold:.2f} (bottom {percentile}%)")
    
    # Recalculate at-risk students based on threshold
    at_risk_data = course_data[course_data['Final Mark'] <= threshold].copy()
    
    # Get dynamic assessment components
    primary_cols = get_assessment_components(course_data)
    
    # Calculate average for each component
    component_avgs = {col: course_data[col].mean() for col in primary_cols if col in course_data.columns}
    
    # For each student, find their weakest areas
    student_weaknesses = []
    
    if len(at_risk_data) > 0:
        for _, student in at_risk_data.iterrows():
            # Calculate relative performance for each component
            weaknesses = []
            for col in primary_cols:
                if not pd.isna(student[col]) and component_avgs.get(col, 0) > 0:
                    relative_performance = (student[col] - component_avgs[col]) / component_avgs[col]
                    weaknesses.append((col, relative_performance))
            
            # Sort by relative performance (ascending)
            weaknesses.sort(key=lambda x: x[1])
            
            # Get top 3 weaknesses (or fewer if less than 3 components)
            top_weaknesses = weaknesses[:min(3, len(weaknesses))]
            
            # Generate intervention recommendations
            interventions = []
            for area, _ in top_weaknesses:
                if 'MCQ' in area:
                    interventions.append(f"Additional objective assessment practice in {area}")
                elif 'MEQ' in area:
                    interventions.append(f"Writing workshop focusing on {area}")
                elif 'OSPE' in area:
                    interventions.append("Practical skills lab sessions")
                elif 'PAM' in area:
                    interventions.append("Revisit End of Module Assessment")
                elif 'PBL' in area:
                    interventions.append("Problem-based learning support group")
                else:
                    interventions.append(f"Additional support for {area}")
            
            student_info = {
                'Student_ID': student['Student_ID'],
                'Final_Mark': f"{student['Final Mark']:.2f}",
                'Weakest_Areas': ", ".join([w[0] for w in top_weaknesses]),
                'Interventions': ", ".join(interventions)
            }
            
            # Add course information if it exists and we're looking at combined data
            if 'Course' in student and selected_course == "All Courses":
                student_info['Course'] = student['Course']
                
            student_weaknesses.append(student_info)
    
    # At-risk students table
    st.subheader("At-Risk Students")
    if student_weaknesses:
        st.dataframe(student_weaknesses, use_container_width=True)
    else:
        st.write("No students identified as at-risk with the current threshold.")
    
    section_separator()
    
    # Visualization of at-risk students
    col1, col2 = st.columns(2)
    
    # Get the first two components for the scatter plot, if available
    scatter_cols = primary_cols[:2] if len(primary_cols) >= 2 else []
    
    with col1:
        if len(scatter_cols) == 2:
            # Create explicit category labels for at-risk status
            course_data['Risk Status'] = ['At Risk' if mark <= threshold else 'Not At Risk' 
                                          for mark in course_data['Final Mark']]
            
            # Scatter plot using the first two components
            fig = px.scatter(
                course_data,
                x=scatter_cols[0], y=scatter_cols[1],
                color='Risk Status',
                size="Final Mark",
                color_discrete_sequence=['red', 'lightgrey'],  # Now: red=At Risk, lightgrey=Not At Risk
                title=f"At-Risk Students (Red) by {scatter_cols[0]} vs {scatter_cols[1]} Performance",
                category_orders={"Risk Status": ["At Risk", "Not At Risk"]}  # Ensure consistent ordering
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough assessment components for scatter plot visualization")
    
    with col2:
        if primary_cols:
            # Box plot of component scores by at-risk status
            # Create a melted dataframe for the box plot
            melted_data = pd.DataFrame({
                'At_Risk': course_data['Final Mark'] <= threshold
            }).join(course_data).melt(
                id_vars=['Student_ID', 'Final Mark', 'At_Risk'],
                value_vars=primary_cols,
                var_name='Component', value_name='Score'
            )
            
            # Remove any NaN values
            melted_data = melted_data.dropna(subset=['Score'])
            
            if not melted_data.empty:
                fig = px.box(
                    melted_data,
                    x="Component", y="Score",
                    color="At_Risk",
                    title=f"Component Scores Distribution - {selected_course}",
                    color_discrete_sequence=['lightgrey', 'red'],
                    labels={"color": "At Risk", "At_Risk": "At Risk"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid data for box plot visualization")
        else:
            st.warning("No assessment components for box plot visualization")
