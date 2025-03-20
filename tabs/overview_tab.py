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
        st.write(f"Median Final Mark: {course_data['Final Mark'].median():.2f}")
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
        st.subheader("Component Statistics")
        try:
            st.write(f"Average MCQ: {course_data['MCQ'].mean():.2f}")
        except KeyError:
            st.write("Average MCQ: N/A")
        try:
            st.write(f"Average MEQ: {course_data['MEQ'].mean():.2f}")
        except KeyError:
            st.write("Average MEQ: N/A")
        try:
            st.write(f"Average OSPE: {course_data['OSPE'].mean():.2f}")
        except KeyError:
            st.write("Average OSPE: N/A")
        try:
            st.write(f"Average PAM: {course_data['PAM'].mean():.2f}")
        except KeyError:
            st.write("Average PAM: N/A")
        try:
            st.write(f"Average PBL: {course_data['PBL'].mean():.2f}")
        except KeyError:
            st.write("Average PBL: N/A")
    
    with col2:
        # Create total number of students in each Grade
        st.subheader("Student Grade Distribution")
        
        # Define grade boundaries
        grade_boundaries = {
            'A': 80,
            'A-': 75,
            'B+': 70,
            'B': 65,
            'B-': 60,
            'C+': 55,
            'C': 50,
            'C-': 45,
            'D+': 40,
            'D': 35,
            'E': 0
        }
        
        # Create a function to assign grades based on final marks
        def assign_grade(mark):
            for grade, min_mark in grade_boundaries.items():
                if mark >= min_mark:
                    return grade
            return 'E'  # Default if no other condition is met
        
        # Apply the function to create a new grade column
        grade_data = course_results['data'].copy()
        grade_data['Grade'] = grade_data['Final Mark'].apply(assign_grade)
        
        # Count students in each grade
        grade_counts = grade_data['Grade'].value_counts().reset_index()
        grade_counts.columns = ['Grade', 'Count']
        
        # Define grade order for sorting (reversed from your dictionary to get A at the top)
        grade_order = list(grade_boundaries.keys())
        
        # Map grades to a custom sort order using a dictionary
        grade_sort_map = {grade: i for i, grade in enumerate(grade_order)}
        
        # Sort the dataframe according to the custom order
        grade_counts['Sort_Order'] = grade_counts['Grade'].map(grade_sort_map)
        grade_counts = grade_counts.sort_values('Sort_Order')
        grade_counts = grade_counts.drop('Sort_Order', axis=1)
        
        # Create a color map for grades
        colors = px.colors.qualitative.Bold
        color_discrete_map = {grade: colors[i % len(colors)] for i, grade in enumerate(grade_order)}
        
        # Create a bar chart
        fig = px.bar(
            grade_counts,
            x='Grade', 
            y='Count',
            title=f"Student Grade Distribution - {selected_course}",
            color='Grade',
            color_discrete_map=color_discrete_map,
            text='Count',
            category_orders={"Grade": grade_order}
        )
        
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the data in a table below the chart
        st.write("Number of students in each grade:")
        st.dataframe(grade_counts, use_container_width=True)
    
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
                    'Pass Rate (%)': (course_data['Final Mark'] >= 50).mean() * 100,
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
                title="Pass Rate by Course (≥50)",
                color='Course'
            )
            st.plotly_chart(fig, use_container_width=True)
