import streamlit as st
import pandas as pd
import numpy as np
import os
from student_analysis2 import analyze_student_performance, analyze_multiple_courses

# Import visualization modules
from tabs.overview_tab import render_overview_tab
from tabs.intervention_tab import render_intervention_tab
from tabs.clusters_tab import render_clusters_tab
from tabs.components_tab import render_components_tab
from tabs.portfolio_tab import render_portfolio_tab
from utils.helper_functions import section_separator, get_assessment_components

# Set page config
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'course_files' not in st.session_state:
    st.session_state.course_files = {}  # Dictionary to store course files
if 'results' not in st.session_state:
    st.session_state.results = None  # To store analysis results
if 'selected_courses' not in st.session_state:
    st.session_state.selected_courses = []  # To store selected courses for analysis
if 'analyze_button_clicked' not in st.session_state:
    st.session_state.analyze_button_clicked = False

# App title
st.title("Multi-Course Student Performance Analysis Dashboard")

# Sidebar for file uploads and course selection
with st.sidebar:
    st.header("Course Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a course CSV file", type="csv", key="file_upload")
    
    if uploaded_file is not None:
        # Get course name from user
        course_name = st.text_input("Enter a name for this course:", f"Course {len(st.session_state.course_files) + 1}")
        
        # Add button to add the course
        if st.button("Add Course"):
            # Save the file to a temporary location
            temp_file_path = f"temp_{course_name.replace(' ', '_')}.csv"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add to course files dictionary
            st.session_state.course_files[course_name] = temp_file_path
            st.success(f"Added {course_name} to courses!")
    
    # Display list of added courses
    if st.session_state.course_files:
        st.subheader("Added Courses")
        for i, (course, _) in enumerate(st.session_state.course_files.items()):
            st.write(f"{i+1}. {course}")
        
        # Option to remove courses
        course_to_remove = st.selectbox("Select course to remove:", 
                                      ["None"] + list(st.session_state.course_files.keys()),
                                      key="remove_course")
        
        if course_to_remove != "None" and st.button("Remove Course"):
            # Delete the temporary file
            if os.path.exists(st.session_state.course_files[course_to_remove]):
                os.remove(st.session_state.course_files[course_to_remove])
            
            # Remove from dictionary
            del st.session_state.course_files[course_to_remove]
            st.success(f"Removed {course_to_remove}!")
            st.rerun()  # Rerun the app to update the sidebar

    # Multiselect for courses to analyze
    if st.session_state.course_files:
        st.subheader("Select Courses to Analyze")
        st.session_state.selected_courses = st.multiselect(
            "Choose courses",
            options=list(st.session_state.course_files.keys()),
            default=list(st.session_state.course_files.keys()),
            key="course_selector"
        )
    
    # In the sidebar of app.py, after course selection
    if st.session_state.course_files:
        st.sidebar.subheader("Assessment Component Weights")
        
        # Initialize weights dictionary if not in session
        if 'component_weights' not in st.session_state:
            st.session_state.component_weights = {}
        
        # Get all unique component names across all uploaded files
        all_components = set()
        for course_name, file_path in st.session_state.course_files.items():
            try:
                # Read a few rows to extract column names
                sample_data = pd.read_csv(file_path, nrows=5)
                # Get assessment components
                exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster']
                components = [col for col in sample_data.columns 
                            if col not in exclude_cols and not col.startswith('Essay')]
                all_components.update(components)
            except Exception as e:
                st.sidebar.warning(f"Could not read components from {course_name}: {e}")
        
        # Display weight sliders for each component
        if all_components:
            st.sidebar.write("Adjust component weights (should sum to 100):")
            
            # Calculate initial equal weights
            equal_weight = 100 / len(all_components)
            total_weight = 0
            
            # Create a slider for each component
            for component in sorted(all_components):
                # Get existing weight or use equal weight as default
                current_weight = st.session_state.component_weights.get(component, equal_weight)
                
                # Create a slider for this component
                new_weight = st.sidebar.number_input(
                    f"{component} weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_weight),
                    step=0.5,
                    key=f"weight_{component}"
                )
                
                # Update the weight in session state
                st.session_state.component_weights[component] = new_weight
                total_weight += new_weight
            
            # Show the total weight
            weight_color = "green" if 99.0 <= total_weight <= 101.0 else "red"
            st.sidebar.markdown(f"**Total weight: <span style='color:{weight_color}'>{total_weight:.1f}%</span>**", 
                            unsafe_allow_html=True)
            
            # Normalize weights button
            if st.sidebar.button("Normalize Weights to 100%"):
                if total_weight > 0:
                    normalization_factor = 100 / total_weight
                    for component in st.session_state.component_weights:
                        st.session_state.component_weights[component] *= normalization_factor
                    st.sidebar.success("Weights normalized to sum to 100%")
                    st.rerun()
        
        # After component weight settings in the sidebar
        if all_components:
            st.sidebar.write("Component maximum possible scores:")
            
            # Initialize max scores dictionary if not in session
            if 'max_scores' not in st.session_state:
                st.session_state.max_scores = {}
            
            # Create an input field for each component's max score
            for component in sorted(all_components):
                # Get existing max score or use 100 as default
                current_max = st.session_state.max_scores.get(component, 100)
                
                # Create an input for this component's max score
                new_max = st.sidebar.number_input(
                    f"{component} max score",
                    min_value=1.0,
                    max_value=1000.0,
                    value=float(current_max),
                    step=1.0,
                    key=f"max_{component}"
                )
                
                # Update the max score in session state
                st.session_state.max_scores[component] = new_max

        # Button to run analysis
        if st.sidebar.button("Analyze Selected Courses"):
            with st.spinner("Running analysis..."):
                # Create a dictionary with only the selected courses
                selected_files = {course: st.session_state.course_files[course] 
                                for course in st.session_state.selected_courses}
                
                # Run the analysis for selected courses with custom weights
                st.session_state.results = analyze_multiple_courses(
                    selected_files,
                    component_weights=st.session_state.component_weights,
                    component_max_scores=st.session_state.max_scores
                )
                st.session_state.analyze_button_clicked = True
            
            st.sidebar.success("Analysis complete!")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This dashboard analyzes student performance across multiple courses. "
        "Upload CSV files containing student assessment data to generate insights "
        "on at-risk students, performance clusters, and assessment components."
    )

# Main content area - only show if analysis has been run
if st.session_state.analyze_button_clicked and st.session_state.results:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Early Intervention", "Student Clusters", "Component Analysis", "Student e-Portfolio"])
    
    # Get all course names including "All Courses" if it exists
    all_course_names = list(st.session_state.results.keys())
    
    # Tab 1: Overview
    with tab1:
        render_overview_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
    
    # Tab 2: Early Intervention
    with tab2:
        render_intervention_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
    
    # Tab 3: Student Clusters
    with tab3:
        render_clusters_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
    
    # Tab 4: Component Analysis
    with tab4:
        render_components_tab(st.session_state.results, all_course_names, section_separator)
    
    # Tab 5: Student e-Portfolio
    with tab5:
        render_portfolio_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
else:
    # Display initial instructions if no analysis has been run
    st.info(
        """
        ### Welcome to the Multi-Course Student Performance Analysis Dashboard!
        
        To get started:
        1. Use the sidebar to upload CSV files containing your course data
        2. Name each course and click "Add Course"
        3. Once you've added all courses, select the ones you want to analyze
        4. Click "Analyze Selected Courses" to run the analysis
        
        Your CSV files should have the following structure:
        - Student_ID: Student identifier
        - Assessment components (MCQ, MEQ, OSPE, PAM, PBL, etc.)
        - Final Mark: Overall final mark
        
        The analysis will provide insights on student intervention needs, performance clusters, and assessment components.
        """
    )

    # Sample CSV structure
    st.subheader("Example CSV Structure")
    sample_data = pd.DataFrame({
        'Student_ID': ['A161342', 'A152487', 'A178642'],
        'MCQ': [15.5, 18.2, 12.7],
        'MEQ': [20.1, 19.5, 17.8],
        'OSPE': [6.5, 7.8, 5.9],
        'PAM': [12.5, 14.2, 11.8],
        'PBL': [8.5, 9.2, 7.8],
        'Final Mark': [67.5, 72.4, 61.2]
    })
    st.dataframe(sample_data)