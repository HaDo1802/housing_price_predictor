"""
Housing Price Prediction - Streamlit Web Application

This app provides a user-friendly interface for predicting housing prices
using the trained Gradient Boosting model.

Features:
- Interactive input forms for all required features
- Input validation
- Prediction with uncertainty intervals
- Feature importance visualization
- Professional result formatting

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.inference_pipeline import InferencePipeline
from config.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .interval-text {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0.5rem;
    }
    .feature-section {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1890ff;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the inference pipeline (cached to avoid reloading)"""
    try:
        pipeline = InferencePipeline('models/production')
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


@st.cache_resource
def load_config():
    """Load configuration (cached)"""
    try:
        config_manager = ConfigManager('config/config.yaml')
        return config_manager.get_config()
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()


def validate_inputs(inputs: dict, required_features: list) -> tuple:
    """
    Validate user inputs.
    
    Returns:
        (is_valid: bool, missing_fields: list, error_messages: list)
    """
    missing_fields = []
    error_messages = []
    
    # Check for missing fields
    for feature in required_features:
        if feature not in inputs or inputs[feature] is None or inputs[feature] == '':
            missing_fields.append(feature)
    
    # Check for invalid numeric values
    for feature, value in inputs.items():
        if feature in required_features:
            # Skip if already missing
            if feature in missing_fields:
                continue
                
            # Validate numeric ranges
            if isinstance(value, (int, float)):
                if value < 0:
                    error_messages.append(f"{feature}: Cannot be negative")
    
    is_valid = len(missing_fields) == 0 and len(error_messages) == 0
    
    return is_valid, missing_fields, error_messages


def create_input_form(config):
    """Create the input form based on configured features"""
    
    st.markdown('<div class="main-header">🏠 Housing Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter property details to predict the sale price</div>', unsafe_allow_html=True)
    
    # Get features from config
    numeric_features = config.features.numeric
    categorical_features = config.features.categorical
    
    # Initialize session state for inputs
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {}
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📊 Property Features", "📍 Location & Style"])
    
    inputs = {}
    
    with tab1:
        st.subheader("Numerical Features")
        
        # Organize numeric features into columns
        col1, col2, col3 = st.columns(3)
        
        # Group features logically
        size_features = ['Lot Area', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Gr Liv Area', 'Garage Area']
        quality_features = ['Overall Qual', 'Overall Cond']
        year_features = ['Year Built', 'Year Remod/Add']
        room_features = ['Bedroom AbvGr', 'Full Bath', 'Half Bath', 'TotRms AbvGrd', 'Fireplaces', 'Garage Cars']
        
        with col1:
            st.markdown("**Size & Area**")
            for feature in size_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=None,
                        step=10.0,
                        help=f"Enter the {feature.lower()} in square feet" if 'SF' in feature or 'Area' in feature else f"Enter {feature.lower()}",
                        key=f"input_{feature}"
                    )
        
        with col2:
            st.markdown("**Quality & Year**")
            for feature in quality_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=1,
                        max_value=10,
                        value=None,
                        step=1,
                        help=f"Rate from 1 (poor) to 10 (excellent)",
                        key=f"input_{feature}"
                    )
            
            for feature in year_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=1800,
                        max_value=2026,
                        value=None,
                        step=1,
                        help=f"Enter the {feature.lower()}",
                        key=f"input_{feature}"
                    )
        
        with col3:
            st.markdown("**Rooms & Facilities**")
            for feature in room_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=0,
                        max_value=20,
                        value=None,
                        step=1,
                        help=f"Number of {feature.lower()}",
                        key=f"input_{feature}"
                    )
    
    with tab2:
        st.subheader("Categorical Features")
        
        col1, col2 = st.columns(2)
        
        # Define options for categorical features
        categorical_options = {
            'Neighborhood': [
                'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
                'Crawfor', 'Edwards', 'Gilbert', 'Greens', 'GrnHill', 'IDOTRR',
                'Landmrk', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes',
                'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW',
                'Somerst', 'StoneBr', 'Timber', 'Veenker'
            ],
            'MS Zoning': ['A (agr)', 'C (all)', 'FV', 'RH', 'RL', 'RM'],
            'Bldg Type': ['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'],
            'House Style': ['1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl'],
            'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
            'Central Air': ['N', 'Y'],
            'Garage Type': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'nan']
        }
        
        with col1:
            for i, feature in enumerate(categorical_features[:4]):
                if feature in categorical_options:
                    inputs[feature] = st.selectbox(
                        feature,
                        options=[''] + categorical_options[feature],
                        help=f"Select {feature.lower()}",
                        key=f"input_{feature}"
                    )
                    # Convert empty string to None
                    if inputs[feature] == '':
                        inputs[feature] = None
        
        with col2:
            for i, feature in enumerate(categorical_features[4:]):
                if feature in categorical_options:
                    inputs[feature] = st.selectbox(
                        feature,
                        options=[''] + categorical_options[feature],
                        help=f"Select {feature.lower()}",
                        key=f"input_{feature}"
                    )
                    # Convert empty string to None
                    if inputs[feature] == '':
                        inputs[feature] = None
    
    return inputs, numeric_features + categorical_features


def display_prediction_results(prediction, lower, upper, top_features):
    """Display prediction results in a professional format"""
    
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-value">${prediction:,.0f}</div>
        <div class="interval-text">
            Confidence Interval: ${lower:,.0f} - ${upper:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for additional info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Prediction Statistics")
        
        # Calculate margin of error
        margin = (upper - lower) / 2
        margin_pct = (margin / prediction) * 100
        
        stats_df = pd.DataFrame({
            'Metric': ['Predicted Price', 'Lower Bound', 'Upper Bound', 'Margin of Error', 'Confidence Level'],
            'Value': [
                f'${prediction:,.0f}',
                f'${lower:,.0f}',
                f'${upper:,.0f}',
                f'${margin:,.0f} (±{margin_pct:.1f}%)',
                '95%'
            ]
        })
        
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### 🔑 Top 5 Most Important Features")
        
        # Create horizontal bar chart for top features
        fig = go.Figure(go.Bar(
            x=top_features['importance'].values,
            y=top_features['feature'].values,
            orientation='h',
            marker=dict(
                color=top_features['importance'].values,
                colorscale='Blues',
                showscale=False
            ),
            text=top_features['importance'].apply(lambda x: f'{x:.4f}'),
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Feature Importance Scores",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Price range visualization
    st.markdown("### 📈 Price Range Visualization")
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=['Lower Bound', 'Prediction', 'Upper Bound'],
        y=[lower, prediction, upper],
        mode='markers+lines',
        marker=dict(size=[15, 25, 15], color=['#ff7f0e', '#1f77b4', '#ff7f0e']),
        line=dict(color='#1f77b4', width=2),
        name='Price Range'
    ))
    
    # Add shaded area for confidence interval
    fig.add_trace(go.Scatter(
        x=['Lower Bound', 'Prediction', 'Upper Bound'],
        y=[lower, lower, lower],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=['Lower Bound', 'Prediction', 'Upper Bound'],
        y=[upper, upper, upper],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(31, 119, 180, 0.2)',
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title="Predicted Price with Confidence Interval",
        xaxis_title="",
        yaxis_title="Price ($)",
        height=400,
        hovermode='x unified',
        yaxis=dict(tickformat='$,.0f')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    st.markdown(f"""
    <div class="info-box">
        <strong>💡 How to interpret these results:</strong><br>
        • The predicted sale price is <strong>${prediction:,.0f}</strong><br>
        • We are 95% confident the actual price will be between <strong>${lower:,.0f}</strong> and <strong>${upper:,.0f}</strong><br>
        • The top 5 features shown above had the most influence on this prediction<br>
        • The margin of error is <strong>±${margin:,.0f}</strong> ({margin_pct:.1f}%)
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application logic"""
    
    # Load pipeline and config
    with st.spinner("Loading prediction model..."):
        pipeline = load_pipeline()
        config = load_config()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1946/1946488.png", width=100)
        st.title("About")
        st.markdown("""
        This application uses a **Gradient Boosting Regressor** model 
        trained on the Ames Housing dataset to predict house prices.
        
        **Model Performance:**
        - R² Score: 0.917
        - RMSE: $25,793
        - MAE: $15,819
        
        **Instructions:**
        1. Fill in all required property details
        2. Click "Predict Price" button
        3. Review prediction with confidence interval
        4. Check which features influenced the prediction most
        """)
        
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.info(f"Model: {pipeline.metadata['model_type']}")
        st.info(f"Features: {len(pipeline.metadata['feature_names'])}")
        
        st.markdown("---")
        st.markdown("**Need Help?**")
        st.markdown("""
        - All fields are required
        - Numerical values must be non-negative
        - Quality ratings are 1-10
        - Years should be between 1800-2026
        """)
    
    # Main content
    inputs, required_features = create_input_form(config)
    
    # Predict button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Price", use_container_width=True, type="primary")
    
    if predict_button:
        # Validate inputs
        is_valid, missing_fields, error_messages = validate_inputs(inputs, required_features)
        
        if not is_valid:
            # Display errors
            if missing_fields:
                st.markdown(f"""
                <div class="error-message">
                    <strong>❌ Missing Required Fields:</strong><br>
                    {', '.join(missing_fields)}
                </div>
                """, unsafe_allow_html=True)
            
            if error_messages:
                st.markdown(f"""
                <div class="error-message">
                    <strong>❌ Invalid Values:</strong><br>
                    {'<br>'.join(error_messages)}
                </div>
                """, unsafe_allow_html=True)
            
            st.warning("⚠️ Please fill in all required fields with valid values before making a prediction.")
        
        else:
            # Make prediction
            with st.spinner("🔄 Calculating prediction..."):
                try:
                    # Convert inputs to dataframe
                    input_df = pd.DataFrame([inputs])
                    
                    # Get prediction with uncertainty
                    predictions, lower_bounds, upper_bounds = pipeline.predict_with_uncertainty(input_df)
                    
                    prediction = predictions[0]
                    lower = prediction + lower_bounds[0]
                    upper = prediction + upper_bounds[0]
                    
                    # Get feature importance
                    top_features = pipeline.get_feature_importance(top_n=5)
                    
                    # Display results
                    display_prediction_results(prediction, lower, upper, top_features)
                    
                    # Success message
                    st.success("✅ Prediction completed successfully!")
                    
                    # Download button for results
                    results_data = {
                        'Predicted Price': [f'${prediction:,.0f}'],
                        'Lower Bound (95% CI)': [f'${lower:,.0f}'],
                        'Upper Bound (95% CI)': [f'${upper:,.0f}'],
                        'Margin of Error': [f'${(upper - lower) / 2:,.0f}'],
                    }
                    results_df = pd.DataFrame(results_data)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=csv,
                        file_name="house_price_prediction.csv",
                        mime="text/csv",
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Housing Price Predictor v1.0 | Built with Streamlit | 
        Powered by Gradient Boosting ML Model</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()