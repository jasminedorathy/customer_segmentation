import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
.upload-box {
    padding: 30px;
    border: 2px dashed #ccc;
    border-radius: 10px;
    text-align: center;
    margin: 20px 0;
    transition: all 0.3s;
}
.upload-box:hover {
    border-color: #4CAF50;
}
.error {
    color: #ff4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Customer Segmentation Dashboard")
st.markdown("""
**Drag and drop your customer data file below**  
Supported formats: CSV, Excel (.xlsx)
""")

# --- Drag-and-Drop File Upload ---
uploaded_file = st.file_uploader(
    label=" ",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
    key="file_uploader",
    help="Drag and drop your file here or click to browse"
)

if uploaded_file:
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        # Show raw data preview
        with st.expander("🔍 Preview Raw Data (First 5 Rows)"):
            st.dataframe(data.head())

        # --- Data Processing ---
        st.subheader("⚙️ Data Processing")
        
        # Auto-detect RFM columns (case-insensitive)
        col_mapping = {
            'recency': ['recency', 'days_since_last', 'recency_days', 'r_score'],
            'frequency': ['frequency', 'purchase_count', 'orders', 'f_score'],
            'monetary': ['monetary', 'total_spend', 'revenue', 'm_score', 'amount']
        }
        
        detected_cols = {}
        for rfm_type, possible_names in col_mapping.items():
            for col in data.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    detected_cols[rfm_type] = col
                    break
        
        if len(detected_cols) == 3:
            # Rename columns to standard names
            data = data.rename(columns={
                detected_cols['recency']: 'recency',
                detected_cols['frequency']: 'frequency',
                detected_cols['monetary']: 'monetary'
            })
            st.success("✅ RFM columns automatically detected!")
        else:
            st.error("""
            ❌ Could not detect required RFM columns.  
            **We need:**  
            - Recency (days since last purchase)  
            - Frequency (number of orders)  
            - Monetary (total spend)  
            Found columns: """ + str(data.columns.tolist()))
            st.stop()

        # Feature scaling
        features = data[['recency', 'frequency', 'monetary']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        data['cluster'] = clusters
        data['segment'] = data['cluster'].map({
            0: "Low-Value",
            1: "Medium-Value",
            2: "High-Value",
            3: "Premium"
        })
        
        # --- Visualizations ---
        st.subheader("📈 Cluster Analysis")
        
        # 1. Cluster Distribution
        cluster_counts = data['segment'].value_counts().reset_index()
        cluster_counts.columns = ['segment', 'count']
        
        fig1 = px.bar(
            cluster_counts,
            x='segment',
            y='count',
            color='segment',
            title="Customer Distribution by Segment",
            text='count'
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. 3D Scatter Plot
        fig2 = px.scatter_3d(
            data,
            x='recency',
            y='frequency',
            z='monetary',
            color='segment',
            hover_name=data.columns[0],
            title="3D Customer Segmentation",
            labels={
                'recency': 'Recency (days)',
                'frequency': 'Frequency',
                'monetary': 'Monetary ($)'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Cluster Statistics
        with st.expander("📊 Detailed Cluster Statistics"):
            st.dataframe(
                data.groupby('segment').agg({
                    'recency': ['mean', 'median'],
                    'frequency': ['mean', 'median'],
                    'monetary': ['mean', 'median']
                }).style.background_gradient()
            )
        
        # Download results
        st.download_button(
            label="💾 Download Segmented Data",
            data=data.to_csv(index=False),
            file_name="customer_segments.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"🚨 Error processing file: {str(e)}")
        st.stop()

else:
    st.info("ℹ️ Please upload a file to get started")