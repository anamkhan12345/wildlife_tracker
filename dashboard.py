import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sqlite3
import pandas as pd
import datetime

st.set_page_config(page_title="Detection Dashboard", layout="wide")

# Auto-refresh every 10 seconds
st_autorefresh(interval=10000, key="my_autorefresh_key")

def live_dashboard():
    st.title("🎯 Object Detection Dashboard")

    # Get all stats
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()

    cursor.execute("SELECT SUM(detection_count) FROM detections")
    total_all_time = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT SUM(detection_count) 
        FROM detections 
        WHERE DATE(timestamp) = DATE('now')
    """)
    total_today = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT image_path, timestamp
        FROM detections 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    latest_result = cursor.fetchone()
    latest_image = latest_result[0] if latest_result else None
    latest_timestamp = latest_result[1] if latest_result else None

    cursor.execute("""
        SELECT image_path, max_detection_area
        FROM detections 
        ORDER BY max_detection_area DESC 
        LIMIT 1
    """)
    largest_result = cursor.fetchone()
    largest_image = largest_result[0] if largest_result else None
    largest_area = largest_result[1] if largest_result else 0

    cursor.execute("""
        SELECT 
            hour_of_day,
            AVG(detection_count) as avg_detections
        FROM detections
        GROUP BY hour_of_day
        ORDER BY CAST(hour_of_day AS INTEGER)
    """)
    hourly_data = cursor.fetchall()
    hourly_df = pd.DataFrame(hourly_data, columns=['Hour', 'Avg Detections'])

    conn.close()

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Detections", total_all_time)
    with col2:
        st.metric("Detections Today", total_today)

    # Images with timestamps
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Largest Detection")
        if largest_image:
            st.image(largest_image, use_container_width=True, caption=f"Area: {largest_area:.0f} px²")
        else:
            st.info("No detections yet")

    with col2:
        st.subheader("Latest Detection")
        if latest_image:
            st.image(latest_image, use_container_width=True)
            # Display timestamp below image
            if latest_timestamp:
                # Format the timestamp nicely
                try:
                    dt = datetime.strptime(latest_timestamp, "%Y-%m-%d %H:%M:%S")
                    formatted_time = dt.strftime("%B %d, %Y at %I:%M:%S %p")
                    st.caption(f"📅 {formatted_time}")
                except:
                    # If timestamp format is different, just show it as-is
                    st.caption(f"📅 {latest_timestamp}")
        else:
            st.info("No detections yet")

    # Graph
    st.subheader("Average Detections Per Hour")
    if not hourly_df.empty:
        st.line_chart(hourly_df.set_index('Hour'))
    else:
        st.info("No data yet")

# Run the live dashboard
live_dashboard()