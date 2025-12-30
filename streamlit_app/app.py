import streamlit as st
from scene_manager import create_new_scene
from streamlit_app.run_pipeline import run_inspection
import plotly.graph_objects as go
from streamlit_app.utils import pcd_to_numpy
import sys
from pathlib import Path
import open3d as o3d

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))



st.set_page_config(
    page_title="3D Crack Inspection",
    layout="wide"
)

st.title("🧱 3D Crack Inspection System")

st.markdown("""
Upload multiple images of a structure.
The system will reconstruct the scene in 3D and localize cracks.
""")

uploaded_files = st.file_uploader(
    "Upload inspection images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    scene_dir, images_dir = create_new_scene()

    for file in uploaded_files:
        file_path = images_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.success(f"{len(uploaded_files)} images saved to {scene_dir.name}")
    st.session_state["scene_dir"] = str(scene_dir)

if "scene_dir" in st.session_state:
    if st.button("🚀 Run Inspection"):
        with st.spinner("Running reconstruction and crack detection..."):
            try:
                pcd = run_inspection(
                    scene_dir=st.session_state["scene_dir"],
                    model_path="detection/models/exported/crack_unet_v1.pth"
                )

                st.session_state["pcd"] = pcd
                st.success("Inspection completed successfully!")

            except Exception as e:
                st.error("Inspection failed")
                st.exception(e)

if "pcd" in st.session_state:
    st.subheader("🧠 3D Crack Localization Result")

    points, colors = pcd_to_numpy(st.session_state["pcd"])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=colors,
                    opacity=0.9
                )
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)
