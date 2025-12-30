from scene_manager import create_new_scene
from streamlit_app.run_pipeline import run_inspection
from streamlit_app.utils import pcd_to_numpy, crack_stats, export_confidence_csv
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
import sys

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
        status = st.empty()
        progress = st.progress(0)

        try:
            status.info("Running 3D reconstruction (COLMAP)...")
            progress.progress(20)

            pcd, ply_path = run_inspection(
                scene_dir=st.session_state["scene_dir"],
                model_path="detection/models/exported/crack_unet_v1.pth"
            )


            progress.progress(90)
            status.info("Finalizing 3D crack fusion...")

            st.session_state["pcd"] = pcd
            st.session_state["ply_path"] = ply_path
            progress.progress(100)
            status.success("Inspection completed successfully!")

        except Exception as e:
            status.error("Inspection failed")
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

if "pcd" in st.session_state:
    st.subheader("📊 Crack Statistics")

    stats = crack_stats("analysis/crack_confidence.npy")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total 3D Points", stats["total_points"])
    col2.metric("Crack Points", stats["crack_points"])
    col3.metric("Crack Ratio", f"{stats['crack_ratio']:.2%}")

    st.metric("Max Crack Confidence", f"{stats['max_confidence']:.2f}")
    st.metric("Mean Confidence", f"{stats['mean_confidence']:.2f}")


results_dir = Path(st.session_state["scene_dir"]) / "results"
csv_path = results_dir / "crack_confidence.csv"

export_confidence_csv(
    "analysis/crack_confidence.npy",
    csv_path
)

st.session_state["csv_path"] = csv_path

st.subheader("💾 Download Results")

col1, col2 = st.columns(2)

with col1:
    with open(st.session_state["ply_path"], "rb") as f:
        st.download_button(
            label="⬇ Download 3D Crack Point Cloud (.ply)",
            data=f,
            file_name="crack_localization.ply",
            mime="application/octet-stream"
        )

with col2:
    with open(st.session_state["csv_path"], "rb") as f:
        st.download_button(
            label="⬇ Download Crack Confidence (.csv)",
            data=f,
            file_name="crack_confidence.csv",
            mime="text/csv"
        )
