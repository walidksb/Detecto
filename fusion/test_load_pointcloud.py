import open3d as o3d

def main():
    pcd = o3d.io.read_point_cloud("reconstruction/output/scene1.ply")

    print("Number of points:", len(pcd.points))

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
