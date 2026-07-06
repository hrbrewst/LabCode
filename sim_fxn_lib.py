# functions for simulation
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import mujoco
import yaml
import scipy
import xml.etree.ElementTree as ET
import os
import mujoco

def initialize_simulation(tMax, dt, xml_path, stl, camera_name="diag", resolution=(960, 540), framerate=100):
    framerate = 100
    base_dir = os.getcwd()
    stl_path = os.path.join(base_dir, 'asset', stl)

    model, data = load_mujoco_model(xml_path)
    mujoco.mj_resetData(model, data)

    renderer = mujoco.Renderer(model, height=resolution[1], width=resolution[0])
    renderer.update_scene(data, camera=camera_name)

    t = np.arange(0, tMax, dt)
    model.opt.timestep = dt

    sand_h_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "sand_height")

    frames = []

    return model, data, renderer, t, dt, frames, framerate, sand_h_id, stl_path

def load_mujoco_model(path):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data

def load_control_data_yaml(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)

    gait = y.get("gait", {})
    out = {}

    def to_rad(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr
        # If magnitudes exceed 2π, assume degrees and convert
        return np.deg2rad(arr) if np.nanmax(np.abs(arr)) > (2*np.pi + 1e-6) else arr

    for leg, joints in gait.items():           # legs: L1 R1 L2 R2 L3 R3
        knee  = to_rad(joints.get("knee",  []))
        ankle = to_rad(joints.get("ankle", []))
        out[f"theta1_{leg}"] = knee            # theta1_* == knee
        out[f"theta2_{leg}"] = ankle           # theta2_* == ankle

    # Optional: keep useful metadata if you need it later
    out["_meta"] = {
        "heading": y.get("heading"),
        "step_length": y.get("step_length"),
        "step_height": y.get("step_height"),
        "stance_frac": y.get("stance_frac"),
        "swing_path_rotation": y.get("swing_path_rotation"),
        "phases": y.get("phases", {}),
        "neutral_points": y.get("neutral_points", {}),
        "init_pos_local": y.get("init_pos_local", {}),
        "delta_pos_local": y.get("delta_pos_local", {}),
    }
    return out

def interpolate_array(b, L, x):
    repeated_b = np.tile(b, x)
    length_b_interp = L
    b_interp_func = scipy.interpolate.interp1d(np.arange(len(repeated_b)), repeated_b, kind='linear', fill_value='extrapolate')
    b_interpolated = b_interp_func(np.linspace(0, len(repeated_b) - 1, length_b_interp))
    return b_interpolated

def get_named_bodies_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    namespace = {'mujoco': 'http://mujoco.org/schema'}
    
    body_names = [
        body.get("name")
        for body in root.findall(".//body", namespace)
        if body.get("name")
    ]

    return body_names

def load_and_process_mesh(stl_path, scale_factor=1000):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)
    
    # Scale the vertices
    vertices_np = np.asarray(mesh.vertices)
    vertices_np = vertices_np / scale_factor
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    
    # Get vertices and faces
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Create body from mesh
    body = body_from_mesh(mesh)
    
    return body, vertices, faces, mesh

def body_from_mesh(mesh):

    # Ensure normals are computed
    mesh.compute_triangle_normals()  # Open3D handles normals
    
    # Get normal vectors of each triangle (face)
    n_vec = np.asarray(mesh.triangle_normals)
    
    # Calculate centroid of each triangle
    r_vec = np.zeros_like(n_vec)
    for i, face in enumerate(mesh.triangles):  # Open3D uses 'triangles' instead of 'faces'
        p1, p2, p3 = np.asarray(mesh.vertices)[face]
        r_vec[i, :] = np.mean([p1, p2, p3], axis=0)

    # Compute area of each triangle
    A_vec = np.zeros(len(n_vec))
    for i, face in enumerate(mesh.triangles):
        r1, r2, r3 = np.asarray(mesh.vertices)[face]
        A_vec[i] = np.linalg.norm(np.cross(r2 - r1, r3 - r1)) / 2

    body = {'r': r_vec, 'n': n_vec, 'A': A_vec}
    return body

def initialize_sites_on_mesh(model, data, mesh, sitename="force", bodyname="flipper_1"):
    """Initialize site positions in body frame coordinates (called once)"""
    vertices = np.asarray(mesh.vertices)
    num_sites = model.nsite
    num_faces = len(mesh.triangles)
    num_active_sites = min(num_sites, num_faces)
    # print(f"num active sites: { num_active_sites}  body: {bodyname} ")
    # Store local positions relative to body frame
    for i in range(num_active_sites):
        site_name = f"{sitename}_site_{i}"
        # print(site_name)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        
        face = np.asarray(mesh.triangles)[i]
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        centroid_local = (v0 + v1 + v2) / 3.0
        
        # Set site position in local coordinates
        model.site_pos[site_id] = centroid_local
        
        # Set site parent to be the flipper body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bodyname)
        model.site_bodyid[site_id] = body_id

    # Forward kinematics to update all positions
    mujoco.mj_forward(model, data)

def quaternion_to_euler(quat):
    # Create a Rotation object from the quaternion
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Note: MuJoCo uses [w, x, y, z] order
    # Convert to Euler angles (roll, pitch, yaw) in radians
    euler = r.as_euler('xyz', degrees=False)
    return euler  # Returns roll, pitch, yaw

def rft_3D_body_full_mat(body_m, origin_m, orientation_rad, vel_mps, ang_vel_radps, RFTCOEFF=3.75, sand_height_m=0.0):
    # Convert inputs to centimeter units at the beginning
    origin_cm = origin_m * 100
    r_cm = body_m['r'] * 100
    area_cm2 = body_m['A'] * 10000
    vel_cmps = vel_mps * 100
    sand_height_cm = sand_height_m * 100
    psi = orientation_rad[0]   # yaw (rotation about E3 axis)
    theta = orientation_rad[1] # pitch (rotation down from E3)
    phi = orientation_rad[2]

    # Define rotation matrices
    Rz = np.array([[np.cos(psi), np.sin(psi), 0],
                   [-np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(theta), 0, -np.sin(theta)],
                   [0, 1, 0],
                   [np.sin(theta), 0, np.cos(theta)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), np.sin(phi)],
                   [0, -np.sin(phi), np.cos(phi)]])

    # Total rotation matrix (assuming Z-Y-X Euler angles)
    R = Rx @ Ry @ Rz

    # Rotate position & normal vectors of all faces based on current orientation
    r_rot_vec_cm = (R.T @ r_cm.T).T
    depth_cm = origin_cm[2] + r_rot_vec_cm[:, 2]
    n_rot_vec = (R.T @ body_m['n'].T).T

    # Calculate velocity of each plate element based on body velocities (in cm/s and cm)
    n_elements = r_cm.shape[0]
    ang_vel_radps_tiled = np.tile(ang_vel_radps, (n_elements, 1))
    v_vec_cmps = vel_cmps + np.cross(ang_vel_radps_tiled, r_rot_vec_cm)

    # Normalize velocity & normal vectors
    v_norm_vec = v_vec_cmps / (np.linalg.norm(v_vec_cmps, axis=1, keepdims=True) + 1e-10)
    n_norm_vec = n_rot_vec / (np.linalg.norm(n_rot_vec, axis=1, keepdims=True) + 1e-10)

    # Logical conditions to count force on element
    leading_edge = np.sum(n_norm_vec * v_norm_vec, axis=1) > 0
    intruding = depth_cm < sand_height_cm
    include = leading_edge & intruding

    # Isolate the elements that satisfy this condition
    n_inc = n_norm_vec[include, :]
    v_inc = v_norm_vec[include, :]
    r_inc_cm = r_rot_vec_cm[include, :]




    # generate basis for RFT decomposition
    n_inc_z_abs = np.abs(n_inc[:, 2])
    horizontal_case = n_inc_z_abs >= 1 - 1e-3
    nominal_case = n_inc_z_abs < 1 - 1e-3

    v_inc_modified = v_inc + np.array([1e-5, 0, 0])
    e2_vec = np.zeros_like(n_inc)
    e2_vec[horizontal_case, :] = (v_inc_modified[horizontal_case, :] * np.array([1, 1, 0])) / np.linalg.norm((v_inc_modified[horizontal_case, :] * np.array([1, 1, 0])), axis=1, keepdims=True)
    e2_vec[nominal_case, :] = (n_inc[nominal_case, :] * np.array([1, 1, 0])) / np.linalg.norm((n_inc[nominal_case, :] * np.array([1, 1, 0])), axis=1, keepdims=True)

    e1_vec = np.cross(e2_vec, np.tile([0, 0, 1], (np.sum(include), 1)))

    # decompose velocity vector
    v1_vec = np.sum(v_inc * e1_vec, axis=1, keepdims=True) * e1_vec
    v23_vec = v_inc - v1_vec

    # calculate 2D RFT parameters
    beta_vec = np.arctan2(n_inc[:, 2], np.sum(n_inc * e2_vec, axis=1)) + np.pi / 2
    gamma_vec = -np.arctan2(v23_vec[:, 2], np.sum(v23_vec * e2_vec, axis=1))
    # RFTCOEFF = 3.75
    # RFTCOEFF = 2
    # from Li et al. -> M matrices for each granular medium
    M_generic = np.array([0.206, 0.169, 0.212, 0.358, 0.055, -0.124, 0.253, 0.007, 0.088])
    M_testing = M_generic * RFTCOEFF

    # 2D alpha components (output in N/cm^3)
    def afunc(gamma_vec, beta_vec, M):
        fit_z = 0
        fit_x = 0
        A_0_0, A_1_0, B_1_1, B_0_1, B_neg1_1, C_1_1, C_0_1, C_neg1_1, D_1_0 = M
        for m in range(-1, 2):
            for n in range(2):
                arg = 2 * np.pi * (m * beta_vec / np.pi + n * gamma_vec / (2 * np.pi))
                if m == 0 and n == 0: Amn, Bmn, Cmn, Dmn = A_0_0, 0, 0, 0
                elif m == 1 and n == 0: Amn, Bmn, Cmn, Dmn = A_1_0, 0, 0, D_1_0
                elif m == 1 and n == 1: Amn, Bmn, Cmn, Dmn = 0, B_1_1, C_1_1, 0
                elif m == 0 and n == 1: Amn, Bmn, Cmn, Dmn = 0, B_0_1, C_0_1, 0
                elif m == -1 and n == 1: Amn, Bmn, Cmn, Dmn = 0, B_neg1_1, C_neg1_1, 0
                else: Amn, Bmn, Cmn, Dmn = 0, 0, 0, 0
                fit_z += Amn * np.cos(arg) + Bmn * np.sin(arg)
                fit_x += Cmn * np.cos(arg) + Dmn * np.sin(arg)
        return fit_z, fit_x

    aZ, aX = afunc(gamma_vec, beta_vec, M_testing)  # in N/cm^3
    _, aY = afunc(np.array([0]*len(gamma_vec)), np.array([0]*len(beta_vec)), M_testing)  # in N/cm^3

    # Calculate scaling factors (dimensionless)
    vt = np.linalg.norm(v1_vec, axis=1)
    ct_fit_A = [0.440850096954369, 3.62263982880971, 1.60910808139526, 0.41]
    f1 = ct_fit_A[0] * (np.tanh(ct_fit_A[1] * vt - ct_fit_A[2]) + np.tanh(ct_fit_A[2])) / ct_fit_A[3]

    vn = np.linalg.norm(v23_vec, axis=1)
    cn_fit_B = [1.99392673405210, 1.61146827229181, 0.973746396532650, 4.31]
    f23 = (
        cn_fit_B[0] *
        (np.arctanh(cn_fit_B[1] * vn - cn_fit_B[2]) + np.arctanh(cn_fit_B[2])) /
        cn_fit_B[3])

    area_cm2 = body_m['A'][include] * 10000
    depth_cm_included = depth_cm[include]

    # Calculate forces in Newtons
    f1_magnitude = -f1.reshape(-1, 1) * aY.reshape(-1, 1) * np.sign(np.einsum('ij,ij->i', v_inc, e1_vec)).reshape(-1, 1) * depth_cm_included[:, None] * area_cm2[:, None]
    F1 = f1_magnitude * e1_vec

    f2_magnitude = -f23.reshape(-1, 1) * aX.reshape(-1, 1) * depth_cm_included[:, None] * area_cm2[:, None]
    F2 = f2_magnitude * e2_vec

    f3_magnitude = f23.reshape(-1, 1) * aZ.reshape(-1, 1) * depth_cm_included[:, None] * area_cm2[:, None]
    F3 = f3_magnitude * np.array([0, 0, 1])
    F_i = F1 + F2 + F3

    # Moments need to be calculated using the lever arm in meters (cross product with force in Newtons)
    r_inc_m = body_m['r'][include]
    Mi_mat = np.cross(r_inc_m, F_i)
    M_total = np.sum(Mi_mat, axis=0)

    F_mat_full = np.zeros((body_m['r'].shape[0], 3))
    F_mat_full[include] = F_i
    F_total = np.sum(F_i, axis=0)

    return F_total, M_total, F_i, Mi_mat, include, F_mat_full, RFTCOEFF

def sort_by_face_centroid_x(faces, vertices, site_ids, F_full):
    face_centroids = np.mean(vertices[faces], axis=1)
    sort_indices = np.argsort(face_centroids[:, 0])
    return (
        faces[sort_indices],
        np.array(site_ids)[sort_indices],
        F_full[sort_indices]
    )

def calculate_face_areas(vertices, faces):
    face_areas = []
    import sys
    for face in faces:
        v0 = np.asarray(vertices[face[1]] - vertices[face[0]], dtype=np.float64)
        v1 = np.asarray(vertices[face[2]] - vertices[face[0]], dtype=np.float64)
        if v0.shape != (3,) or v1.shape != (3,):
            print(f"Skipping face {face} due to incorrect shape: v0={v0.shape}, v1={v1.shape}")
            sys.exit(1)
        cr = np.cross(v0, v1)
        face_areas.append(np.linalg.norm(cr) / 2)
    return np.array(face_areas, dtype=np.float64)


