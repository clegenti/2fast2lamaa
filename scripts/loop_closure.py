import polyscope as ps
import open3d as o3d
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

### Parameters for the detection
# Maximum drift ratio for matching submaps (e.g., 0.02 means 2 cm per meter)
kMaxDrift = 0.02
# Search radius atop the drift (in meters)
kNeighborsRadius = 60.0
# Pixel size for the image representation of the submaps (in meters)
kImagePixelSize = 0.5
# Cap the height values to this value (in meters)
kCapHeight = 3.0
# Aligned the map to a plane using PCA (if map is roughly planar and not gravity aligned)
kPlaneAlignment = True
# Minimum overlap ratio (in the image representation) to accept a map match
kMinOverlapRatio = 0.6
# Minimum ratio of pixels with < kHeightSimilarityThreshold height difference
kHeightSimilarityThreshold = 1.0
kMinHeightConsistencyRatio = 0.6
# Threshold for ICP convergence
kICPThreshold = 0.50
# Minimum fitness to accept an ICP result
kICPFitThreshold = 0.5
# Number of ICP iterations
kNumICPIterations = 200


### Parameters for the graph optimization
# Standard deviation of odometry position [m]
kOdomPosStd = 1.0
# Standard deviation of odometry rotation [rad]
kOdomRotStd = 0.1 * np.pi / 180.0
# Standard deviation of loop closure position [m]
kLoopPosStd = 0.2
# Standard deviation of loop closure rotation [rad]
kLoopRotStd = 0.1 * np.pi / 180.0
# Loss function scale for loop closures
kLoopLossScalePos = 2.0
kLoopLossScaleRot = 0.1 * np.pi / 180.0


### Parameters for the bundle adjustment


# Only for visualization
kColors = np.random.rand(100,3)


def performLoopClosure(path, visualize=False):
    # Read the submaps and their trajectories
    pcds, poses, trajectories, dists = readSubmaps(path)

    # Align to a plane
    if kPlaneAlignment:
        poses = alignSubmapsToPlane(pcds, poses)

    # Get image like data
    images, cam_mats = getImagesFromSubmaps(pcds, poses, pixel_size=kImagePixelSize)

    # Extract SIFT features in each image
    keypoints, descriptors = extractFeatures(images)

    # Get potential matches based on distance and drift characteristics
    potential_matches = getCandidatesToMatch(poses, dists, kNeighborsRadius, kMaxDrift)

    # Match features between images
    visual_matches, transforms = visualMatching(keypoints, descriptors, cam_mats, poses, potential_matches, images)

    # Align heights of matched submaps (and reject if not enough overlap)
    height_aligned_matches, height_aligned_transforms = heightAlignment(pcds, poses, visual_matches, transforms)

    refined_matches = height_aligned_matches
    refined_transforms = height_aligned_transforms

    ## Geometric refinement of the matches
    #refined_matches, refined_transforms = geometricRefinement(pcds, height_aligned_matches, height_aligned_transforms)


    # Visualize matches
    if visualize:
        visualizeSubmaps(pcds, poses, height_aligned_matches)

    # Run pose graph optimization
    pose_graph_poses = runPoseGraphOptimization(path, pcds, poses, refined_matches, refined_transforms)

    # Apply the optimized poses to the trajectories
    trajectory, timestamps = applyOptimizedPosesToTrajectory(trajectories, pose_graph_poses)


    # Visualize all submaps
    if visualize:
        visualizeSubmaps(pcds, pose_graph_poses, height_aligned_matches, trajectory=trajectory)

    # Save the final trajectory to file
    trajectory_file = os.path.join(path, "loop_closed_trajectory.txt")
    saveTrajectoryToFile(trajectory, timestamps, trajectory_file)



def saveTrajectoryToFile(trajectory, timestamps, filename):
    print("Saving final trajectory to file:", filename)
    with open(filename, 'w') as f:
        for i in range(trajectory.shape[0]):
            rotvec = R.from_matrix(trajectory[i][:3, :3]).as_rotvec()
            pose_str = str(timestamps[i]) + " " + str(trajectory[i][0,3]) + " " + str(trajectory[i][1,3]) + " " + str(trajectory[i][2,3]) + " " + str(rotvec[0]) + " " + str(rotvec[1]) + " " + str(rotvec[2])
            f.write(f"{pose_str}\n")



def applyOptimizedPosesToTrajectory(trajectories, optimized_poses):
    print("Applying optimized poses to the trajectories...")
    # Quick and dirty averaging directly the rotation vectors and positions
    def avgPoses(pose_A, pose_B, alpha=0.5):
        delta_pose = np.linalg.inv(pose_A) @ pose_B
        delta_rotvec = R.from_matrix(delta_pose[:3, :3]).as_rotvec()
        avg_pose = np.eye(4)
        avg_pose[:3, 3] = pose_A[:3, 3] + delta_pose[:3, 3] * alpha
        avg_pose[:3, :3] = pose_A[:3, :3] @ R.from_rotvec(delta_rotvec * alpha).as_matrix()
        return avg_pose

    t1 = time.time()
    full_trajectory = {}
    for i in range(len(trajectories)):
        temp_transforms = []
        temp_timestamps = []
        correction_low = None
        correction_high = None
        # Go through all trajectory poses by order of timestamps
        for timestamp in sorted(trajectories[i].keys()):
            trans = trajectories[i][timestamp]
            global_trans = optimized_poses[i] @ trans
            if i > 0 and timestamp in trajectories[i-1]:
                pose_A = optimized_poses[i-1] @ trajectories[i-1][timestamp]
                correction_low = (pose_A, global_trans)
                pose_avg = avgPoses(pose_A, global_trans)
            elif i < len(trajectories)-1 and timestamp in trajectories[i+1]:
                pose_A = optimized_poses[i+1] @ trajectories[i+1][timestamp]
                if correction_high is None:
                    correction_high = (global_trans, pose_A)
                pose_avg = avgPoses(global_trans, pose_A)
            else:
                temp_timestamps.append(timestamp)
                temp_transforms.append(optimized_poses[i] @ trans)
                continue
            if timestamp not in full_trajectory:
                full_trajectory[timestamp] = []
            full_trajectory[timestamp].append(pose_avg)
        distances = []
        dist = 0
        if correction_low is not None:
            dist += np.linalg.norm(correction_low[0][:3, 3] - temp_transforms[0][:3, 3])
        distances.append(dist)
        for j in range(1, len(temp_transforms)):
            dist += np.linalg.norm(temp_transforms[j][:3, 3] - temp_transforms[j-1][:3, 3])
            distances.append(dist)
        if j == len(temp_transforms) -1 and correction_high is not None:
            dist += np.linalg.norm(correction_high[0][:3, 3] - temp_transforms[-1][:3, 3])
        delta_pose_A = np.eye(4)
        if correction_low is not None:
            delta_pose_A = np.linalg.inv(correction_low[1]) @ avgPoses(correction_low[0], correction_low[1])
        delta_pose_B = np.eye(4)
        if correction_high is not None:
            delta_pose_B = np.linalg.inv(correction_high[0]) @ avgPoses(correction_high[0], correction_high[1])

        alpha_values = np.array(distances) / dist if dist > 0 else np.zeros(len(distances))
        for j in range(len(temp_transforms)):
            correction = avgPoses(delta_pose_A, delta_pose_B, alpha=alpha_values[j])
            corrected_pose = temp_transforms[j] @ correction
            if temp_timestamps[j] not in full_trajectory:
                full_trajectory[temp_timestamps[j]] = []
            full_trajectory[temp_timestamps[j]].append(corrected_pose)

    
    trajectories_output = []
    timstamps = []
    for timestamp in full_trajectory:
        poses_list = full_trajectory[timestamp]
        if len(poses_list) == 1:
            trajectories_output.append(poses_list[0])
            timstamps.append(timestamp)
        else:
            # Check if the two poses are close enough
            pos_A = poses_list[0][:3, 3]
            pos_B = poses_list[1][:3, 3]
            dist = np.linalg.norm(pos_A - pos_B)
            if dist > 0.0001:
                print("Warning: poses for timestamp", timestamp, "are too different! Dist:", dist)

            trajectories_output.append(poses_list[0])
            timstamps.append(timestamp)
    
    # Sort by timestamp
    sorted_indices = np.argsort(timstamps)
    full_trajectory = [trajectories_output[i] for i in sorted_indices]
    full_trajectory = np.array(full_trajectory)

    t2 = time.time()
    print("Applied optimized poses to trajectory in", t2 - t1, "seconds.")
    return full_trajectory, [timstamps[i] for i in sorted_indices]


def runPoseGraphOptimization(path, pcds, poses, matches, transforms):
    # Write the clouds, poses, matches, and transforms to files
    writeToFiles(path, pcds, poses, matches, transforms)

    loop_results_dir = os.path.join(path, "loop_closure_results")

    cmd = "ros2 run ffastllamaa offline_loop_closure " + \
            "-d " + loop_results_dir + " " + \
            "--odom_pos_std " + str(kOdomPosStd) + " " + \
            "--odom_rot_std " + str(kOdomRotStd) + " " + \
            "--loop_pos_std " + str(kLoopPosStd) + " " + \
            "--loop_rot_std " + str(kLoopRotStd) + " " + \
            "--loop_loss_scale_pos " + str(kLoopLossScalePos) + " " + \
            "--loop_loss_scale_rot " + str(kLoopLossScaleRot)

    os.system(cmd)

    # Read the poses in the output file
    pose_graph_data = np.loadtxt(os.path.join(loop_results_dir, "pose_graph_poses.txt"))
    optimized_poses = np.zeros((pose_graph_data.shape[0], 4,4))
    for i in range(pose_graph_data.shape[0]):
        rotvec = pose_graph_data[i, 3:6]
        rotmat = R.from_rotvec(rotvec).as_matrix()
        optimized_poses[i][:3, :3] = rotmat
        optimized_poses[i][:3, 3] = pose_graph_data[i, 0:3]
        optimized_poses[i][3, 3] = 1.0

    return optimized_poses


def writeToFiles(path, pcds, poses, matches, transforms):
    print("Writing loop closures to files...")
    t1 = time.time()
    output_dir = os.path.join(path, "loop_closure_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write point clouds
    pcds_dir = os.path.join(output_dir, "pcds")
    if not os.path.exists(pcds_dir):
        os.makedirs(pcds_dir)
    for i in range(len(pcds)):
        o3d.io.write_point_cloud(os.path.join(pcds_dir, "submap_"+ str(i) + ".ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds[i])))

    # Write poses
    poses_file = os.path.join(output_dir, "poses.txt")
    with open(poses_file, 'w') as f:
        for i in range(len(poses)):
            rotvec = R.from_matrix(poses[i][:3, :3]).as_rotvec()
            pose_str = str(poses[i][0,3]) + " " + str(poses[i][1,3]) + " " + str(poses[i][2,3]) + " " + str(rotvec[0]) + " " + str(rotvec[1]) + " " + str(rotvec[2])
            f.write(f"{pose_str}\n")

    # Write matches and transforms
    matches_file = os.path.join(output_dir, "matches_and_transforms.txt")
    with open(matches_file, 'w') as f:
        for i in range(len(matches)):
            match_str = f"{matches[i][0]} {matches[i][1]}"
            rotvec = R.from_matrix(transforms[i][:3, :3]).as_rotvec()
            match_str += " " + str(transforms[i][0,3]) + " " + str(transforms[i][1,3]) + " " + str(transforms[i][2,3]) + " " + str(rotvec[0]) + " " + str(rotvec[1]) + " " + str(rotvec[2])
            f.write(f"{match_str}\n")

    # Copy the info files
    for i in range(len(pcds)):
        src_info_file = os.path.join(path, "submap_" + str(i) + ".info")
        dst_info_file = os.path.join(pcds_dir, "submap_" + str(i) + ".info")
        if os.path.exists(src_info_file):
            os.system("cp " + src_info_file + " " + dst_info_file)


    t2 = time.time()
    print("Finished writing files in", t2 - t1, "seconds.")




def geometricRefinement(pcds, matches, transforms):
    print("Performing geometric refinement of the matches...")
    t1 = time.time()

    # Perform ICP between matched submaps to refine the transforms
    valid = [False] * len(transforms)
    refined_transforms = transforms.copy()
    # Compute the normals of all point clouds
    o3d_pcds = []
    for i in range(len(pcds)):
        print("Computing normals for ICP on submap", i, "/", len(pcds), end="\r")
        pcd = pcds[i]
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
        o3d_pcds.append(o3d_pcd)

    for i in range(len(matches)):
        print("Refining match", i, "/", len(matches), "between submaps", matches[i][0], "and", matches[i][1], end="\r")
        id_1 = matches[i][0]
        id_2 = matches[i][1]
        pcd_1 = o3d_pcds[id_1]
        pcd_2 = o3d_pcds[id_2]

        # Point to plane ICP
        icp_result = o3d.pipelines.registration.registration_icp(pcd_2, pcd_1, max_correspondence_distance=kICPThreshold, init=transforms[i], estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=kNumICPIterations))
        fitness = icp_result.fitness
        trans = icp_result.transformation

        # Visualize for debug
        #print("Refined fitness:", fitness)
        #print("Diff transform:\n", np.linalg.inv(transforms[i]) @ trans)
        #visualizeSubmaps([pcds[id_1], pcds[id_2]], [np.eye(4), transforms[i]], [(0,1)])

        if fitness > kICPFitThreshold:
            refined_transforms[i] = trans
            valid[i] = True


    # Keep only valid transforms
    refined_transforms = [t for i, t in enumerate(refined_transforms) if valid[i]]
    refined_matches = [m for i, m in enumerate(matches) if valid[i]]

    t2 = time.time()
    print("Refined matches from", len(matches), "to", len(refined_matches), "in", t2 - t1, "seconds")

    return refined_matches, refined_transforms


def heightAlignment(pcds, poses, matches, transforms):
    print("Aligning submaps heights...")
    t0 = time.time()
    new_trans = transforms.copy()
    valid = [False] * len(transforms)
    for i in range(len(matches)):
        print("Processing match", i, "/", len(matches), "between submaps", matches[i][0], "and", matches[i][1], end="\r")
        # Get the height of the point clouds
        id_1 = matches[i][0]
        id_2 = matches[i][1]
        pcd_1 = pcds[id_1]
        pcd_2 = (transforms[i][:3,:3] @ pcds[id_2].T).T + transforms[i][:3,3]

        pcd_1 = (poses[id_1][:3,:3] @ pcd_1.T).T + poses[id_1][:3,3]
        pcd_2 = (poses[id_1][:3,:3] @ pcd_2.T).T + poses[id_1][:3,3]

        # Get two point clouds heights in image-like representation
        min_xy = np.min(np.vstack((pcd_1[:, :2], pcd_2[:, :2])), axis=0)
        max_xy = np.max(np.vstack((pcd_1[:, :2], pcd_2[:, :2])), axis=0)
        img_res = 2*kImagePixelSize
        img_size = np.ceil((max_xy - min_xy) / img_res).astype(int) + 1
        img_1 = np.zeros((img_size[0], img_size[1]))
        img_1_counter = np.zeros((img_size[0], img_size[1]))
        img_2 = np.zeros((img_size[0], img_size[1]))
        img_2_counter = np.zeros((img_size[0], img_size[1]))
        pts_img_1 = np.floor((pcd_1[:, :2] - min_xy) / img_res).astype(int)
        pts_img_2 = np.floor((pcd_2[:, :2] - min_xy) / img_res).astype(int)
        # Fill the image with the points mean z value using vectorized operations
        np.add.at(img_1, (pts_img_1[:, 0], pts_img_1[:, 1]), pcd_1[:, 2])
        np.add.at(img_1_counter, (pts_img_1[:, 0], pts_img_1[:, 1]), 1)
        np.add.at(img_2, (pts_img_2[:, 0], pts_img_2[:, 1]), pcd_2[:, 2])
        np.add.at(img_2_counter, (pts_img_2[:, 0], pts_img_2[:, 1]), 1)

        # Get the mean height
        valid_mask_1 = img_1_counter > 0
        img_1[valid_mask_1] = img_1[valid_mask_1] / img_1_counter[valid_mask_1]
        valid_mask_2 = img_2_counter > 0
        img_2[valid_mask_2] = img_2[valid_mask_2] / img_2_counter[valid_mask_2]

        # Find overlapping area
        overlap_mask = valid_mask_1 & valid_mask_2
        overlap_ratio_A = np.sum(img_1_counter[overlap_mask]) / pts_img_1.shape[0]
        overlap_ratio_B = np.sum(img_2_counter[overlap_mask]) / pts_img_2.shape[0]

        if overlap_ratio_A < kMinOverlapRatio or overlap_ratio_B < kMinOverlapRatio:
            continue

        # Compute height difference in overlapping area
        height_diff = img_2[overlap_mask] - img_1[overlap_mask]
        mean_height_diff = np.mean(height_diff)

        # Check height consistency
        height_consistent_mask = np.abs(height_diff - mean_height_diff) < kHeightSimilarityThreshold
        height_consistency_ratio = np.sum(height_consistent_mask) / height_diff.shape[0]
        if height_consistency_ratio < kMinHeightConsistencyRatio:
            continue


        # Height transform
        height_transform = np.eye(4)
        height_transform[2, 3] = -mean_height_diff
        new_trans[i] = np.linalg.inv(poses[id_1]) @ height_transform @ poses[id_1] @ transforms[i]
        valid[i] = True



        # Plot for debug
        #print("Mean height diff for match", i, "between submaps", id_1, "and", id_2, ":", mean_height_diff)
        #visualizeSubmaps([pcds[id_1], pcds[id_2]], [np.eye(4), new_trans[i]], [(0,1)])

    # Keep only valid transforms
    new_trans = [t for i, t in enumerate(new_trans) if valid[i]]
    new_match = [m for i, m in enumerate(matches) if valid[i]]

    t1 = time.time()
    print("Height alignment took", t1 - t0, "seconds.")
    return new_match, new_trans


def visualMatching(keypoints, descriptors, cam_mats, poses, potential_matches, images=None):
    print("Performing visual matching between candidate submap pairs...")
    t0 = time.time()
    bf = cv2.BFMatcher()
    verified_matches = []
    transforms = []
    for (i, j) in potential_matches:
        print("Processing submap pair:", i, j, end="\r")
        des1 = descriptors[i]
        des2 = descriptors[j]
        if des1 is None or des2 is None:
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        # Remove the features that have a different scale
        for m in matches:
            kp1 = keypoints[i][m[0].queryIdx]
            kp2 = keypoints[j][m[0].trainIdx]
            scale_ratio = kp1.size / kp2.size
            if np.abs(scale_ratio - 1.0) < 0.05:
                good_matches.append(m)
        if len(good_matches) < 10:
            continue

        # Apply ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) < 10:
            continue

        # Perform rigid transformation estimation with RANSAC
        dst_pts = np.float32([ keypoints[i][m.queryIdx].pt for m in good_matches ]).reshape(-1,2)
        src_pts = np.float32([ keypoints[j][m.trainIdx].pt for m in good_matches ]).reshape(-1,2)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0)

        if M is not None:
            trans, scale = affineToPoseAndScale(M, cam_mats[i], cam_mats[j], kImagePixelSize)
            if np.abs(scale - 1.0) > 0.1:
                continue
            verified_matches.append((i, j))
            transforms.append(np.linalg.inv(poses[i]) @ trans @ poses[j])

        # Debug: visualize matches and registration
        if len(good_matches) > 0 and images is not None:
            img1 = cv2.cvtColor((images[i] / (2*kCapHeight) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor((images[j] / (2*kCapHeight) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img_matches = cv2.drawMatches(img1, keypoints[i], img2, keypoints[j], good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # Create image overlaying with transparency the two submaps (add color)
            #img1[:, :, 0] = img1[:, :, 0]
            img1[:, :, 1:3] = 0
            #img2[:, :, 1] = img2[:, :, 1] * 0.5 + 0 * 0.5
            img2[:, :, :2] = 0
            img2_registered = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))
            image_2_registered = cv2.warpAffine(images[j], M, (img1.shape[1], img1.shape[0]))
            img2_registered[image_2_registered == kCapHeight, :] = 255
            img1[images[i] == kCapHeight, :] = 255
            overlay = cv2.addWeighted(img1, 0.5, img2_registered, 0.5, 0)

            fig, ax = plt.subplots(1,2, figsize=(15,7))
            ax[0].imshow(img_matches)
            ax[1].imshow(overlay)
            plt.show()

    t1 = time.time()
    print("Verified", len(verified_matches), "matches in", np.round(t1 - t0, 2), "seconds.")
    return verified_matches, transforms



def affineToPoseAndScale(affine_matrix, cam_mat_source, cam_mat_target, pix_res):
    T_cv_local_map = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, 1]])
    T_local_map_cv = np.linalg.inv(T_cv_local_map)

    # Get the scale from the affine matrix
    scale = np.linalg.norm(affine_matrix[0, :2])
    # Get the rotation from the affine matrix
    rotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
    pose = np.eye(3)
    pose[0, 0] = np.cos(rotation)
    pose[0, 1] = -np.sin(rotation)
    pose[1, 0] = np.sin(rotation)
    pose[1, 1] = np.cos(rotation)
    pose[0, 2] = affine_matrix[0, 2]
    pose[1, 2] = affine_matrix[1, 2]

    # Convert the pose to the local map frame
    pose = np.linalg.inv(cam_mat_source) @ T_local_map_cv @ pose @ T_cv_local_map @ cam_mat_target

    se3_pose = np.eye(4)
    se3_pose[:2, :2] = pose[:2, :2]
    se3_pose[0, 3] = pose[0, 2]
    se3_pose[1, 3] = pose[1, 2]

    return se3_pose, scale


def getCandidatesToMatch(poses, dists, radius, max_drift):
    print("Getting candidate matches between submaps based on distance...")
    t0 = time.time()
    potential_matches = []
    num_submaps = len(poses)
    for i in range(num_submaps):
        for j in range(i+2, num_submaps):
            # Compute the maximum distance for matching
            max_dist = dists[j][1] - dists[i][0]
            max_dist = max_dist * max_drift + radius
            
            # Check distance between centroids
            dist_centroids = np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])
            if dist_centroids < radius:
                potential_matches.append((i, j))
            
    t1 = time.time()
    print("Found", len(potential_matches), "candidate matches in", np.round(t1 - t0, 2), "seconds.")
    return potential_matches



def extractFeatures(images):
    print("Extracting SIFT features from images...")
    t0 = time.time()
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for i in range(len(images)):
        print("Processing image", i,"/", len(images), end="\r")
        img_uint8 = (images[i] / np.max(images[i]) * 255).astype(np.uint8)
        kp, des = sift.detectAndCompute(img_uint8, None)
        keypoints.append(kp)
        descriptors.append(des)

    t1 = time.time()
    print("Extracted SIFT features in", np.round(t1 - t0, 2), "seconds.")
    return keypoints, descriptors



def getImagesFromSubmaps(pcds, poses, pixel_size=0.2):
    print("Getting images from submaps...")
    t0 = time.time()
    images = []
    cam_mats = []
    for i in range(len(pcds)):
        print("Processing submap", i,"/", len(pcds), end="\r")
        pts = (poses[i][:3, :3] @ pcds[i].T).T + poses[i][:3, 3]
        pts[:, 2] -= poses[i][2, 3]  # Set ground to z=0


        # Get the xy bounds
        min_xy = np.min(pts[:, :2], axis=0)
        max_xy = np.max(pts[:, :2], axis=0)
        size_xy = max_xy - min_xy
        img_size = np.ceil(size_xy / pixel_size).astype(int) + 1
        img = np.zeros((img_size[0], img_size[1]))
        counter = np.zeros((img_size[0], img_size[1]))

        cam_mat = np.array([[1.0/pixel_size, 0, -min_xy[0]/pixel_size],
                            [0, 1.0/pixel_size, -min_xy[1]/pixel_size],
                            [0, 0, 1]])

        pts_img = (cam_mat @ np.hstack((pts[:,:2], np.ones((pts.shape[0], 1)))).T).T[:, :2].astype(int)
        # Transform points to image coordinates and fill the image to get the mean z value
        np.add.at(img, (pts_img[:, 0], pts_img[:, 1]), pts[:, 2])
        np.add.at(counter, (pts_img[:, 0], pts_img[:, 1]), 1)
        counter[counter == 0] = 1
        img = img / counter

        # Cap the height values
        img = np.clip(img, a_min=-kCapHeight, a_max=kCapHeight)
        img += kCapHeight

        # Stack for the output
        images.append(img)
        cam_mats.append(cam_mat)

    t1 = time.time()
    print("Extracted images from submaps in", np.round(t1 - t0, 2), "seconds.")
    return images, cam_mats


# Fit a plane (PCA) to the map and align to it
def alignSubmapsToPlane(pcds, poses):
    world_pcds = putSubmapsInWorld(pcds, poses)
    all_pts = np.vstack(world_pcds)
    centroid = np.mean(all_pts, axis=0)
    centered_pts = all_pts - centroid
    cov = np.cov(centered_pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    # The normal is the eigenvector with smallest eigenvalue
    normal = eigvecs[:, np.argmin(eigvals)]
    # Compute rotation to align normal to z axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    s = np.linalg.norm(v)
    if s < 1e-6:
        R_align = np.eye(3)
    else:
        v_skew = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
        R_align = np.eye(3) + v_skew + v_skew @ v_skew * ((1 - c) / (s ** 2))
    # Apply rotation to all poses
    for i in range(len(poses)):
        poses[i][:3, :3] = R_align @ poses[i][:3, :3]
        poses[i][:3, 3] = R_align @ (poses[i][:3, 3] - centroid) + centroid

    # For each submap count the number of points below posistion z=0
    under = 0
    total = 0
    for i in range(len(pcds)):
        pts = (poses[i][:3, :3] @ pcds[i].T).T
        under += np.sum(pts[:, 2] < 0)
        total += pts.shape[0]

    if under / total < 0.5:
        print("Flipping map to have ground below z=0")
        R_flip = R.from_euler('x', 180, degrees=True).as_matrix()
        for i in range(len(poses)):
            poses[i][:3, :3] = R_flip @ poses[i][:3, :3]
            poses[i][:3, 3] = R_flip @ poses[i][:3, 3]


    return poses

# Read the submaps and their trajectories
def readSubmaps(path):
    # Get the submap files in the raw_output folder
    submap_files = [f for f in os.listdir(path) if f.startswith('submap_') and f.endswith('.ply')]
    traje_files = [f for f in os.listdir(path) if f.startswith('trajectory_submap_') and f.endswith('.csv')]
    # Sort so that the number order is respected
    submap_files = sorted(submap_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    traj_files = sorted(traje_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    # Check that the number of submaps and trajectory files are the same
    if len(submap_files) != len(traje_files):
        print("Error: number of submap files and trajectory files do not match!")
        return [], [], []


    # Read the point clouds and trajectories
    pcds = []
    poses = []
    trajectories = []
    distance_traveled = {}
    last_dist = 0.0
    min_max_dist = []
    for i in range(len(submap_files)):
        # Read files
        pcd = o3d.io.read_point_cloud(os.path.join(path, submap_files[i]))
        traj = np.loadtxt(os.path.join(path, traj_files[i]), delimiter=',', skiprows=1)
        temp_poses = np.zeros((traj.shape[0], 4, 4))
        for j in range(traj.shape[0]):
            temp_poses[j, :3, :3] = R.from_rotvec(traj[j, 4:7]).as_matrix()
            temp_poses[j, :3, 3] = traj[j, 1:4]
            temp_poses[j, 3, 3] = 1.0
            if len(distance_traveled) == 0:
                distance_traveled[traj[j,0]] = 0.0
            elif traj[j,0] not in distance_traveled:
                dist = np.linalg.norm(traj[j, 1:4] - traj[j-1, 1:4])
                distance_traveled[traj[j,0]] = last_dist + dist
                last_dist = distance_traveled[traj[j,0]]
        min_max_dist.append((distance_traveled[traj[0,0]], distance_traveled[traj[-1,0]]))

        # Get the pose closest to the middle of the trajectory
        dists = np.linalg.norm(traj[1:, 1:4] - traj[:-1, 1:4], axis=1)
        cum_dists = np.cumsum(dists)
        half_dist = cum_dists[-1] / 2.0
        mid_idx = np.searchsorted(cum_dists, half_dist)
        poses.append(temp_poses[mid_idx])

        # Express the trajectories in the submap frame
        traj_in_submap = {}
        inv_mid_pose = np.linalg.inv(temp_poses[mid_idx])
        for j in range(traj.shape[0]):
            trans = inv_mid_pose @ temp_poses[j]
            traj_in_submap[traj[j,0]] = trans
        trajectories.append(traj_in_submap)

        # Project points in the submap to the pose
        pts = np.asarray(pcd.points)
        trans = np.linalg.inv(poses[-1])
        pts = (trans[:3, :3] @ pts.T).T + trans[:3, 3]

        pcds.append(pts)

    return pcds, poses, trajectories, min_max_dist



# Put submaps in world frame
def putSubmapsInWorld(pcds, poses):
    world_pcds = []
    for i in range(len(pcds)):
        pts = pcds[i]
        trans = poses[i]
        pts = (trans[:3, :3] @ pts.T).T + trans[:3, 3]
        world_pcds.append(pts)
    return world_pcds


# Get a fixed color for each submap index
def getColorsForSubmaps(num_submap):
    return kColors[num_submap % kColors.shape[0], :]


# Visualize the submaps in world frame
def visualizeSubmaps(pcds, poses, potential_matches = None, radius_pts=0.1, radius_path=1.0, radius_centroids=4.0, trajectory=None):
    # Put submaps in world frame
    world_pcds = putSubmapsInWorld(pcds, poses)

    # Visualize the point clouds and their centroids
    ps.init()
    for i, pcd in enumerate(world_pcds):
        ps_cloud = ps.register_point_cloud(f"submap_{i}", pcd, transparency=0.7)
        ps_cloud.set_radius(radius_pts, relative=False)
        ps_cloud.set_point_render_mode("quad")
        ps_cloud.add_color_quantity("submap_index", np.ones((len(pcd),3))*getColorsForSubmaps(i), enabled=True)
    centroids = np.array([pose[:3, 3] for pose in poses])
    ps_network = ps.register_curve_network("submap_centroids", centroids, edges=np.array([[i, i+1] for i in range(centroids.shape[0]-1)]))
    ps_network.set_color((1.0, 0.0, 0.0))
    ps_network.set_radius(radius_path, relative=False)
    ps_centroids = ps.register_point_cloud("centroids", centroids)
    ps_centroids.set_color((1.0, 0.0, 0.0))
    ps_centroids.set_radius(radius_centroids, relative=False)
    # If potential matches are given, visualize them
    if potential_matches is not None:
        match_edges = np.array(potential_matches)
        ps_matches = ps.register_curve_network("potential_loop_closures", centroids, edges=match_edges)
        ps_matches.set_color((0.0, 1.0, 0.0))
        ps_matches.set_radius(radius_path, relative=False)

    # If trajectory is given, visualize it
    if trajectory is not None:
        traj_points = np.array([pose[:3, 3] for pose in trajectory])
        ps_trajectory = ps.register_curve_network("Merged trajectory", traj_points, edges=np.array([[i, i+1] for i in range(traj_points.shape[0]-1)]))
        ps_trajectory.set_color((0.0, 0.0, 1.0))
        ps_trajectory.set_radius(radius_path, relative=False)

    ps.set_up_dir("neg_z_up")
    ps.set_ground_plane_mode("none")
    ps.show()


















if __name__ == "__main__":
    # Get the path from command line arguments

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "vbr_output/diag_train0/raw_output"

    performLoopClosure(path, visualize=True)