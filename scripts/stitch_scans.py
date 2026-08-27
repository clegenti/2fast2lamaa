#!/usr/bin/env python3
# Stitch the scans saved by the gp_map node into a single point cloud, using the trajectory optimized
# by the pose_graph node.
#
#     python3 stitch_scans.py <map_path> [options]
#
# The scans are the ones written when the `write_scans` parameter of the gp_map node is enabled. They
# live in `<map_path>/scans/<timestamp>.ply` and are expressed in the frame of the sensor (the IMU) at
# their own timestamp. Each one is brought into the map frame with the pose of the same timestamp in
# `<map_path>/pose_graph_trajectory.csv`, so the result reflects the loop closures the pose graph
# found, unlike the `map.ply` written while mapping.
#
# The whole stitched cloud is held in memory before being written, so the `--min-dist` and
# `--min-rot-deg` filters are the way to keep a long sequence manageable.

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

# Both are relative to the map path given on the command line
kScanFolder = "scans"
kDefaultTrajectory = "pose_graph_trajectory.csv"

kDefaultOutputName = "map.ply"

# Report the progress every that many scans
kProgressPeriod = 25


# Read the vertex properties of a binary or ascii ply as a dict of arrays. The properties are
# collected per element so that a `face` element does not interfere with the vertex layout.
def readPlyVertices(ply_path):
    with open(ply_path, "rb") as file:
        header = b""
        while b"end_header" not in header:
            line = file.readline()
            if not line:
                raise ValueError("Truncated ply header in " + ply_path)
            header += line

        fmt = "ascii"
        elements = []
        for line in header.split(b"\n"):
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0] == b"format":
                fmt = tokens[1].decode()
            elif len(tokens) >= 3 and tokens[0] == b"element":
                elements.append({'name': tokens[1].decode(), 'count': int(tokens[2]), 'properties': []})
            elif len(tokens) >= 3 and tokens[0] == b"property" and len(elements) > 0:
                elements[-1]['properties'].append((tokens[1].decode(), tokens[-1].decode()))

        vertex = next((e for e in elements if e['name'] == 'vertex'), None)
        if vertex is None:
            raise ValueError("No vertex element in " + ply_path)
        if vertex['count'] == 0:
            return {name: np.zeros(0) for _, name in vertex['properties']}

        numpy_types = {"double": "f8", "float": "f4", "float32": "f4", "float64": "f8",
                       "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
                       "short": "i2", "ushort": "u2", "uchar": "u1", "char": "i1",
                       "uint8": "u1", "int8": "i1"}
        names = [name for _, name in vertex['properties']]
        if fmt == "ascii":
            values = np.loadtxt(file, max_rows=vertex['count'], ndmin=2)
            return {name: values[:, i].astype(np.float64) for i, name in enumerate(names)}

        dtype = np.dtype([(name, numpy_types[ply_type])
                          for (ply_type, _), name in zip(vertex['properties'], names)])
        data = np.frombuffer(file.read(dtype.itemsize * vertex['count']),
                             dtype=dtype, count=vertex['count'])
        return {name: data[name].astype(np.float64) for name in names}


# Nx3 positions of a scan ply
def readScan(ply_path):
    properties = readPlyVertices(ply_path)
    missing = [name for name in ['x', 'y', 'z'] if name not in properties]
    if len(missing) > 0:
        raise ValueError("Missing propertie(s) " + str(missing) + " in " + ply_path)
    return np.stack([properties['x'], properties['y'], properties['z']], axis=1)


# Write a binary ply with the same layout as the ones the package produces, so that it can be read
# back by the gp_map node and by the usual viewers
def writePly(ply_path, points):
    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              "comment Stitched with stitch_scans.py\n"
              "element vertex " + str(len(points)) + "\n"
              "property double x\n"
              "property double y\n"
              "property double z\n"
              "end_header\n")
    with open(ply_path, "wb") as file:
        file.write(header.encode("ascii"))
        np.ascontiguousarray(points, dtype=np.float64).tofile(file)


# Read a trajectory written by the pose_graph node (`pose_graph_trajectory.csv`) or by the gp_map node
# (`trajectory.csv`). Both hold `timestamp, x, y, z` followed by a rotation vector, and the timestamps
# are nanoseconds, parsed as integers as they do not fit exactly in a float64.
def readTrajectory(csv_path):
    times = []
    positions = []
    rotation_vectors = []
    with open(csv_path, 'r') as file:
        for line in file:
            fields = [f.strip() for f in line.split(',')]
            if (len(fields) < 7) or line.startswith('#') or fields[0].startswith('timestamp'):
                continue
            times.append(int(fields[0]))
            positions.append([float(f) for f in fields[1:4]])
            rotation_vectors.append([float(f) for f in fields[4:7]])

    if len(times) == 0:
        raise ValueError("No pose found in " + csv_path)
    return np.array(times, dtype=np.int64), np.array(positions), R.from_rotvec(np.array(rotation_vectors))


# Indexes of the poses to keep. A threshold of 0 disables its own criterion, so the default (0, 0)
# keeps every scan. When both are set, a scan is kept as soon as one of the two is met, which is the
# criterion the gp_map node uses for its key framing.
def selectPoses(positions, rotations, min_dist, min_rot_deg):
    use_dist = min_dist > 0.0
    use_rot = min_rot_deg > 0.0
    if (not use_dist) and (not use_rot):
        return np.arange(len(positions))

    kept = [0]
    last = 0
    for i in range(1, len(positions)):
        dist = float(np.linalg.norm(positions[i] - positions[last]))
        rot = np.degrees(np.linalg.norm((rotations[i]*rotations[last].inv()).as_rotvec()))
        if (use_dist and (dist >= min_dist)) or (use_rot and (rot >= min_rot_deg)):
            kept.append(i)
            last = i
    return np.array(kept)


def stitchScans(map_path, trajectory_path, scan_folder, min_dist, min_rot_deg):
    times, positions, rotations = readTrajectory(trajectory_path)
    print(f"Read {len(times)} poses from {trajectory_path}")

    kept = selectPoses(positions, rotations, min_dist, min_rot_deg)
    if (min_dist > 0.0) or (min_rot_deg > 0.0):
        print(f"Kept {len(kept)} of the {len(times)} poses "
              f"(at least {min_dist} m or {min_rot_deg} deg apart)")

    clouds = []
    nb_points = 0
    nb_used = 0
    nb_missing = 0
    nb_non_finite = 0
    for counter, index in enumerate(kept):
        scan_path = os.path.join(scan_folder, str(times[index]) + ".ply")
        if not os.path.exists(scan_path):
            nb_missing += 1
            continue

        points = readScan(scan_path)
        finite = np.isfinite(points).all(axis=1)
        nb_non_finite += int(np.sum(~finite))
        points = points[finite]
        if len(points) == 0:
            continue

        # The scans are expressed in the frame of the sensor at their own timestamp
        clouds.append(points @ rotations[index].as_matrix().T + positions[index])
        nb_points += len(points)
        nb_used += 1

        if (counter % kProgressPeriod == 0) or (counter == len(kept) - 1):
            print(f"  {counter + 1}/{len(kept)} poses processed, {nb_used} scans, "
                  f"{nb_points} points ({nb_points*24/1e6:.0f} MB)")

    if nb_missing > 0:
        print(f"{nb_missing} of the kept poses have no scan in {scan_folder} "
              f"(expected when the scans were written with the key framing enabled)")
    if nb_non_finite > 0:
        print(f"Dropped {nb_non_finite} points with non-finite coordinates")
    if nb_used == 0:
        raise FileNotFoundError("None of the poses of the trajectory has a scan in " + scan_folder)

    print(f"Stitched {nb_used} scans into {nb_points} points")
    return np.concatenate(clouds, axis=0)


def main(map_path, output_path, trajectory, min_dist, min_rot_deg):
    if not os.path.isdir(map_path):
        raise FileNotFoundError("Map path not found: " + map_path)

    trajectory_path = trajectory if trajectory is not None else os.path.join(map_path, kDefaultTrajectory)
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError("Trajectory not found: " + trajectory_path
                                + "\nThe pose_graph node writes it in the map path when it optimizes"
                                + " the graph. Use --trajectory to point at another one, for instance"
                                + " the trajectory.csv of the gp_map node.")

    scan_folder = os.path.join(map_path, kScanFolder)
    if not os.path.isdir(scan_folder):
        raise FileNotFoundError("Scan folder not found: " + scan_folder
                                + "\nThe scans are only written when the `write_scans` parameter of"
                                + " the gp_map node is enabled.")

    points = stitchScans(map_path, trajectory_path, scan_folder, min_dist, min_rot_deg)

    if output_path is None:
        output_path = os.path.join(map_path, kDefaultOutputName)
    writePly(output_path, points)
    print(f"Wrote {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")


def parseArgs():
    parser = argparse.ArgumentParser(
            description="Stitch the scans written by the gp_map node into a single point cloud, "
                        "placing each one with the pose the pose_graph node optimized for it.")
    parser.add_argument("map_path",
            help="folder holding scans/ and " + kDefaultTrajectory + " (the `map_path` of the gp_map node)")
    parser.add_argument("-o", "--output", default=None,
            help="output ply (default: <map_path>/" + kDefaultOutputName + ")")
    parser.add_argument("--trajectory", default=None,
            help="trajectory to use instead of <map_path>/" + kDefaultTrajectory
                 + " (the trajectory.csv of the gp_map node has the same format)")
    parser.add_argument("--min-dist", type=float, default=0.0,
            help="only use a scan if its pose is at least that many meters away from the last used "
                 "one, 0 disables this criterion (default: %(default)s)")
    parser.add_argument("--min-rot-deg", type=float, default=0.0,
            help="only use a scan if its pose is at least that many degrees away from the last used "
                 "one, 0 disables this criterion (default: %(default)s). A scan is used as soon as "
                 "one of the two criteria is met, and every scan is used when both are 0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    try:
        main(args.map_path, args.output, args.trajectory, args.min_dist, args.min_rot_deg)
    except (FileNotFoundError, ValueError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
