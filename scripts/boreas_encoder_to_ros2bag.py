import os
import argparse
import rosbag2_py
import pandas as pd
import numpy as np
from sensor_msgs.msg import JointState
from rclpy.serialization import serialize_message

kDefaultSequencePath = "boreas-2024-12-03-12-54"

# Encoder data is kept from 1 second before the first lidar scan to 1 second after the last one
kTimeMargin = 1000000000

def main():
    args = parseArgs()
    processBag(args.sequence_path)

def parseArgs():
    parser = argparse.ArgumentParser(
            description="Convert the wheel encoder data (applanix/dmi.csv) of a boreas sequence into a ROS2 bag. "
                        "The bag is written next to the data, in <sequence_path>/encoder_ros2bag.")
    parser.add_argument("sequence_path", nargs="?", default=kDefaultSequencePath,
            help="path to the boreas sequence folder, it must contain applanix/dmi.csv (default: %(default)s)")
    return parser.parse_args()

def processBag(path):

    # Create the bag writer
    bag_writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=os.path.join(path, "encoder_ros2bag"), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions()
    bag_writer.open(storage_options, converter_options)
    # Create the topics in the bag writer
    bag_writer.create_topic(rosbag2_py.TopicMetadata(id=0, name="/wheel_encoder", type="sensor_msgs/msg/JointState", serialization_format="cdr"))

    first_lidar_time, last_lidar_time = getLidarTimeSpan(path)
    writeEncoderToBag(bag_writer, path, first_lidar_time - kTimeMargin, last_lidar_time + kTimeMargin)

    # Close the bag writer
    bag_writer.close()

# Get the timestamps (in nanoseconds) of the first and the last lidar scan of a sequence.
# The lidar ground truth file is preferred as it directly holds the scan timestamps, but it is not
# available for every sequence, in which case the lidar file names are used instead.
def getLidarTimeSpan(path):
    gt_file = os.path.join(path, "applanix", "lidar_poses.csv")
    if os.path.exists(gt_file):
        print("Reading the lidar timestamps from applanix/lidar_poses.csv")
        gt_df = pd.read_csv(gt_file, sep=",", dtype={"GPSTime": str})
        gt_time = np.array([gpsTimeMicroSecToNanoSec(t) for t in gt_df["GPSTime"]], dtype=np.int64)
        return gt_time.min(), gt_time.max()

    lidar_path = os.path.join(path, "lidar")
    lidar_files = []
    if os.path.isdir(lidar_path):
        lidar_files = [f for f in os.listdir(lidar_path) if f.endswith(".bin")]
    if len(lidar_files) == 0:
        raise FileNotFoundError("Could not find any lidar timestamp for " + path
                + " (neither applanix/lidar_poses.csv nor any lidar/*.bin file)")

    print("Reading the lidar timestamps from the lidar file names")
    lidar_time = np.array([lidarFileNameToNanoSec(f) for f in lidar_files], dtype=np.int64)
    return lidar_time.min(), lidar_time.max()

# Write the wheel encoder data of applanix/dmi.csv to an already opened bag writer (the writer must
# provide a "/wheel_encoder" topic of type sensor_msgs/msg/JointState). The readings outside of
# [`start_time`, `end_time`] (in nanoseconds) are discarded if these are given.
def writeEncoderToBag(bag_writer, path, start_time=None, end_time=None):

    print("Loading wheel encoder data from applanix/dmi.csv")
    # The GPSTime column holds GPS seconds with a sub-microsecond resolution, which does not fit in
    # a float64. It is read as a string and converted to integer nanoseconds to not lose precision.
    dmi_df = pd.read_csv(os.path.join(path, "applanix", "dmi.csv"), sep=",", dtype={"GPSTime": str})
    dmi_time = np.array([gpsTimeToNanoSec(t) for t in dmi_df["GPSTime"]], dtype=np.int64)
    dmi_count = np.array(dmi_df["pulse_count"], dtype=np.int64)

    dmi_period = np.median(np.diff(dmi_time))
    print("----Encoder period: ", dmi_period)

    mask = np.ones(len(dmi_time), dtype=bool)
    if start_time is not None:
        mask = np.logical_and(mask, dmi_time >= start_time)
    if end_time is not None:
        mask = np.logical_and(mask, dmi_time <= end_time)
    dmi_time = dmi_time[mask]
    dmi_count = dmi_count[mask]
    if len(dmi_time) < 1:
        raise ValueError("No encoder reading left in the requested time window")

    # Write the encoder data to the bag. The raw (wrapped) count is written as is, the
    # wheel_gyro_odometry node unwraps it with its `encoder_wrap_max` parameter.
    for i in range(len(dmi_time)):
        # Create the message
        msg = JointState()
        msg.header.stamp.sec = int(dmi_time[i] // 1000000000)
        msg.header.stamp.nanosec = int(dmi_time[i] % 1000000000)
        msg.header.frame_id = "wheel_encoder"
        msg.name = ["wheel_encoder"]
        msg.position = [float(dmi_count[i])]
        # Write the message to the bag
        bag_writer.write("/wheel_encoder", serialize_message(msg), dmi_time[i])

        if (i % 1000) == 0:
            print("Writing encoder data to bag: ", i, " / ", len(dmi_time), end="           \r")

    print("Writing encoder data to bag: ", len(dmi_time), " / ", len(dmi_time))

# Convert a "seconds.fraction" GPS timestamp string (as in applanix/dmi.csv) to integer nanoseconds
def gpsTimeToNanoSec(time_str):
    seconds, _, fraction = str(time_str).strip().partition(".")
    return int(seconds) * 1000000000 + int((fraction + "000000000")[:9])

# Convert a GPS timestamp string given in microseconds (as in the applanix pose files) to integer
# nanoseconds
def gpsTimeMicroSecToNanoSec(time_str):
    micro_seconds, _, fraction = str(time_str).strip().partition(".")
    return int(micro_seconds) * 1000 + int((fraction + "000")[:3])

# Convert the name of a lidar scan file to a timestamp in integer nanoseconds (depending on the
# sequence, the file names are either in microseconds or already in nanoseconds)
def lidarFileNameToNanoSec(file_name):
    time_ns = int(file_name.split(".")[0])
    if time_ns > 1e16:
        return time_ns
    return time_ns * 1000


if __name__ == "__main__":
    main()
