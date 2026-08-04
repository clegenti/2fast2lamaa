import os
import rosbag2_py
import pandas as pd
import numpy as np
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs.msg import JointState
from rclpy.serialization import serialize_message
from rclpy.serialization import deserialize_message
from boreas_encoder_to_ros2bag import writeEncoderToBag, getLidarTimeSpan, kTimeMargin

def main():
    path = "boreas-2024-12-03-12-54"

    year = path.split("-")[1]
    if int(year) < 2024:
        imu = 'applanix' # 'applanix' (the applanix imu is only for legacy on the original boreas dataset)
    else:
        imu = 'dmu' # 'dmu' imu for boreas datasets from 2024 and later
    processBags(path, imu)

def processBags(path, imu):
    
    # Create the bag writer
    bag_writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=os.path.join(path, "lidar_imu_ros2bag"), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions()
    bag_writer.open(storage_options, converter_options)
    # Create the topics in the bag writer
    bag_writer.create_topic(rosbag2_py.TopicMetadata(id=0, name="/velodyne_points", type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"))
    bag_writer.create_topic(rosbag2_py.TopicMetadata(id=1, name="/imu/data", type="sensor_msgs/msg/Imu", serialization_format="cdr"))
    bag_writer.create_topic(rosbag2_py.TopicMetadata(id=2, name="/wheel_encoder", type="sensor_msgs/msg/JointState", serialization_format="cdr"))


    # Load the IMU data from the applanix/imu.txt file
    print("Loading IMU data from imu/dmu_imu.csv")
    if imu == 'dmu':
        imu_df = pd.read_csv(os.path.join(path, "imu", "dmu_imu.csv"), sep=",")
        # Keep only the columns we need
        imu_df = imu_df[["time", 'wx', 'wy', 'wz', 'ax', 'ay', 'az']]
    elif imu == 'applanix':
        imu_df = pd.read_csv(os.path.join(path, "applanix", "imu_raw.csv"), sep=",")
        # Keep only the columns we need
        imu_df = imu_df[["GPSTime", 'angvel_x', 'angvel_y', 'angvel_z', 'accelx', 'accely', 'accelz']]
        # Rename the columns to match the dmu format
        imu_df.columns = ["time", 'wx', 'wy', 'wz', 'ax', 'ay', 'az']
        # Convert the time to nanoseconds with the right date format to not lose precision
        imu_df["time"] = (imu_df["time"].astype(np.float128) * 1e9).astype(np.int64)

    # Get the imu period
    imu_period = imu_df["time"].diff().median()
    print("----IMU period: ", imu_period)
    
    imu_time = []
    imu_data = []

    print("----Interpolating IMU data")
    for i in range(len(imu_df)-1):

        time = imu_df["time"][i]

        imu_time.append(time)
        imu_data.append(np.array(imu_df.iloc[i, -6:]))
        
        # Interpolate the IMU data if the time difference is greater than 1.5 times the imu_period
        if imu_df["time"][i+1] - time > 1.5*imu_period:
            # Get the time difference
            time_diff = imu_df["time"][i+1] - time

            # Get the number of samples to interpolate
            num_samples = np.round(time_diff / imu_period)
            # Get the time vector
            time_vector = np.linspace(time, imu_df["time"][i+1]-imu_period, int(num_samples))
            time_vector = time_vector[1:]
            # Get the interpolated data
            for j in range(len(time_vector)):
                imu_time.append(time_vector[j])
                imu_data.append(np.zeros(6))
                for k in range(6):
                    imu_data[-1][k] = (imu_df.iloc[i+1, -6+k] - imu_df.iloc[i, -6+k]) * (time_vector[j] - time) / (imu_df["time"][i+1] - time) + imu_df.iloc[i, -6+k]

    # Convert the imu_time and imu_data to numpy arrays
    imu_time = np.array(imu_time, dtype=np.int64)
    imu_data = np.array(imu_data)
                

    # Read the lidar data
    lidar_files = [f for f in os.listdir(os.path.join(path, "lidar")) if f.endswith(".bin")]
    lidar_files.sort()

    # Get the first lidar file time
    first_lidar_file = lidar_files[0]
    first_lidar_time = np.int64(first_lidar_file.split(".")[0])
    if first_lidar_time > 1e16:
        first_lidar_time = first_lidar_time
    else:
        first_lidar_time = first_lidar_time * 1000
    
    # Remove the imu data before the first lidar file time
    mask = imu_time >= first_lidar_time - 1000000000  # 1 second before the first lidar file time
    imu_time = imu_time[mask]
    imu_data = imu_data[mask]

    ## For debug, plot the IMU data
    #import matplotlib.pyplot as plt
    #fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    #for i in range(6):
    #    ax = axs[i // 2, i % 2]
    #    ax.plot(imu_time * 1e-9, imu_data[:, i], label=imu_df.columns[-6+i])
    #    ax.set_xlabel("Time (s)")
    #    ax.set_ylabel(imu_df.columns[-6+i])
    #    ax.legend()
    #plt.tight_layout()
    #plt.show()


    # Write the IMU data to the bag
    print("----Writing IMU data to bag")
    for i in range(len(imu_time)):
        # Create the message
        msg = Imu()
        msg.header.stamp.sec = int(imu_time[i] // 1e9)
        msg.header.stamp.nanosec = int(imu_time[i] % 1e9)
        msg.header.frame_id = "dmu_imu"
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        msg.angular_velocity.x = imu_data[i][0]
        msg.angular_velocity.y = imu_data[i][1]
        msg.angular_velocity.z = imu_data[i][2]
        msg.linear_acceleration.x = imu_data[i][3]
        msg.linear_acceleration.y = imu_data[i][4]
        msg.linear_acceleration.z = imu_data[i][5]
        # Write the message to the bag
        bag_writer.write("/imu/data", serialize_message(msg), imu_time[i])


    # Write the wheel encoder data to the bag
    lidar_start_time, lidar_end_time = getLidarTimeSpan(path)
    writeEncoderToBag(bag_writer, path, lidar_start_time - kTimeMargin, lidar_end_time + kTimeMargin)


    for i, lidar_file in enumerate(lidar_files):
        # Read the lidar data
        lidar_data = np.fromfile(os.path.join(path, "lidar", lidar_file), dtype=np.float32)
        lidar_data = lidar_data.reshape(-1, 6)

        times = lidar_data[:, -1].astype(np.float64)
        file_time = float(lidar_file.split(".")[0])
        if file_time > 1e16:
            times = times + file_time * 1e-9
        else:
            times = times + file_time * 1e-6

        times_frame = ((times - times[0])*1e9).astype(np.int32)
        msg_time = np.array([times[0]*1e9], dtype=np.int64)

        # Create the message
        msg = PointCloud2()
        msg.header.stamp.sec = int(msg_time // 1e9)
        msg.header.stamp.nanosec = int(msg_time % 1e9)
        msg.header.frame_id = "velodyne"
        msg.height = 1
        msg.width = len(times_frame)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
            PointField(name="time", offset=18, datatype=PointField.UINT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * len(times_frame)
        msg.is_dense = True
        
        x = lidar_data[:, 0].astype(np.float32)
        y = lidar_data[:, 1].astype(np.float32)
        z = lidar_data[:, 2].astype(np.float32)
        intensity = lidar_data[:, 3].astype(np.float32)
        ring = lidar_data[:, 4].astype(np.uint16)
    
        data = np.zeros(msg.row_step, dtype=np.uint8)

        x_uint8 = np.frombuffer(np.array(x, dtype=np.float32).tobytes(), dtype=np.uint8)
        data[0::msg.point_step] = x_uint8[0::4]
        data[1::msg.point_step] = x_uint8[1::4]
        data[2::msg.point_step] = x_uint8[2::4]
        data[3::msg.point_step] = x_uint8[3::4]
        y_uint8 = np.frombuffer(np.array(y, dtype=np.float32).tobytes(), dtype=np.uint8)
        data[4::msg.point_step] = y_uint8[0::4]
        data[5::msg.point_step] = y_uint8[1::4]
        data[6::msg.point_step] = y_uint8[2::4]
        data[7::msg.point_step] = y_uint8[3::4]
        z_uint8 = np.frombuffer(np.array(z, dtype=np.float32).tobytes(), dtype=np.uint8)
        data[8::msg.point_step] = z_uint8[0::4]
        data[9::msg.point_step] = z_uint8[1::4]
        data[10::msg.point_step] = z_uint8[2::4]
        data[11::msg.point_step] = z_uint8[3::4]
        intensity_uint8 = np.frombuffer(np.array(intensity, dtype=np.float32).tobytes(), dtype=np.uint8)
        data[12::msg.point_step] = intensity_uint8[0::4]
        data[13::msg.point_step] = intensity_uint8[1::4]
        data[14::msg.point_step] = intensity_uint8[2::4]
        data[15::msg.point_step] = intensity_uint8[3::4]
        ring_uint8 = np.frombuffer(np.array(ring, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data[16::msg.point_step] = ring_uint8[0::2]
        data[17::msg.point_step] = ring_uint8[1::2]
        time_uint8 = np.frombuffer(np.array(times_frame, dtype=np.uint32).tobytes(), dtype=np.uint8)
        data[18::msg.point_step] = time_uint8[0::4]
        data[19::msg.point_step] = time_uint8[1::4]
        data[20::msg.point_step] = time_uint8[2::4]
        data[21::msg.point_step] = time_uint8[3::4]
        

        msg.data = data.tolist()
        # Write the message to the bag
        bag_writer.write("/velodyne_points", serialize_message(msg), msg_time[0])

        print("Writing lidar data to bag: ", i, " / ", len(lidar_files), " - ", lidar_file, end="           \r")

    print("")
    # Close the bag writer
    bag_writer.close()


    ## Write a yaml file with the metadata for the convert/compression
    #print("Writing out.yaml file")
    #yaml_file = os.path.join(path, "out.yaml")
    #with open(yaml_file, "w") as f:
    #    f.write("output_bags:\n")
    #    f.write("  - uri: " + os.path.join(path, "lidar_imu_ros2bag") + "\n")
    #    f.write("    all_topics: true\n")
    #    f.write("    storage_id: mcap\n")
    #    f.write("    compression_mode: message\n")
    #    f.write("    compression_format: zstd\n")


    ## Compress the bag
    #print("Compressing bag")
    #os.system(f"ros2 bag convert -i {os.path.join(path, 'lidar_imu_ros2bag_temp')} -o {os.path.join(path, 'out.yaml')}")

    ## Remove the temporary bag
    #os.system(f"rm -rf {os.path.join(path, 'lidar_imu_ros2bag_temp')}")
    #os.system(f"rm -rf {os.path.join(path, 'out.yaml')}")






if __name__ == "__main__":
    main()
    
