# Cloud Info
Header header 

# 每个ring的points的index范围
int32[] startRingIndex # size为16
int32[] endRingIndex   # size为16

# size为点云cloud_deskewed的大小, 表示所有的points
int32[]  pointColInd # point column index in range image 深度图的col
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# 通过imu数据得到, 点云帧的起始orientation(lidar系下)
# Attitude for lidar odometry initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Odometry 
# 通过vins odom数据得到, 点云帧的起始pose
float32 odomX
float32 odomY
float32 odomZ
float32 odomRoll
float32 odomPitch
float32 odomYaw

# Odometry reset ID
int64 odomResetId # vins跟丢以后再重启, odom重新开始

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed, by imageProjection
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature, by featureExtraction
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature, by featureExtraction