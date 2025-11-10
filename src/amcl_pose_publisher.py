#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler

def publish_amcl_pose():
    rospy.init_node('amcl_pose_publisher')

    frame_id = rospy.get_param('~frame_id', 'map')
    x = float(rospy.get_param('~x', -2.000495))
    y = float(rospy.get_param('~y', -0.497500))
    yaw = float(rospy.get_param('~yaw', 0.039))  # radians

    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()

    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0

    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw

    # 6x6 covariance (row-major). Set x, y, yaw; others 0.
    cov = [0.0]*36
    cov[0]  = 0.05   # var(x)
    cov[7]  = 0.05   # var(y)
    cov[35] = 0.05   # var(yaw)
    msg.pose.covariance = cov

    rospy.loginfo("Publishing /initialpose: x=%.6f y=%.6f yaw=%.6f rad", x, y, yaw)

    rate = rospy.Rate(5)
    for _ in range(10):           # publish a few times
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()

    rospy.loginfo("Done publishing /initialpose.")

if __name__ == '__main__':
    try:
        publish_amcl_pose()
    except rospy.ROSInterruptException:
        pass
