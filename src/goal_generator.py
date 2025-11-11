#!/usr/bin/env python
import rospy
import tf
from geometry_msgs.msg import PoseStamped

goal_received = False

def goal_callback(msg):
    global goal_received
    rospy.loginfo("Move base received a goal at (%.2f, %.2f)" % (msg.pose.position.x, msg.pose.position.y))
    goal_received = True

def send_goal():
    global goal_received
    rospy.init_node("send_goal_node")

    # Hard-coded goal values
    x = -1.89
    y = -0.15
    yaw = 0.039261

    # Publisher and subscriber
    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, goal_callback)

    rospy.sleep(1.0)  # wait for publisher connection

    # Convert yaw to quaternion
    quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0
    goal.pose.orientation.x = quat[0]
    goal.pose.orientation.y = quat[1]
    goal.pose.orientation.z = quat[2]
    goal.pose.orientation.w = quat[3]

    rospy.loginfo("Sending fixed goal: x=%.2f, y=%.2f, yaw=%.2f" % (x, y, yaw))
    pub.publish(goal)

    # Wait until callback confirms receipt
    timeout = rospy.Time.now() + rospy.Duration(3.0)  # 3 seconds timeout
    while not rospy.is_shutdown() and not goal_received and rospy.Time.now() < timeout:
        rospy.sleep(0.1)

    if goal_received:
        rospy.loginfo("Goal successfully received by move_base. Exiting.")
    else:
        rospy.logwarn("No confirmation received within timeout. Exiting anyway.")

if __name__ == "__main__":
    try:
        send_goal()
    except rospy.ROSInterruptException:
        pass
