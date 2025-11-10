#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped


def make_outline_marker(frame_id, ns, points, color_rgba, z=0.0):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = 0
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.03  # line width
    m.color.r, m.color.g, m.color.b, m.color.a = color_rgba
    for (x, y) in points:
        p = Point(x=x, y=y, z=z)
        m.points.append(p)
    # close the loop
    m.points.append(Point(x=points[0][0], y=points[0][1], z=z))
    return m


def make_filled_marker(frame_id, ns, center, size_xy, color_rgba, z=0.0, height=0.02):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = 1
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = center[0]
    m.pose.position.y = center[1]
    m.pose.position.z = z + height / 2.0
    m.pose.orientation.w = 1.0
    m.scale.x = size_xy[0]
    m.scale.y = size_xy[1]
    m.scale.z = height
    m.color.r, m.color.g, m.color.b, m.color.a = color_rgba
    return m


def main():
    rospy.init_node('box_marker')

    frame_id = rospy.get_param('~frame_id', 'map')

    # Expected rectangle vertices (counter-clockwise):
    # (1.01, 2.31), (0.69, 2.31), (0.69, 1.85), (1.01, 1.85)
    # (User message repeated one vertex; we complete the set to form a rectangle.)
    v0 = rospy.get_param('~v0', [1.01, 2.31])
    v1 = rospy.get_param('~v1', [0.69, 2.31])
    v2 = rospy.get_param('~v2', [0.69, 1.85])
    v3 = rospy.get_param('~v3', [1.01, 1.85])

    points = [(float(v0[0]), float(v0[1])),
              (float(v1[0]), float(v1[1])),
              (float(v2[0]), float(v2[1])),
              (float(v3[0]), float(v3[1]))]

    # Compute center and size
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    center = ((max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0)
    size_xy = (max(xs) - min(xs), max(ys) - min(ys))

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1, latch=True)
    center_pub = rospy.Publisher('/box_center', PoseStamped, queue_size=1, latch=True)

    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        outline = make_outline_marker(frame_id, 'box_outline', points, (1.0, 0.0, 0.0, 1.0))
        filled = make_filled_marker(frame_id, 'box_filled', center, size_xy, (0.0, 0.5, 1.0, 0.2))
        pub.publish(outline)
        pub.publish(filled)

        # Publish center as PoseStamped for consumers (e.g., goal generator)
        center_msg = PoseStamped()
        center_msg.header.frame_id = frame_id
        center_msg.header.stamp = rospy.Time.now()
        center_msg.pose.position.x = center[0]
        center_msg.pose.position.y = center[1]
        center_msg.pose.position.z = 0.0
        center_msg.pose.orientation.w = 1.0
        center_pub.publish(center_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


