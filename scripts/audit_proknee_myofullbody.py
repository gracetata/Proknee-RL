#!/usr/bin/env python
"""Print the MyoFullBody left prosthesis control boundary."""

from __future__ import annotations

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.proknee import audit_myofullbody_left_leg


def main() -> int:
    env = MyoFullBody(disable_fingers=True)
    audit = audit_myofullbody_left_leg(env.model)
    print("Left prosthesis joints:")
    for joint in audit.joints:
        print(
            f"  {joint.name}: joint_id={joint.joint_id}, "
            f"qpos={joint.qposadr}, qvel={joint.dofadr}, range={joint.joint_range}"
        )
    print("\nMimic sites:")
    for name, site_id in audit.sites:
        print(f"  {name}: site_id={site_id}")
    print("\nLeft lower-limb muscle actuators:")
    for actuator in audit.actuators:
        print(
            f"  {actuator.name}: actuator_id={actuator.actuator_id}, "
            f"dyntype={actuator.dyntype}, ctrlrange={actuator.ctrlrange}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
