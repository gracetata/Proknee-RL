from .bimanual import MyoBimanualArm, MjxMyoBimanualArm
from .myofullbody import MyoFullBody, MjxMyoFullBody
from .myofullbody_prosthesis import MyoFullBodyProsthesisEnv, MjxMyoFullBodyProsthesisEnv


# register muscle environments
MyoBimanualArm.register()
MjxMyoBimanualArm.register()
MyoFullBody.register()
MjxMyoFullBody.register()
MyoFullBodyProsthesisEnv.register()
MjxMyoFullBodyProsthesisEnv.register()
