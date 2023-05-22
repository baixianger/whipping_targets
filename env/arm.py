"""A dm_control Entity for a 7-DOF Panda arm."""
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable


stl_visual_paths = [f'env/meshes/visual/link{i}.stl' for i in range(8)]

stl_collision_paths = [f'env/meshes/collision/link{i}.stl' for i in range(8)]

body_params = [
    {'name': 'link0', 'childclass': 'panda'},
    {'name': 'link1', 'pos': [0, 0, 0.333]},
    {'name': 'link2', 'pos': [0, 0, 0], 'quat': [0.707107, -0.707107, 0, 0]},
    {'name': 'link3', 'pos': [0, -0.316, 0],
        'quat': [0.707107, 0.707107, 0, 0]},
    {'name': 'link4', 'pos': [0.0825, 0, 0],
        'quat': [0.707107, 0.707107, 0, 0]},
    {'name': 'link5', 'pos': [-0.0825, 0.384, 0],
        'quat': [0.707107, -0.707107, 0, 0]},
    {'name': 'link6', 'pos': [0, 0, 0], 'quat': [0.707107, 0.707107, 0, 0]},
    {'name': 'link7', 'pos': [0.088, 0, 0], 'euler': [1.57, 0, 0]},
]

inertial_params = [
    {'pos': [0, 0, 0],
     'quat':[1, 0, 0, 0],
     'mass': 3.06,
     'diaginertia': [0.3, 0.3, 0.3]},
    {'pos': [3.875e-03, 2.081e-03, -0.1750],
     'quat':[1, 0, 0, 0],
     'mass': 4.970684,
     'fullinertia': [7.0337e-01, 7.0661e-01, 9.1170e-03, -1.3900e-04, 6.7720e-03, 1.9169e-02]},
    {'pos': [-3.141e-03, -2.872e-02, 3.495e-03],
     'quat':[1, 0, 0, 0],
     'mass': 0.646926,
     'fullinertia': [7.9620e-03, 2.8110e-02, 2.5995e-02, -3.9250e-03, 1.0254e-02, 7.0400e-04]},
    {'pos': [2.7518e-02, 3.9252e-02, -6.6502e-02],
     'quat':[1, 0, 0, 0],
     'mass': 3.228604,
     'fullinertia':[3.7242e-02, 3.6155e-02, 1.0830e-02, -4.7610e-03, -1.1396e-02, -1.2805e-02]},
    {'pos': [-5.317e-02, 1.04419e-01, 2.7454e-02],
     'quat':[1, 0, 0, 0],
     'mass': 3.587895,
     'fullinertia': [2.5853e-02, 1.9552e-02, 2.8323e-02, 7.7960e-03, -1.3320e-03, 8.6410e-03]},
    {'pos': [-1.1953e-02, 4.1065e-02, -3.8437e-02],
     'quat':[1, 0, 0, 0],
     'mass': 1.225946,
     'fullinertia': [3.5549e-02, 2.9474e-02, 8.6270e-03, -2.1170e-03, -4.0370e-03, 2.2900e-04]},
    {'pos': [6.0149e-02, -1.4117e-02, -1.0517e-02],
     'quat':[1, 0, 0, 0],
     'mass': 1.666555,
     'fullinertia': [1.9640e-03, 4.3540e-03, 5.4330e-03, 1.0900e-04, -1.1580e-03, 3.4100e-04]},
    {'pos': [1.0517e-02, -4.252e-03, 6.1597e-02],
     'quat':[1, 0, 0, 0],
     'mass': 7.35522e-01,
     'fullinertia': [1.2516e-02, 1.0027e-02, 4.8150e-03, -4.2800e-04, -1.1960e-03, -7.4100e-04]},
]

joint_params = [
    {},
    {'name': 'joint1', 'range': [-2.8973, 2.8973], 'frictionloss': 5},
    {'name': 'joint2', 'range': [-1.7628, 1.7628], 'frictionloss': 2},
    {'name': 'joint3', 'range': [-2.8973, 2.8973], 'frictionloss': 2},
    {'name': 'joint4', 'range': [-3.0718, -0.400], 'frictionloss': 0.5},
    {'name': 'joint5', 'range': [-2.8973, 2.8973], 'frictionloss': 1},
    {'name': 'joint6', 'range': [-1.6573, 2.1127], 'frictionloss': 0.5},
    {'name': 'joint7', 'range': [-2.8973, 2.8973], 'frictionloss': 0.5},
]

general_params = [
    {},
    {'name': 'actuator_1', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87],
     'gainprm': [4500,], 'biasprm': [0, -4500, -450]},
    {'name': 'actuator_2', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87],
     'gainprm': [4500,], 'biasprm': [0, -4500, -450]},
    {'name': 'actuator_3', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87],
     'gainprm': [3500,], 'biasprm': [0, -3500, -350]},
    {'name': 'actuator_4', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87],
     'gainprm': [3500,], 'biasprm': [0, -3500, -350]},
    {'name': 'actuator_5', 'forcerange': [-12, 12], 'ctrlrange': [-12, 12],
     'gainprm': [2000,], 'biasprm': [0, -2000, -200]},
    {'name': 'actuator_6', 'forcerange': [-12, 12], 'ctrlrange': [-12, 12],
     'gainprm': [2000,], 'biasprm': [0, -2000, -200]},
    {'name': 'actuator_7', 'forcerange': [-12, 12], 'ctrlrange': [-12, 12],
     'gainprm': [2000,], 'biasprm': [0, -2000, -200]},
]

motor_params = [
    {},
    {'name': 'actuator_1', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87]},
    {'name': 'actuator_2', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87]},
    {'name': 'actuator_3', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87]},
    {'name': 'actuator_4', 'forcerange': [-87, 87], 'ctrlrange': [-87, 87]},
    {'name': 'actuator_5', 'forcerange': [-12, 12], 'ctrlrange': [-12, 12]},
    {'name': 'actuator_6', 'forcerange': [-12, 12], 'ctrlrange': [-12, 12]},
    {'name': 'actuator_7', 'forcerange': [-12, 12], 'ctrlrange': [-12, 12]},
]


class ArmObservables(composer.Observables):
    """Observables for the Arm entity. The self._entity is the given composer.Entity class"""
    @composer.observable
    def arm_joints_qpos(self):
        """Returns the qpos of the arm joints."""
        arm_joints = self._entity.arm_joints
        return observable.MJCFFeature('qpos', arm_joints)

    @composer.observable
    def arm_joints_qvel(self):
        """Returns the qvel of the arm joints."""
        arm_joints = self._entity.arm_joints
        return observable.MJCFFeature('qvel', arm_joints)

    @composer.observable
    def arm_joints_qacc(self):
        """Returns the qacc of the arm joints."""
        arm_joints = self._entity.arm_joints
        return observable.MJCFFeature('qacc', arm_joints, aggregator='max')

    @composer.observable
    def arm_joints_qfrc(self):
        """Returns the qfrc of the arm joints."""
        arm_joints = self._entity.arm_joints
        return observable.MJCFFeature('qfrc_applied', arm_joints, aggregator='max')


class Arm(composer.Entity):
    """A 7-DOF Panda arm."""

    def _build(self, *args, **kwargs):
        """Initializes the arm."""
        ctrl_type = kwargs.get('ctrl_type', 'position')
        self._model = self._make_model(ctrl_type=ctrl_type)
        self._joints = self._model.find_all('joint')
        self._actuators = self._model.find_all('actuator')
        self._ee_site = self._model.find('site', 'ee_site')

    def _build_observables(self):
        """Returns the observables for the arm."""
        return ArmObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def end_effector_site(self):
        """Returns the site for install an end-effector."""
        return self._ee_site

    @property
    def arm_joints(self):
        """Returns a tuple containing the arm joints."""
        return self._joints

    @property
    def actuators(self):
        """Returns a tuple containing the actuators in this arm."""
        return self._actuators

    def _make_model(self, ctrl_type='position'):
        """Returns an MJCF model of a Panda arm."""
        model = mjcf.RootElement(model='arm')
        # set compiler
        model.compiler.angle = 'radian'
        model.compiler.autolimits = True
        # set assets
        collision_meshes = [model.asset.add('mesh',
                                name=f'link{i}_collision',
                                file=stl_collision_paths[i])
                            for i in range(8)]
        visual_meshes = [model.asset.add('mesh',
                                name=f'link{i}_visual',
                                file=stl_visual_paths[i])
                         for i in range(8)]
        # Set default
        default = model.default.add('default', dclass='panda')
        default.joint.pos = [0, 0, 0]
        default.joint.axis = [0, 0, 1]
        default.joint.limited = True
        default.joint.damping = 100
        default.position.forcelimited = True
        default.position.ctrllimited = True
        default.position.user = [1002, 40, 2001, -0.005, 0.005]
        viusal_default = default.add('default', dclass='visual')
        viusal_default.geom.type = 'mesh'
        viusal_default.geom.contype = 0
        viusal_default.geom.conaffinity = 0
        viusal_default.geom.group = 0
        viusal_default.geom.rgba = [.95, .99, .92, 1]
        viusal_default.geom.mass = 0
        collision_default = default.add('default', dclass='collision')
        collision_default.geom.type = 'mesh'
        collision_default.geom.contype = 1
        collision_default.geom.conaffinity = 1
        collision_default.geom.group = 3
        collision_default.geom.rgba = [.5, .6, .7, 1]
        # Set worldbody
        parent_body = model.worldbody
        # Add joint and bind actuator
        ctrl = 'general' if ctrl_type == 'position' else 'motor'
        ctrl_params = general_params if ctrl_type == 'position' else motor_params
        for i in range(8):
            parent_body = parent_body.add('body', **body_params[i])
            parent_body.add('geom', dclass='visual', mesh=visual_meshes[i])
            parent_body.add('geom', dclass='collision',
                            mesh=collision_meshes[i])
            if i != 0:
                temp_joint = parent_body.add('joint', **joint_params[i])
                model.actuator.add(ctrl, joint=temp_joint, dclass='panda',
                                   **ctrl_params[i])
            if i == 7:
                parent_body.add('site', name='ee_site', pos=[0, 0, 0.107],
                                size=[0.005, 0.005, 0.005], euler=[0, 0, -1.57])

        return model

def test():
    """Test the arm with different control settings."""
    arm = Arm(ctrl_type='torque')
    mjcf_model = arm.mjcf_model
    xml = mjcf_model.to_xml_string()
    # save to file
    with open('env/test/panda.xml', 'w', encoding="utf-8") as file:
        file.write(xml)
    print("xml saved to env/test/panda.xml")

if __name__ == '__main__':
    test()
    