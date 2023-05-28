"""A Whip Entity."""
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable

_COUNT = 30 # number of links in the whip
_MASS = 0.5 # mass of the whip
_LENGTH = 1 # length of the whip

_UNIT_L = _LENGTH / _COUNT
_UNIT_M = _MASS / _COUNT

whip_params = {
    'joint0': {'type': 'hinge', 
               'axis': [1, 0, 0],
               'group': 3,
               'springref': 0, 
               'stiffness': 0.242, 
               'damping': 0.092},
    'joint1': {'type': 'hinge',
               'axis': [0, 1, 0],
               'group': 3,
               'springref': 0,
               'stiffness': 0.242,
               'damping': 0.092},
    'sphere': {'type': 'sphere', 
               'pos': [0, 0, _UNIT_L], 
               'size': [0.006,], 
               'mass': _UNIT_M},
    'cylinder': {'type': 'cylinder', 
                 'fromto': [0, 0, 0, 0, 0, _UNIT_L], 
                 'size': [0.006,], 
                 'mass': 0},
    'capsule': {'type': 'capsule', 
                'fromto': [0, 0, 0, 0, 0, _UNIT_L],
                'size': [0.006,],
                'mass': _UNIT_M},
}


class WhipObservables(composer.Observables):
    """Observables for the Whip entity. The self._entity is the given composer.Entity class"""
    @composer.observable
    def whip_begin_xpos(self):
        """Returns the xpos of the whip begin node."""
        whip_begin = self._entity.whip_begin
        return observable.MJCFFeature('xpos', whip_begin)

    @composer.observable
    def whip_end_xpos(self):
        """Returns the xpos of the whip end node."""
        whip_end = self._entity.whip_end
        return observable.MJCFFeature('xpos', whip_end)


class Whip(composer.Entity):
    """A Whip Entity."""
    def _build(self, *args, **kwargs):
        """Initializes the whip."""
        self._model = self._make_model(**kwargs)
        self._whip_begin = self._model.find('body', 'B0')
        self._whip_end = self._model.find('body', 'whip_end')
        self._whip_joints = self._model.find_all('joint')

    def _build_observables(self):
        """Returns the observables for the whip."""
        return WhipObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def whip_end(self):
        """Returns the whip end."""
        return self._whip_end

    @property
    def whip_begin(self):
        """Returns the whip begin."""
        return self._whip_begin

    @property
    def whip_joints(self):
        """Returns the whip joints."""
        return self._whip_joints

    def update_whip_color(self, physics):
        """Updates the activation of the whip."""
        physics.bind(self._model.find_all('geom')).rgba = [1, 0, 0, 1]

    def _make_model(self, whip_type=0):
        """Returns an MJCF model of a Whip."""
        if whip_type not in [0, 1]:
            raise ValueError("whip type must be 0 or 1")

        model = mjcf.RootElement(model='whip')
        # set compiler
        model.compiler.angle = 'radian'
        model.compiler.autolimits = True
        model.option.integrator = "implicit"
        model.option.impratio = "10"

        whip_material = model.asset.add('material', name="white", rgba="1 1 1 1")
        whip = model.worldbody.add('body', name="whip")
        whip.add('geom', type="sphere", size=[0.045,], material=whip_material)
        temp_body = whip.add('body', name='B0', pos=[0, 0, 0.045])

        for i in range(_COUNT):
            temp_body.add('joint', name=f'J0_{i}', **whip_params['joint0'])
            temp_body.add('joint', name=f'J1_{i}', **whip_params['joint1'])
            if whip_type == 0:
                temp_body.add('geom',  name=f'G0_{i}',
                              material=whip_material, **whip_params['sphere'])
                temp_body.add('geom',  name=f'G1_{i}',
                              material=whip_material, **whip_params['cylinder'])
            elif whip_type == 1:
                temp_body.add('geom',  name=f'G0_{i}',
                              material=whip_material, **whip_params['capsule'])
            name = f'B{i+1}' if i != _COUNT - 1 else 'whip_end'
            temp_body = temp_body.add('body',  name=name, pos=[0, 0, _UNIT_L])

        return model

def test():
    """Test the whip."""
    for i in range(2):
        whip = Whip(whip_type=i)
        mjcf_model = whip.mjcf_model
        xml = mjcf_model.to_xml_string()
        # save to file
        with open(f'env/test/whip{i}.xml', 'w', encoding="utf-8") as file:
            file.write(xml)
        print(f"xml saved to env/test/whip{i}.xml")

if __name__ == '__main__':
    test()
