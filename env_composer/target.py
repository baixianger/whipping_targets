"""A Target Entity for being whipped."""
import numpy as np
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable


N = 3
mesh_paths = [f'env/meshes/targets/material_{i}.obj' for i in range(N)]
texture_paths = [f'env/meshes/targets/material_{i}.png' for i in range(N)]

class TargetObservables(composer.Observables):
    """A touch sensor which averages contact force over physics substeps."""
    @composer.observable
    def target_xpos(self):
        """Returns the xpos of the target."""
        target_body = self._entity.target_body
        return observable.MJCFFeature('xpos', target_body)


class Target(composer.Entity):
    """A Target Entity."""
    def _build(self, *args, **kwargs):
        """Initializes the target."""
        self._model = self._make_model(**kwargs)
        self._target_body = self._model.find('body', 'target')
        self._target_geom = self._model.find('geom', 'target_geom1')

    def _build_observables(self):
        """Returns the observables for the target."""
        return TargetObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def target_body(self):
        """Returns the target body."""
        return self._target_body

    @target_body.setter
    def target_body(self, value):
        """Sets the target body."""
        self._target_body = value

    def _make_model(self, random=False):
        """"""
        model = mjcf.RootElement(model='target')
        visual = model.default.add('default', dclass='visual')
        for key, value in {'group':2, 'type':'mesh', 'contype':0, 'conaffinity':0}.items():
            setattr(visual.geom, key, value)
        collision = model.default.add('default', dclass='collision')
        for key, value in {'group':3, 'type':'mesh'}.items():
            setattr(collision.geom, key, value)

        i = np.random.randint(0, N) if random else 0
        texture = model.asset.add('texture', type='2d',
                                  name='target_texture', file=texture_paths[i])
        material = model.asset.add('material', name='target_material',
                                   texture=texture, specular=0.0, shininess=0.0)
        mesh = model.asset.add('mesh', name='target_mesh', file=mesh_paths[i])

        target = model.worldbody.add('body', name='target')
        target.add('joint', name='target_joint', type='ball')
        target.add('geom', name='target_geom0', type='mesh',
                   mesh=mesh, material=material, dclass='visual')
        target.add('geom', name='target_geom1', type='mesh',
                   mesh=mesh, mass=0.5, dclass='collision')
        return model

def test():
    """Test the target."""
    target = Target()
    mjcf_model = target.mjcf_model
    xml = mjcf_model.to_xml_string()
    # save to file
    with open('env/test/target.xml', 'w', encoding="utf-8") as file:
        file.write(xml)
    print("xml saved to env/test/target.xml")

if __name__ == '__main__':
    test()
