"""Test for PPO"""
# pylint: disable=import-error
# pylint: disable=line-too-long
import IPython
from omegaconf import DictConfig, OmegaConf
import hydra
from dm_control import mjcf
from dm_control import viewer
from dm_control import composer
import numpy as np
from env.task import SingleStepTask


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):
    """Test for PPO"""
    print(OmegaConf.to_yaml(cfg))
    print(type(cfg.algo.target_kl))

    task = SingleStepTask(
        ctrl_type="position",
        target=0)
    env = composer.Environment(task=task, random_state=42)
    env.reset()
    action_spec = env.action_spec()
    print(action_spec)

    # 获取target的位置 三种方式
    target = env.task.entities.target.target_body
    target_pos = env.physics.bind(target).pos
    print(target_pos)
    target_mjcf = env.task.entities.target.mjcf_model
    target_attachment_frame = mjcf.get_attachment_frame(target_mjcf)
    target_pos = env.physics.bind(target_attachment_frame).xpos
    print(target_pos, target_attachment_frame)
    target_pos = env.task._target_pos()
    print(target_pos)


    # Define a uniform random policy.
    def random_policy(time_step):
        del time_step  # Unused.
        return np.random.uniform(low=action_spec.minimum,
                            high=action_spec.maximum,
                            size=action_spec.shape)
    # IPython.embed()
    all_bodys = env.task.root_entity.mjcf_model.find_all("body")
    all_geoms = env.task.root_entity.mjcf_model.find_all("geom")
    all_joints = env.task.root_entity.mjcf_model.find_all("joint")
    ee_site = env.task.entities.arm._ee_site

    whip_end = env.task.entities.whip._whip_end
    whip_begin = env.task.entities.whip._whip_begin
    whip_mjcf = env.task.entities.whip.mjcf_model
    whip_attachment_frame = mjcf.get_attachment_frame(whip_mjcf)
    pos = env.physics.bind(whip_attachment_frame).pos

    target = env.task.entities.target.target_body
    target_mjcf = env.task.entities.target.mjcf_model
    target_attachment_frame = mjcf.get_attachment_frame(target_mjcf)
    pos = env.task._target_pos()
    pos = env.physics.bind(target_attachment_frame).pos
    env.physics.bind(target_attachment_frame).xpos = env.task._target_pos()
    env.task.entities.target.set_pose(env.task._target_pos())

    env.physics.bind(env.task.entities.arm.arm_joints).qpos = np.array([3, 3, 3, 3, 3, 3, 3])
    # IPython.embed()
    # # Launch the viewer application.
    # viewer.launch(env, policy=random_policy)

if __name__ == '__main__':
    test() # pylint: disable=no-value-for-parameter
