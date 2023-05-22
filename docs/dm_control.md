## Entity类
成员变量：
- `_post_init_hooks`
- `_parent`
- `_attached` entity对象的列表
- `observable_options` 此时要检查初始化输入的参数有没有`observable_options`，如果有则将其存入`_observables`属性中，`_get_observation.set_state(observable_options)`。
- `_build()`
- `_observables` 调用`_build_observables()`

property方法：
- `observables()`
- `mjcf_model()`
- `parent()`
- `attachment_site()`
- `root_body()` 如果对象自己没有被attach也就是没有parent，则返回自己的worldbody，否则返回附着在parent上的具体位置

成员方法：
- `_build()`
- `_build_observables()` 生成`Observables`对象,注意不是`Observable`对象, 一般要自己复现
- `iter_entities(exclude_self=False)` 枚举所有的`entity`对象
- `attach(entity, attach_site)` 将`entity`对象附加到parent对象上, 如果没有指定attach_site，则调用`attachment_site()`属性方法，返回自身的mjcf_model做为attach_site。这个方法也会改写对传入的`entity`对象的_parent属性，将其存入对象自己的引用。
- `detach()` 将对象自己从parent对象上分离，前提是自己已经被附加到了parent对象上

- `initialize_episode_mjcf(random_state)` 初始化MJCF模型
- `after_compile()` MJCF模型编译后的回调函数
- `initialize_episode()` 初始化episode
- `before_step()` 每个step之前的回调函数
- `before_substep()` 每个substep之前的回调函数
- `after_substep()` 每个substep之后的回调函数

其他的一些辅助方法：
- `global_vector_to_local_frame(physics, vec_in_world_frame)` 方法会调用`root_body()`方法，得到对象的mjcf模型在父对象中的挂载点(元素)相对于父对象的worldbody的旋转矩阵self_mat，如果没有父对象，其实返回的是自己的worldbody对象相对自己自身的旋转矩阵，即一个单位矩阵，等于vec_in_world_frame就是vec_in_local_frame。 计算公式`self_mat * vec_in_world_frame`
- `global_xmat_to_local_frame(physics, xmat)` xmat是父对象的一个元素相对于父对象worldbody的旋转矩阵，方法会调用root_body()方法，得到对象的mjcf模型在父对象中的挂载点(元素)相对于父对象的worldbody的旋转矩阵self_mat，如果没有父对象，其实返回的是自己的worldbody对象相对自己自身的旋转矩阵，即一个单位矩阵，等于xmat就是xmat_in_local_frame. 计算公式`self_mat.T * xmat`，因为旋转矩阵的逆等于自身的转置。
- `get_pose(physics)` 返回自身相对于父对象的位置和方位。如果是freejoint返回的是全局坐标，如果没有freejoint返回的是局部坐标，即attachment_frame(一个mjcf自动添加到父mujc模型上用来包裹子mjcf的一个虚拟body元素)相对于自己上一级body元素的位置和方位。
- `set_pose(physics, position, quaternion)` 如果是通过freejoint挂载到parent上则设置的是全局的坐标，如果是其他设置的是相对的坐标
- `shift_pose(physics, position, quaternion, rotate_velocity)` 跟`set_pose()`类似，只是设置的是在原先的基础上的偏移量，参数rotate_velocity可以决定要不要对线速度的方向也进行对应的旋转，注意此时也只是选择线速度的方向，不改变角速度。
- `get_velocity(physics)` 只能返回freejoint的线速度和角速度
- `set_velocity(physics, velocity, angular_velocity)` 只能设置freejoint的线速度和角速度
- `configure_joints(physics, position)` 配置freejoint的qpos, 最好复写一遍此方法实现对joint的精细配置


抽象方法：
- `_build()` 需要`entitiy`子类自己实现的初始化方法，可以接收自定义的参数
- `mjcf_model()` 需要`entity`子类自己实现的返回MJCF模型的`property方法`


## Observable类

成员变量：
- `_update_interval`  更新时间
- `self._buffer_size` 
- `self._delay` 模拟传感器的延迟，而且可以是变量也可以是方法
- `self._aggregator` 返回一个函数，但必须min max mean median sum的操作
- `self._corruptor` 模拟传感器的噪声
- `self._enabled` 是否启用

属性方法：
- `update_interval()`
- `buffer_size()`
- `delay()`
- `aggregator()`
- `corruptor()`
- `enabled()`
- `array_spec()`
- `observation_callable(physics, random_state)` 调用具体实现的_callable方法，实现对传感器的观测
- `_call__(physics, random_state)` 将observation_callable方法包装成类方法，实现对传感器的观测
- `configure()` 配置传感器的参数, 例如`_update_interval`, `_buffer_size`, `_delay`, `_aggregator`, `_corruptor`, `_enabled`等, 因为同常子类初始化时一般不传入这些配置，配置同常由与entity绑定的observables对象来配置，这时就会调用此方法来间接配置传感器的参数。
- `observable`的参数的具体配置流程如下：
    - entity对象初始化时接收的一众参数里有个叫'observable_options'的参数，是字典类型里面包含了各项设置键值对，entity有个成员是observables对象，
    - 通过entity对象内部的observables对象成员的set_options方法来接受这些参数，set_options方法会检查传入的参数字典的key是否是预先定义好的。同时observables对象内部又有一个字典型的成员变量`_observables`，用来存储通过修饰符`@define.observable`传入的每个observable对象。
    - 继续在set_options方法内，分别对内部的每个observable对象调用configure方法来配置每个传感器的参数，配置参数的过程其实等同于修改observable的一系列保存配置参数的私有成员。

抽象方法：
- `_callable(physics)`

继承类
- `Generic` 定义了私有成员`_raw_callable`，是一个对传感器的具体实现方法的引用，以此实现`_callable`方法
- `MujocoFeature` 新增了`_kind`和`_feature_name`两个成员变量，前者表示要索引的元素类型，后者表示要查询的参数名称。通过`physics.named.data._kind._feature_name`来实现`_callable`方法，得到具体的参数值。注意`_feature_name`可能是对象的成员变量也可能是成员方法，具体取决于data对象的规定。
- `MJCFFeature` 新增了`_mjcf_element`，`_kind`和`_index`三个成员变量. 对于`_mjcf_element`成员无论初始化时给定的是一个`mjcf.Element`, 还是`mjcf.Element`的列表，一律转成列表，方便可以对多个元素批量查询. `_index`可以选项，用来指定选择返回参数的某个元素。对于`_kind`用来指定`mjcf.Physics.bind(_mjcf_element)`下具体要查找的元素类型。`_callable`的具体实现依托于`mjcf.Physics.bind(_mjcf_element)`方法。 如果没有提供具体的`_index`，还是实现了`__getitem__(key)`方法来实现对具体返回参数的索引。
- `MJCFCamera`实现了获取某个摄像机的画面，这个类可以方便给模拟器增加CV任务。

During each control step the evaluation of observables is optimized such that only callables for observables that can appear in future observations are evaluated. For example, if we have an observable with `update_interval=1` and `buffer_size=1` then it will only be evaluated once per control step, even if there are multiple simulation steps per control step. This avoids the overhead of computing intermediate observations that would be discarded.

对观测的评估在每个控制步骤中进行优化，以便只评估可以出现在未来观测中的可调用对象。 例如，如果我们有一个具有`update_interval = 1`和`buffer_size = 1`的可观察对象，那么它将每个控制步骤仅评估一次，即使每个控制步骤中有多个模拟步骤。 这避免了计算将被丢弃的中间观测的开销。


## Observables类

此类是Entity内部所有Observable对象的容器，子类通过@define.observable修饰器，来返回一个Observable对象。

成员变量：
- `_entity` 与其绑定的Entity对象
- `_observables` 是一个OrdereddDict对象，修饰器@define.observable会传入很多对应的Observable对象，比如MJCFFeature，MujocoFeature，Generic等，这些对象都会被存入_observables中，key是方法名，value是对象本身。
- `_keys_helper` 

属性方法：
- `keys_helper()` 返回_ObservableKeys对象，可以查看具体的Observable对象的全称，`mjcf_model.full_identifier/observable_name`

成员方法：
- `as_dict(full_qualified)` 返回_observables，如果full_qualified为True，则会把对应的entity对象的full_identifier作为前缀添加到key中。
- `get_observable(name, name_fully_qualified)` 返回对应的observable对象，如果name_fully_qualified为True，则代表输入的name参数是带full_identifier前缀的key，内部实现时清除了前缀后再进行索引。
- `set_options(options)` 同常与其绑定的entity初始化时会调用这个方法实现对内部observable对象的设置，比如设置update_interval, buffer_size, delay, corruptor, aggregator等。
- `enable_all()` 
- `disable_all`
- `add_observable(name, observable, enabled)` 保留一个能手动继续添加observable对象的接口。一般情况会通过修饰器@define.observable来添加。


## Task类

成员有:
- `_entity`
- `root_entity` 返回_entity对象
- `iter_entities` 返回所有被组装起来的子entity对象

属性方法：
- `task_observables` 返回OrderedDict字典，value为Observable对象
- `observables` 不仅返回task_observers，这个属性方法会把attach到root_entity上的所有子entity的observable都返回
- `control_timestep` 此属性方法会检查root_entity是否存在，同时检查有无设置control_timestep，返回physics_timestep，单位都是秒
- `physics_timestep` 此属性方法会检查root_entity是否存在，如果root_entity包装的mjcf模型中的option内没有设置timestep, 默认是0.002秒， mujoco的默认配置。
- `physics_steps_per_control_step` 此属性会返回control_timestep和physics_timestep的比值，如果不是整数倍，会抛出异常。

成员方法：
- `_check_root_entity(callee_name)` 检查root_entity是否存在，如果不存在则抛出异常
- `set_timesteps(control_timestep, physics_timestep)` 此方法会同时调用control_timestep和physics_timestep的setter方法，来设置两个时间步长，同时会检查两个步长之间是不是整数倍。
- `action_spec(physics)` 返回一个BoundeArray对象，是numpy数组的子类，用来表示action的上下限。同时把每个actuator的名称也存入其中，如果在搭建mjcf模型的时候没有声明name，则会使用actuator的id作为名称。
- `get_reward_spec()` 用来定义非标量奖励值
- `get_discount_spec()` 用来定义非标量折扣值

跟环境交互的回调函数：
- `after_compile(physics, random_state)` 需要自己实现的callback方法
- `initialize_episode_mjcf(physics, random_state)` 下一个episode开始前，对mjcf模型进行初始化的回调方法
- `initialize_episode(physics, random_state)` 下一个episode开始前，对物理模拟器进行初始化的回调方法，在initialize_episode_mjcf方法的调用之后
- `before_step(physics, action, random_state)` 在一个代理的控制step之前调用的回调方法，用来配置actuator的控制信号
- `before_substep(physics, action, random_state)` 在一次物理模拟step之前调用的回调方法
- `after_substep(physics, random_state)` 在一次物理模拟step之后调用的回调方法
- `after_step(physics, random_state)` 在一个代理的控制step之后调用的回调方法，可能是可以用来获取observation
- `should_terminate_episode(physics, random_state)` 基于physica即模拟器的运行状态，决定是否要终止完成此episode，注意一个episode可能会有多个控制step，所以这个方法会被多次调用。
- `get_reward(physics, random_state)` 返回一个标量奖励值

跟环境交互的抽象回调函数：
- `get_reward(physics, random_state)` 对于奖励的设置必须自己实现


## _EnviromentHooks和_CommonEnvironment类

`_Hook`类中定义了两种不同的`'entity_hooks`', `'extra_hooks'`类别

`_EnvironmentHooks`类 定义了hook的接口，这个对象的存在是为了确保在更复杂的任务中不会产生重大的开销，而不需要调用空实体钩子。简单来说，这个对象的目的是为了提高程序运行的效率，避免不必要的计算。当某些钩子没有被使用时，通过使用这个对象
- 接受一个task对象完成初始化，并用`_task`成员对其引用
- 初始化时把这些hook名称都转化成类的成员变量，每个都是一个初始化好的`_Hook对象`       
    - `_initialize_episode_mjcf` allows the MJCF model to be modified be- tween episodes
    - `_after_compile`
    - `_initialize_episode` used to initialize the physics state between episodes
    - `_before_step` One key role of this callback is to translate agent actions into the Physics control vector
    - `_before_substep` These callbacks are useful for detecting transient events that may occur in the middle of an environment step, for example a button is pressed
    - `_after_substep`
    - `_after_step`
    - For a given callback, the one that is defined in the Task is executed first, followed by those defined in each Entity following depth-first traver- sal of the Entity tree starting from the root (which by convention is the arena) and the order in which Entities were attached。
- 初始化时通过`refresh_entity_hooks()`成员方法，可以遍历初始化时保存到`_task`成员内的task对象的所有entity对象，然后获取他们所有的回调函数的引用，同时检查函数的`__code__.co_code`值对空函数进行过滤，将散落在不同entity对象上的回调函数，按照回调函数的类别进行归类。比如`self._initialize_episode_mjcf`是一个`_Hook`对象，里面的成员变量`entity_hooks`列表会保存所有散落在不同entity中对应的`_initialize_episode_mjcf`回调函数的引用。
- 通过`add_extra_hook(hook_name, hook_callable)`自定义将回调函数添加到对应`hook_name`所属的类别的`_Hook`对象内的`extra_hooks`列表中。
- `initialize_episode_mjcf(random_state)`, `after_compile(physics, random_state)`, `initialize_episode(physics, random_state)`, `before_step(physics, random_state)`, `before_substep(physics, random_state)`, `after_substep(physics, random_state)`, `after_step(physics, random_state)`成员方法提供了对外的统一接口。先对`内部task对象`的这些方法进行调用，然后再对对应种类的`_Hook对象`的`entity_hooks`和`extra_hooks`列表中的回调函数进行调用。

`_CommonEnvironment`类
- 核心成员变量：
    - `_task` 保存传入的task对象的引用
    - `_hooks` 用`_EnvironmentHooks类`的机制，分门别类的保存所有的回调函数的引用
    - `_legacy_step` 用来保存是否使用旧的step方法，mujoco中一次forward实现了step1和step2两个步骤，默认是先1后2，但在dm_control中，先2后1，所以这个成员变量用来保存是否使用旧的step方法
    - `_physics` 保存mujoco的physics对象的引用
    - `_observation_updater` 用来保存对应的Updater对象
    - `_random_state`
    - `_time_limit`
    - `_n_sub_steps` 由`_recompile_physics_and_update_observables()`成员方法创建
- 属性方法：
    - `physics()` 返回`_physics`成员变量
    - `task()` 返回`_task`成员变量
    - `random_state()` 返回`_random_state`属性
- 核心成员方法：
    - `_recompile_physics_and_update_observables()` 在初始化时会调用此方法，此方法又会先调用成员方法`_recompile_physics()`获取模拟器实例，然后更新成员变量`_hooks`中各类回调函数的引用(EnvironmentHook对象的refresh_entity_hooks()方法)，接着调用`_hooks`中的`after_compile(physics, random_state)`方法，然后调用成员方法`_make_observation_updater()`返回`observation.Updater`对象，
    -  `_recompile_physics()` 编译模型并将physics对象保存到成员变量`_physics`中
    - `_make_observation_updater()` 返回`observation.Updater对象`并保存到成员变量`_observation_updater`中
    - `control_timestep()` 返回模拟器的控制时间间隔,单位秒

## Environment类
`Environment类`继承了前面的`_CommonEnvironment类`和dm_env库中提供的`Environment类`
`dm_env.Environment类`是一个抽象类，定义了`reset`， `step`， `reward_spec`， `discount_spec`， `observation_spec`， `action_spec`， `close`等抽象方法

成员变量：
- 继承自`_CommonEnvironment类`的成员变量`_task`, `_time_limit`, `_physics`, `_random_state`, `_observation_updater`, `_hooks`, `_legacy_step`
- `_max_reset_attempts` 重置模拟器的最大尝试次数
- `_reset_next_step` 是否在下一次step时重置模拟器，`默认true`

成员方法：
- `reset()` 重置模拟器，间接调用`_reset_attempt()`方法, 返回一个`dm_env.TimeStep对象`
- `_reset_attempt()` 重置模拟器，返回一个`dm_env.TimeStep对象`
- `_step_spec()` 返回一个dm_env.TimeStep对象，此方法已经弃用
- `step(action)` 传入一个action，返回一个`dm_env.TimeStep对象`
    - 检查_reset_next_step如果是true，调用reset()方法重置模拟器，然后将_reset_next_step设置为false
    - 调用成员变量_hooks的_before_step(physics, action, random_state)方法
    - 调用成员变量_observation_updater的prepare_for_next_control_step()方法
    - 调用成员方法_substep(action)_n_sub_steps次，调用成员变量_observation_updater的update()方法_n_sub_steps-1次
    - 调用成员变量_hooks的_after_step(physics, random_state)方法
    - 调用成员变量_observation_updater的update()方法, 把缺的那一帧的观察值返回
    - 调用成员变量_task的get_reward(physics)方法，返回本次step标量奖励
    - 调用成员变量_task的get_discount(physics)方法，返回本次step标量折扣
    - 调用成员变量_task的should_terminate_episode(physics)方法，返回一个布尔值，是否满足终止本episode的条件同时与_time_limit成员变量的值进行比较，得到终止的标志布尔值
    - 调用成员变量_observation_updater的get_observation()方法，返回最终本次step的观察值
    - 把前面返回的各种值打包成一个dm_env.TimeStep对象返回，注意如果terminating的值是true，那么这个TimeStep对象是MID状态，否则是LAST状态
    - * 这里还有一个检查模拟器物理状态是否连续的机制，如果不连续，那么会抛出异常，将奖励和折扣设置为0，terminating标志设置为true，然后返回一个LAST状态的dm_env.TimeStep对象，退出episode。
- `_substep(action)` 将hook中的`before_substep`回调函数, 成员变量_physics的`step()`方法，hook中的`after_substep`回调函数包装到一起使用
- `action_spec()` 间接调用成员变量`_task`的`action_spec(physics)`方法返回一个`dm_env.specs.BoundedArraySpec对象`，表示action的规格
- `reward_spec()` 间接调用成员变量`_task`中实现了`get_reward_spec()`方法, 返回非标量的reward，如果没有定义，则调用本对象`父类dm_env.Environment`的`reward_spec()`方法，返回一个`specs.Array(shape=(), dtype=float, name='reward')`
- `discount_spec()` 间接调用成员变量`_task`中实现了`get_discount_spec()`方法, 返回非标量的discount，如果没有定义，则调用本对象`父类dm_env.Environment`的`discount_spec()`方法，返回一个`specs.BoundedArray(shape=(), dtype=float, minimum=0., maximum=1., name='discount')`
- `observation_spec()` 间接调用成员变量`_observation_updater`的`observation_spec()`方法返回一个`Array` or `BoundedArray` spec类型的对象，表示观察值的规格