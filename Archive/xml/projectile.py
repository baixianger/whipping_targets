import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os


xml_path = 'scene.xml'                                  # import scene
simend = 30                                             # simulation time
print_camera_config = 1                                 # set to 1 to print camera config
                                                        # this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# random select a object to throw
# id = np.random.randint(0, 9)
# bID = f'{id:02}'
# jID = f'obj{bID}'

def init_controller(scene, data):
    # initialize the controller here. This function is called once, in the beginning
    # initialize the beginning position of the projectile here
    # randomize body position, but keep it with in the range of the room 4 meters
    # and hight 1.5-1.8 which is the height range when the object is thrown
    scene.opt.gravity[-1] = - 0.1
    ids = [i for i in range(9)]
    for id in ids:
        bID = f'{id:02}'
        jID = f'obj{bID}'

        radius = 4
        x0 = np.random.uniform(-radius, radius)
        y0 = (radius**2 - x0**2)**0.5 * np.random.choice([-1, 1])
        z0 = np.random.uniform(0, .5)
        theta = -np.arctan(y0 / x0)
        scene.body("00").pos = np.array([x0, y0, z0])

        # through the object to 6-8 meters away
        g = -scene.opt.gravity[-1] # gravity
        dist = np.random.uniform(6, 8)
        mass = scene.body("00").mass[0]
        pitch = np.random.uniform(30, 45) * np.pi / 180 # ptich angle is between 30-45 degree
        yaw = np.arctan(0.5 / radius)
        yaw = np.random.uniform(-yaw, yaw)
        yaw = theta # + yaw
        # 计算初始速度
        v0 = (g * dist / np.sin(2 * pitch)) ** 0.5
        vx = v0 * np.cos(pitch) * np.cos(yaw)
        vy = v0 * np.cos(pitch) * np.sin(yaw)
        vz = v0 * np.sin(pitch)

        data.joint(jID).qpos = np.array([x0, y0, z0, 1, 0, 0, 0])
        data.joint(jID).qvel = np.array([vx, vy, vz, 0, 0.1, 0.1])

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass
    # ids = [i for i in range(9)]
    # for id in ids:
    #     bID = f'{id:02}'
    #     jID = f'obj{bID}'
    #     # Force = -c*vx*|v| i + -c*vy*|v| j + -c*vz*|v| k
    #     vx = data.qvel[0];
    #     vy = data.qvel[1];
    #     vz = data.qvel[2];
    #     v = np.sqrt(vx**2+vy**2+vz**2)
    #     c = 0.5
    #     # data.qfrc_applied[0] = -c*vx*v;
    #     # data.qfrc_applied[1] = -c*vy*v;
    #     # data.qfrc_applied[2] = -c*vz*v;
    #     # data.xfrc_applied[1][0] = -c*vx*v;
    #     # data.xfrc_applied[1][1] = -c*vy*v;
    #     # data.xfrc_applied[1][2] = -c*vz*v;
    #     data.joint(jID).qfrc_applied = np.array([-c*vx*v, -c*vy*v, -c*vz*v, 0, 0, 0])
    #     # data.body("00").xfrc_applied = np.array([-c*vx*v, -c*vy*v, -c*vz*v, 0, 0, 0])



def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                     # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90 ; cam.elevation = -22 ; cam.distance =  10
# cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])
cam.azimuth = 90 ; cam.elevation = -89 ; cam.distance = 11
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])


#initialize the controller
init_controller(model,data)


#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
