import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

# ===================== 路径与参数 =====================
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "g1.xml"

GAIT_FREQ = 0.8          # 步态频率 Hz
AMP_HIP_P = 0.30         # 髋关节前后摆幅 (rad)
AMP_KNEE_SWING = 0.80    # 摆动相膝盖弯曲幅值 (rad)
AMP_ANKLE_P = 0.20       # 踝关节俯仰幅值 (rad)
AMP_ARM = 0.25           # 手臂摆动幅值 (rad)
AMP_WAIST_ROLL = 0.03    # 腰部 roll 幅值 (rad)

# 支撑/摆动相占空比：支撑相占 60%
STANCE_RATIO = 0.6

# 足底相对 site 的偏移（你的模型脚底在 site 下方约 0.03 m）
FOOT_OFFSET_Z = 0.03

AUTO_RESET = False


# ===================== 工具函数 =====================
def build_act_map(model):
    act_map = {}
    for i in range(model.nu):
        name = model.actuator(i).name
        if isinstance(name, bytes):
            name = name.decode()
        act_map[name] = i
    return act_map


def get_stand_qpos(model):
    for i in range(model.nkey):
        kn = model.key(i).name
        if isinstance(kn, bytes):
            kn = kn.decode()
        if "stand" in kn.lower():
            return model.key(i).qpos.copy()
    return model.key(0).qpos.copy()


# ===================== 加载模型 =====================
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
act_map = build_act_map(model)
stand_qpos = get_stand_qpos(model)
pelvis_id = model.body("pelvis").id

print("Actuator map keys:", sorted(act_map.keys()))
print(f"Stand qpos len={len(stand_qpos)}, nu={model.nu}")

required_keys = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
]
for key in required_keys:
    assert key in act_map, f"Missing actuator: {key}"


# ===================== 步态生成 =====================
def leg_phase(p):
    """输入相位 p，返回 (是否支撑相, 相位内进度 0~1)"""
    p = np.mod(p, 2 * np.pi)
    stance_end = STANCE_RATIO * 2 * np.pi
    if p < stance_end:
        return True, p / stance_end
    else:
        return False, (p - stance_end) / (2 * np.pi - stance_end)


def compute_ctrl(t):
    phase = 2 * np.pi * GAIT_FREQ * t
    ctrl = np.zeros(model.nu)

    # ---------- 腿部 ----------
    for side, phase_offset in [("left", 0.0), ("right", np.pi)]:
        p = phase + phase_offset
        is_stance, s = leg_phase(p)

        if is_stance:
            # 支撑相：膝盖接近 0（因为站立 keyframe 腿是直的）
            # 髋从前往后摆：s=0 时前摆，s=1 时后摆
            hip_p = AMP_HIP_P * np.cos(np.pi * s)   # +AMP -> -AMP
            knee = 0.02                              # 几乎伸直
            ankle_p = -0.5 * hip_p                   # 保持脚掌水平
        else:
            # 摆动相：膝盖弯曲让脚离地，髋从后往前摆
            hip_p = -AMP_HIP_P * np.cos(np.pi * s)  # -AMP -> +AMP
            knee = AMP_KNEE_SWING * np.sin(np.pi * s)
            ankle_p = AMP_ANKLE_P * np.sin(np.pi * s)

        ctrl[act_map[f"{side}_hip_pitch_joint"]] = hip_p
        ctrl[act_map[f"{side}_knee_joint"]] = knee
        ctrl[act_map[f"{side}_ankle_pitch_joint"]] = ankle_p

    # ---------- 腰部：左右重心转移 ----------
    # 左腿支撑时（phase 在支撑相），重心偏左，waist_roll 给一个小角度
    ctrl[act_map["waist_roll_joint"]] = AMP_WAIST_ROLL * np.sin(phase)
    ctrl[act_map["waist_yaw_joint"]] = 0.0
    ctrl[act_map["waist_pitch_joint"]] = 0.0

    # ---------- 手臂：与对侧腿同相 ----------
    for side, phase_offset in [("left", np.pi), ("right", 0.0)]:
        p = phase + phase_offset
        ctrl[act_map[f"{side}_shoulder_pitch_joint"]] = AMP_ARM * np.sin(p)

    return ctrl


# ===================== 初始化 =====================
data.qpos[:] = stand_qpos
data.qvel[:] = 0
data.ctrl[:] = 0
mujoco.mj_forward(model, data)

fallen = False


def check_fallen():
    global fallen
    z_axis = data.xmat[pelvis_id][6:9]
    if z_axis[2] < 0.3:
        fallen = True


def reset():
    global fallen
    data.qpos[:] = stand_qpos
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    fallen = False


# ===================== 主循环 =====================
print("开始仿真。按 ESC 或关闭窗口退出。")
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 90

    last_print = -1.0
    while viewer.is_running():
        t = data.time

        if AUTO_RESET and fallen:
            reset()
            t = 0.0

        data.ctrl[:] = compute_ctrl(t)
        mujoco.mj_step(model, data)
        check_fallen()

        # 每秒打印一次足底高度和骨盆高度
        if data.time - last_print >= 1.0:
            last_print = data.time
            try:
                lf = data.site("left_foot").xpos
                rf = data.site("right_foot").xpos
                lf_z = lf[2] - FOOT_OFFSET_Z
                rf_z = rf[2] - FOOT_OFFSET_Z
                pz = data.xpos[pelvis_id][2]
                print(f"t={data.time:5.2f}  L_foot_z={lf_z:+.3f}  R_foot_z={rf_z:+.3f}  pelvis_z={pz:.3f}")
            except Exception as e:
                print("print error:", e)

        with viewer.lock():
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.sync()

        time.sleep(model.opt.timestep)