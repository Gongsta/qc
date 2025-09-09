import numpy as np
import matplotlib.pyplot as plt

# ---- index map for raw next_ob (length = 27) ----
IDX = {
    "ee_pos": slice(12, 15),    # x,y,z
    "ee_yaw": 15,
    "blk_pos": slice(19, 22),   # x,y,z
    "blk_yaw": 26,
}

def as_2d(array_like):
    arr = np.asarray(array_like)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def extract_from_raw_traj(raw_traj):
    """
    raw_traj: (T, 27) array (or list of 27-vectors).
    Returns:
      ee_xyz:  (T, 3)
      blk_xyz: (T, 3)
      blk_yaw: (T,)
    """
    R = as_2d(raw_traj)
    ee_xyz  = R[:, IDX["ee_pos"]]
    blk_xyz = R[:, IDX["blk_pos"]]
    blk_yaw = R[:, IDX["blk_yaw"]]
    return ee_xyz, blk_xyz, blk_yaw

def set_equal_3d(ax):
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xr = xlim[1]-xlim[0]; yr = ylim[1]-ylim[0]; zr = zlim[1]-zlim[0]
    r = max(xr, yr, zr)
    xm = np.mean(xlim); ym = np.mean(ylim); zm = np.mean(zlim)
    ax.set_xlim3d([xm-r/2, xm+r/2])
    ax.set_ylim3d([ym-r/2, ym+r/2])
    ax.set_zlim3d([zm-r/2, zm+r/2])

def plot_ee_and_block_trajectories(raw_trajectories, show_block_yaw=True, title="EE & Block Trajectories"):
    """
    raw_trajectories: List of trajectories, where each trajectory is:
        - np.ndarray of shape (T, 27), or
        - list of length T with each element being a (27,) array-like.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    for i, traj in enumerate(raw_trajectories):
        ee_xyz, blk_xyz, blk_yaw = extract_from_raw_traj(traj)

        # EE path
        ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], label=f"EE traj {i}")
        ax.scatter(ee_xyz[0, 0],  ee_xyz[0, 1],  ee_xyz[0, 2],  marker='o')  # start
        ax.scatter(ee_xyz[-1, 0], ee_xyz[-1, 1], ee_xyz[-1, 2], marker='^')  # end

        # Block path
        ax.plot(blk_xyz[:, 0], blk_xyz[:, 1], blk_xyz[:, 2])
        ax.scatter(blk_xyz[0, 0], blk_xyz[0, 1], blk_xyz[0, 2], marker='s')

        # Optional yaw “ticks” on XY plane to indicate heading
        if show_block_yaw and len(blk_yaw) > 0:
            idxs = np.linspace(0, len(blk_xyz) - 1, num=min(15, len(blk_xyz)), dtype=int)
            L = 0.02  # tick length; adjust to your scene scale
            for k in idxs:
                x0, y0, z0 = blk_xyz[k]
                dx = L * np.cos(blk_yaw[k])
                dy = L * np.sin(blk_yaw[k])
                ax.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0])  # short line on XY

    ax.legend(loc='upper right')
    set_equal_3d(ax)
    plt.tight_layout()
    plt.show()
