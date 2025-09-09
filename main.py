import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_bool('use_wandb', False, "use wandb")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        if self.wandb_logger is not None:
            self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3D

# Fixed colors for plotting/animation
EE_COLOR = 'tab:blue'     # End-effector trajectory color
BLK_COLOR = 'tab:orange'  # Block trajectory color
CURSOR_COLOR = 'tab:red'  # Leading cursor color

def set_equal_3d(ax):
    # Equal scaling for 3D axes
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ranges = [xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]]
    centers = [np.mean(xlim), np.mean(ylim), np.mean(zlim)]
    r = max(ranges) / 2.0
    ax.set_xlim3d([centers[0]-r, centers[0]+r])
    ax.set_ylim3d([centers[1]-r, centers[1]+r])
    ax.set_zlim3d([centers[2]-r, centers[2]+r])

def _offset3d(scatter, x, y, z):
    scatter._offsets3d = (np.array([x]), np.array([y]), np.array([z]))

def _split_into_episodes(terminals):
    """
    Split dataset into episodes based on terminals array.
    Returns list of episode index ranges [start, end).
    """
    T = len(terminals)
    if T == 0:
        return []

    starts = [0]
    ends = []

    for t in range(T):
        if terminals[t] == 1.0 or terminals[t] == True:
            ends.append(t+1)  # inclusive end
            if t+1 < T:
                starts.append(t+1)  # next starts after terminal

    if len(ends) < len(starts):
        ends.append(T)

    return list(zip(starts, ends))

def _get_default_indices(observations, ee_cols, blk_cols, yaw_col):
    """Auto-detect default column indices based on observation width if not provided.
    Layout for width >= 27 (per user mapping):
      12:15 -> effector_pos (x,y,z)
      19:22 -> block_0_pos (x,y,z)
      26    -> block_0_yaw
    Fallback for small vectors: (0,1,2) and (3,4,5) with no yaw.
    """
    D = observations.shape[-1]
    default_ee = (12, 13, 14) if D >= 27 else (0, 1, 2)
    default_blk = (19, 20, 21) if D >= 27 else (3, 4, 5)
    default_yaw = 26 if D >= 27 else None
    ee_cols = ee_cols if ee_cols is not None else default_ee
    blk_cols = blk_cols if blk_cols is not None else default_blk
    yaw_col = yaw_col if yaw_col is not None else default_yaw
    return ee_cols, blk_cols, yaw_col

def extract_from_dataset(observations, ee_cols=None, blk_cols=None, yaw_col=None, block_index=0):
    """
    Extract EE and block trajectories from dataset observations using the provided
    column indices, or auto-detected defaults if None.
    Returns (ee_xyz, blk_xyz, blk_yaw_or_empty)
    """
    D = observations.shape[-1]
    # Detect cos/sin yaw layout: 19 + 9*k per block
    if ee_cols is None and blk_cols is None and yaw_col is None and D >= 19 and (D - 19) % 9 == 0:
        ee_xyz = observations[:, 12:15].astype(float)
        base = 19 + 9 * int(block_index)
        if base + 9 > D:
            raise ValueError(f"block_index {block_index} out of bounds for observation width {D}")
        blk_xyz = observations[:, base:base+3].astype(float)
        cos_yaw = observations[:, base+7].astype(float)
        sin_yaw = observations[:, base+8].astype(float)
        blk_yaw = np.arctan2(sin_yaw, cos_yaw).astype(float)
        return ee_xyz, blk_xyz, blk_yaw

    ee_cols, blk_cols, yaw_col = _get_default_indices(observations, ee_cols, blk_cols, yaw_col)
    ee_xyz = observations[:, ee_cols].astype(float)
    blk_xyz = observations[:, blk_cols].astype(float)
    if yaw_col is None:
        blk_yaw = np.empty((len(observations),), dtype=float)
    else:
        blk_yaw = observations[:, yaw_col].astype(float)
    return ee_xyz, blk_xyz, blk_yaw

def build_episode_segments(
    dataset,
    ee_cols=None, blk_cols=None, yaw_col=None,
    stride=1
):
    """
    Build per-episode segments (ee_xyz, blk_xyz, blk_yaw) with subsampling and optional frame/episode caps.
    Returns (parsed_segments, total_frames, num_episodes_considered).
    """
    observations = dataset['observations']
    terminals = dataset['terminals']

    episodes = _split_into_episodes(terminals)
    parsed = []
    total_frames = 0
    for s, e in episodes:
        obs_seg = observations[s:e]
        if len(obs_seg) < 2:
            continue
        step = max(1, stride)

        XYZ_CENTER = np.array([0.425, 0.0, 0.0])
        XYZ_SCALER = 10.0
        ee_xyz, blk_xyz, blk_yaw = extract_from_dataset(obs_seg, ee_cols, blk_cols, yaw_col)
        ee_xyz = ee_xyz / XYZ_SCALER + XYZ_CENTER
        blk_xyz = blk_xyz / XYZ_SCALER + XYZ_CENTER
        ee_xyz = ee_xyz[::step]
        blk_xyz = blk_xyz[::step]
        blk_yaw = blk_yaw[::step]

        parsed.append((ee_xyz, blk_xyz, blk_yaw))
        total_frames += len(ee_xyz)

    return parsed, total_frames, len(episodes)

def set_axes_limits_from_parsed(ax, parsed):
    """Set axis limits from a list of (ee_xyz, blk_xyz, blk_yaw) segments and equalize scale."""
    if len(parsed) == 0:
        return
    all_ee = np.concatenate([p[0] for p in parsed], axis=0)
    all_blk = np.concatenate([p[1] for p in parsed], axis=0)
    all_xyz = np.vstack([all_ee, all_blk])
    ax.set_xlim(all_xyz[:,0].min(), all_xyz[:,0].max())
    ax.set_ylim(all_xyz[:,1].min(), all_xyz[:,1].max())
    ax.set_zlim(all_xyz[:,2].min(), all_xyz[:,2].max())
    set_equal_3d(ax)

def compute_success_rate(dataset):
    """
    Compute success rate as successful_episodes / total_episodes.
    - Episodes are split using `terminals` (1 at end of episode).
    - An episode is considered successful if any step within it has mask == 0,
      per the definition that mask == 0 only when the task is complete.
    Returns a float in [0,1].
    """
    masks = np.asarray(dataset['masks']).astype(float)
    terminals = np.asarray(dataset['terminals']).astype(float)
    episodes = _split_into_episodes(terminals)
    total_episodes = len(episodes)
    if total_episodes == 0:
        return 0.0
    successful = 0
    for s, e in episodes:
        if np.any(masks[s:e] == 0.0):
            successful += 1
    return successful / float(total_episodes)

def plot_trajectories(
    dataset,
    ee_cols=None, blk_cols=None, yaw_col=None,
    title="EE & Block Trajectories",
    save_dir=".", png_name="ee_block_trajectories.png",
    stride=1, max_episodes=None
):
    """
    Static plot: handles dataset with observations, terminals, etc.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    parsed, _, num_eps = build_episode_segments(
        dataset,
        ee_cols=ee_cols, blk_cols=blk_cols, yaw_col=yaw_col,
        stride=stride
    )
    print(f"Found {num_eps} episodes")

    for ee_xyz, blk_xyz, blk_yaw in tqdm.tqdm(parsed):
        # EE in blue; Block in orange; leading EE head in red
        ax.plot(ee_xyz[:,0], ee_xyz[:,1], ee_xyz[:,2], color=EE_COLOR, alpha=0.9)

    set_axes_limits_from_parsed(ax, parsed)
    plt.tight_layout()
    out_png = os.path.join(save_dir, png_name)
    plt.savefig(out_png)
    plt.close(fig)
    return out_png

def animate_trajectories(
    dataset,
    ee_cols=None, blk_cols=None, yaw_col=None,
    show_block_yaw=True,
    title="EE & Block Trajectories (Animated)",
    save_dir=".",
    out_name="ee_block_trajectories",
    fps=30, dpi=75,  # Reduced from 150 for faster rendering
    stride=20,       # Increased from 10 to reduce frames
):
    """
    Animated video from dataset with observations, terminals, etc.
    - Subsamples frames by `stride`
    - Uses terminals to split episodes properly.
    """
    os.makedirs(save_dir, exist_ok=True)

    parsed, total_frames, num_eps = build_episode_segments(
        dataset,
        ee_cols=ee_cols, blk_cols=blk_cols, yaw_col=yaw_col,
        stride=stride
    )
    print(f"Found {num_eps} episodes, animating first {len(parsed)}")

    # Compute per-episode success flags aligned with parsed episodes
    terminals = np.asarray(dataset['terminals']).astype(float)
    masks = np.asarray(dataset['masks']).astype(float)
    episodes_full = _split_into_episodes(terminals)

    successes = []
    for s, e in episodes_full:
        if e - s < 2:
            continue
        # Episode considered included → compute success (mask == 0 anywhere in [s:e))
        successes.append(bool(np.any(masks[s:e] == 0.0)))

    if total_frames < 2:
        raise ValueError("Not enough frames to animate. Check your indices/episode split/stride.")

    fig = plt.figure(figsize=(9,7), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    # Pre-create artists for each episode (render sequentially across frames)
    artists = []
    for i, (ee_xyz, blk_xyz, blk_yaw) in enumerate(parsed):
        ee_line, = ax.plot([], [], [], lw=2, color=EE_COLOR, alpha=0.9)
        blk_line, = ax.plot([], [], [], lw=2, linestyle='--', color=BLK_COLOR, alpha=0.9)
        ee_start = ax.scatter(ee_xyz[0,0], ee_xyz[0,1], ee_xyz[0,2], marker='o', color=EE_COLOR, s=60)
        ee_head  = ax.scatter(ee_xyz[0,0], ee_xyz[0,1], ee_xyz[0,2], marker='^', color=CURSOR_COLOR, s=70)
        blk_start= ax.scatter(blk_xyz[0,0], blk_xyz[0,1], blk_xyz[0,2], marker='s', color=BLK_COLOR, s=60)

        yaw_line = None
        if show_block_yaw and yaw_col is not None:
            yaw_line = Line3D([], [], [], color=BLK_COLOR, alpha=0.7)
            ax.add_line(yaw_line)

        artists.append({
            "ee_line": ee_line, "blk_line": blk_line,
            "ee_start": ee_start, "ee_head": ee_head, "blk_start": blk_start,
            "yaw_line": yaw_line, "ee_xyz": ee_xyz, "blk_xyz": blk_xyz, "blk_yaw": blk_yaw
        })

    # Compute global limits from parsed
    set_axes_limits_from_parsed(ax, parsed)
    # Episode status overlay (top-left inside axes)
    status_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, ha='left', va='top')
    plt.tight_layout()

    # Flatten (ep, t) into a single global timeline
    ep_lengths = [len(a["ee_xyz"]) for a in artists]
    ep_offsets = np.cumsum([0] + ep_lengths[:-1])

    def _map_frame_to_ep_t(f):
        # f in [0, total_frames-1]
        # find ep idx such that f < ep_offsets[i] + ep_lengths[i]
        i = np.searchsorted(ep_offsets, f+1, side="right") - 1
        t = f - ep_offsets[i]
        return i, t

    def init():
        changed = []
        for a in artists:
            a["ee_line"].set_data([], []); a["ee_line"].set_3d_properties([])
            a["blk_line"].set_data([], []); a["blk_line"].set_3d_properties([])
            if a["yaw_line"] is not None:
                a["yaw_line"].set_data([], []); a["yaw_line"].set_3d_properties([])
                a["yaw_line"].set_visible(False)
            a["ee_start"].set_visible(False)
            a["blk_start"].set_visible(False)
            a["ee_head"].set_visible(False)
            changed.extend([a["ee_line"], a["blk_line"], a["ee_start"], a["blk_start"], a["ee_head"]])
            if a["yaw_line"] is not None:
                changed.append(a["yaw_line"])
        status_text.set_text("")
        return changed

    # Pre-compute yaw data to avoid repeated calculations
    if show_block_yaw and yaw_col is not None:
        L = 0.02
        for a in artists:
            if len(a["blk_yaw"]) > 0:
                # Pre-compute yaw line endpoints for all frames
                yaw_data = []
                for t in range(len(a["blk_xyz"])):
                    x0, y0, z0 = a["blk_xyz"][t]
                    dx = L * np.cos(a["blk_yaw"][t])
                    dy = L * np.sin(a["blk_yaw"][t])
                    yaw_data.append(([x0, x0+dx], [y0, y0+dy], [z0, z0]))
                a["precomputed_yaw"] = yaw_data

    def update(f):
        i, t = _map_frame_to_ep_t(f)

        # Show only the current episode; hide others entirely
        for j in range(len(artists)):
            aj = artists[j]
            if j == i:
                # Current episode: show up to current time t
                a = artists[i]
                ee, blk = a["ee_xyz"], a["blk_xyz"]
                a["ee_line"].set_data(ee[:t+1,0], ee[:t+1,1]); a["ee_line"].set_3d_properties(ee[:t+1,2])
                a["blk_line"].set_data(blk[:t+1,0], blk[:t+1,1]); a["blk_line"].set_3d_properties(blk[:t+1,2])
                a["ee_start"].set_visible(True)
                a["blk_start"].set_visible(True)
                a["ee_head"].set_visible(True)
                _offset3d(a["ee_head"], ee[t,0], ee[t,1], ee[t,2])
                if a["yaw_line"] is not None and len(a["blk_yaw"]) > 0 and t < len(a["precomputed_yaw"]):
                    # Use precomputed data for current yaw position
                    yaw_x, yaw_y, yaw_z = a["precomputed_yaw"][t]
                    a["yaw_line"].set_data(yaw_x, yaw_y); a["yaw_line"].set_3d_properties(yaw_z)
                    a["yaw_line"].set_visible(True)
            
            else:
                # Hide non-current episodes completely
                aj["ee_line"].set_data([], []); aj["ee_line"].set_3d_properties([])
                aj["blk_line"].set_data([], []); aj["blk_line"].set_3d_properties([])
                if aj["yaw_line"] is not None:
                    aj["yaw_line"].set_data([], []); aj["yaw_line"].set_3d_properties([])
                    aj["yaw_line"].set_visible(False)
                aj["ee_start"].set_visible(False)
                aj["blk_start"].set_visible(False)
                aj["ee_head"].set_visible(False)

        # Update episode status text
        if 0 <= i < len(successes):
            status = "Success" if successes[i] else "Failure"
            status_text.set_text(f"Episode {i+1}/{len(successes)}: {status}")
        else:
            status_text.set_text("")

        # Return all artists that might have changed
        return []

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=total_frames, interval=1000.0/fps,
        blit=False, repeat=False, save_count=total_frames
    )

    base = os.path.join(save_dir, out_name)

    # Use faster writer settings
    writer = FFMpegWriter(fps=fps,
                         extra_args=['-preset', 'fast', '-crf', '28'],  # Faster encoding
                         metadata={'title': out_name})
    path = base + ".mp4"
    anim.save(path, writer=writer)
    plt.close(fig)
    return path


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    if FLAGS.use_wandb:
        run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    else:
        run = None
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb if FLAGS.use_wandb else None,
    )

    print("success rate ", compute_success_rate(train_dataset))
    animate_trajectories(train_dataset)
    # plot_trajectories(train_dataset)
    return

    offline_init_time = time.time()
    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        breakpoint()
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    if FLAGS.use_wandb:
        with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
            f.write(run.url)

if __name__ == '__main__':
    app.run(main)
