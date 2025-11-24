import numpy as np
import matplotlib.pyplot as plt


class visualization:
    def __init__(self):
        pass
        
    def trajectory_2d(self,
                      x,
                      z,
                      save_figure=False,
                      filename='trajectory_2d.png'
        ):

        plt.figure()
        plt.plot(x, z, marker='o', linestyle='-')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('X vs Z Endpoint Trajectory')
        plt.grid(True)
        plt.show()
        if save_figure:
            plt.savefig(filename)
        
    def trajectory_3d(self,
                      pts,
                      save_figure=False,
                      filename='trajectory_3d.png'
        ):
        pts = np.asarray(pts)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', linewidth=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory')
        if save_figure:
            plt.savefig(filename)
            
    def current_shape_3d(self,
                        shearable_rod,
                        save_figure=False,
                        filename='current_shape.png'
        ):
        from matplotlib.colors import to_rgb

        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(
            shearable_rod.position_collection[0, ...],
            shearable_rod.position_collection[2, ...],
            shearable_rod.position_collection[1, ...],
            c=to_rgb("xkcd:bluish"),
            label="n=" + str(shearable_rod.n_elems),
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        
        # Set equal aspect ratio for proper scaling
        # Get current axis limits
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        zlims = ax.get_zlim()
        
        # Calculate the range for each axis
        x_range = xlims[1] - xlims[0]
        y_range = ylims[1] - ylims[0]
        z_range = zlims[1] - zlims[0]
        
        # Set equal aspect ratio based on the largest range
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
        
        plt.show()
        if save_figure:
            plt.savefig(filename)
        
    def current_shape_2d(self,
                        shearable_rod,
                        save_figure=False,
                        filename='current_shape.png'
        ):
        from matplotlib.colors import to_rgb

        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111)

        ax.plot(
            shearable_rod.position_collection[2, ...],
            shearable_rod.position_collection[0, ...],
            c=to_rgb("xkcd:bluish"),
            label="n=" + str(shearable_rod.n_elems),
        )
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        plt.show()
        if save_figure:
            plt.savefig(filename)

        
    def velocity_norm_history(self,
                              time: np.ndarray,
                              velocity: np.ndarray,  # shape should be (N, 3)
                              save_figure=False,
                              filename='velocity_norm_history.png'
        ):
        velocity_norm = np.linalg.norm(velocity, axis=1)
        plt.figure()
        plt.plot(time, velocity_norm)
        plt.xlabel('Time')
        plt.ylabel('Velocity Norm')
        plt.title('Velocity Norm History')
        if save_figure:
            plt.savefig(filename)        
        
        
    def plot_video(
        self,
        plot_params: dict,
        video_name: str = "video.mp4",
        fps: int = 60,
        xlim: tuple[float, float] = (0, 4),
        ylim: tuple[float, float] = (-1, 1),
    ) -> None:  # (time step, x/y/z, node)
        import matplotlib.animation as manimation
        from tqdm import tqdm
        positions_over_time = np.array(plot_params["position"])

        num_frames = len(plot_params["time"])
        duration = num_frames / fps
        print(f"Plotting video: {num_frames} frames, {duration:.2f} seconds duration")
        
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z [m]", fontsize=16)
        ax.set_ylabel("x [m]", fontsize=16)
        rod_lines_2d = ax.plot(positions_over_time[0][2], positions_over_time[0][0])[0]
        # plt.axis("equal")
        with writer.saving(fig, video_name, dpi=150):
            for time in tqdm(range(1, len(plot_params["time"]))):
                rod_lines_2d.set_xdata([positions_over_time[time][2]])
                rod_lines_2d.set_ydata([positions_over_time[time][0]])
                writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

    def plot_video_limited(
        self,
        plot_params: dict,
        video_name: str = "video_limited.mp4",
        max_samples: int = 3000,
        xlim: tuple[float, float] = (0, 4),
        ylim: tuple[float, float] = (-1, 1),
    ) -> None:
        import matplotlib.animation as manimation
        from tqdm import tqdm

        times = np.asarray(plot_params["time"])
        positions_over_time = np.asarray(plot_params["position"])

        if times.ndim == 0 or len(times) == 0:
            raise ValueError("plot_params['time'] must contain at least one entry.")
        if positions_over_time.shape[0] != len(times):
            raise ValueError(
                "First dimension of plot_params['position'] must match length of plot_params['time']."
            )

        total_available = len(times)
        if total_available <= max_samples:
            indices = np.arange(total_available, dtype=int)
        else:
            indices = np.linspace(0, total_available - 1, max_samples)
            indices = np.round(indices).astype(int)
            indices[0] = 0
            indices[-1] = total_available - 1
            indices = np.unique(indices)
            if indices[0] != 0:
                indices = np.insert(indices, 0, 0)
            if indices[-1] != total_available - 1:
                indices = np.append(indices, total_available - 1)

        subsampled_times = times[indices]
        subsampled_positions = positions_over_time[indices]
        num_frames = len(subsampled_times)

        if num_frames == 0:
            raise ValueError("No frames selected for plotting.")

        total_time = float(subsampled_times[-1] - subsampled_times[0])
        if total_time > 0.0 and num_frames > 0:
            computed_fps = num_frames / total_time
        else:
            computed_fps = 1.0

        if computed_fps <= 0.0:
            computed_fps = 1.0

        duration = num_frames / computed_fps
        print(
            f"Plotting limited video: using {num_frames} of {total_available} frames, "
            f"{computed_fps:.2f} fps, {duration:.2f} seconds duration"
        )

        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
        writer = FFMpegWriter(fps=computed_fps, metadata=metadata)
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z [m]", fontsize=16)
        ax.set_ylabel("x [m]", fontsize=16)
        rod_lines_2d = ax.plot(
            subsampled_positions[0][2], subsampled_positions[0][0]
        )[0]
        with writer.saving(fig, video_name, dpi=150):
            for frame_idx in tqdm(range(1, num_frames)):
                rod_lines_2d.set_xdata([subsampled_positions[frame_idx][2]])
                rod_lines_2d.set_ydata([subsampled_positions[frame_idx][0]])
                writer.grab_frame()

        plt.close(plt.gcf())

    def plot_video_auto_limits(
        self,
        plot_params: dict,
        video_name: str = "video_auto_limits.mp4",
        max_samples: int = 3000,
        padding_ratio: float = 0.05,
    ) -> None:
        positions_over_time = np.asarray(plot_params["position"])

        if positions_over_time.shape[0] == 0:
            raise ValueError("plot_params['position'] must contain at least one entry.")

        if positions_over_time.ndim < 2:
            raise ValueError("plot_params['position'] must have at least two dimensions.")

        if positions_over_time.shape[1] == 3:
            oriented_positions = positions_over_time
        elif positions_over_time.shape[-1] == 3:
            oriented_positions = np.moveaxis(positions_over_time, -1, 1)
        else:
            raise ValueError(
                "plot_params['position'] must include a coordinate axis of length 3."
            )

        num_frames = oriented_positions.shape[0]
        reshaped_positions = oriented_positions.reshape(num_frames, 3, -1)
        z_values = reshaped_positions[:, 2, :].ravel()
        x_values = reshaped_positions[:, 0, :].ravel()

        finite_mask = np.isfinite(z_values) & np.isfinite(x_values)
        if not np.any(finite_mask):
            raise ValueError(
                "plot_params['position'] must contain finite values for x and z coordinates."
            )

        z_values = z_values[finite_mask]
        x_values = x_values[finite_mask]

        z_min = float(z_values.min())
        z_max = float(z_values.max())
        x_min = float(x_values.min())
        x_max = float(x_values.max())

        if np.isclose(z_min, z_max):
            z_center = 0.5 * (z_min + z_max)
            z_half_range = max(abs(z_center) * padding_ratio, 1e-3)
            xlim = (z_center - z_half_range, z_center + z_half_range)
        else:
            padding_z = padding_ratio * (z_max - z_min)
            xlim = (z_min - padding_z, z_max + padding_z)

        if np.isclose(x_min, x_max):
            x_center = 0.5 * (x_min + x_max)
            x_half_range = max(abs(x_center) * padding_ratio, 1e-3)
            ylim = (x_center - x_half_range, x_center + x_half_range)
        else:
            padding_x = padding_ratio * (x_max - x_min)
            ylim = (x_min - padding_x, x_max + padding_x)

        self.plot_video_limited(
            plot_params=plot_params,
            video_name=video_name,
            max_samples=max_samples,
            xlim=xlim,
            ylim=ylim,
        )

