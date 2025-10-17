from pathlib import Path
import imageio.v2 as imageio


def animate_files(frames_dir: Path, output_filename: Path) -> None:
    """
    given a folder full of png images, make them into an animation and save
    """

    filenames = sorted(frames_dir.glob("*.png"))

    with imageio.get_writer(
        output_filename, mode="I", duration=500, loop=0
    ) as writer:
        for output_filename in filenames:
            image = imageio.imread(output_filename)
            writer.append_data(image)
            # filename.unlink()
