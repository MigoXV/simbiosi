# import logging
# import os
# from pathlib import Path

# import typer

# app = typer.Typer()
# logger = logging.getLogger(__name__)


# @app.command()
# def scan_dir(
#     input_dir: Path = typer.Argument(
#         os.getenv("INPUT_DIR"), help="The directory to scan"
#     ),
#     clean_output_dir: Path = typer.Argument(
#         os.getenv("CLEAN_OUTPUT_DIR"), help="The directory to save the results"
#     ),
#     noise_output_dir: Path = typer.Argument(
#         os.getenv("NOISE_OUTPUT_DIR"),
#         help="The directory to save the noise results",
#     ),
#     expand: float = typer.Option(
#         0, "-e", help="Expand audio to a certain duration(Unit: hour)"
#     ),
# ):
#     from notturno.infer.scan_dir import scan_dir
#     from notturno.infer.utils.expand_audio import expand_audio

#     if not input_dir:
#         raise typer.BadParameter(
#             "You must specify an input directory or set the INPUT_DIR environment variable"
#         )
#     if clean_output_dir is None:
#         clean_output_dir = input_dir.parent / (input_dir.name + "-cleaned")
#     if noise_output_dir is None:
#         noise_output_dir = input_dir.parent / (input_dir.name + "-noise")
#     logger.info(
#         f"Scanning {input_dir} and saving the results in {clean_output_dir} and {noise_output_dir}"
#     )
#     if expand > 0:
#         clean_tmp_dir = clean_output_dir.parent / (clean_output_dir.name + "-tmp")
#         noise_tmp_dir = noise_output_dir.parent / (noise_output_dir.name + "-tmp")
#         # scan_dir(input_dir, clean_tmp_dir, noise_output_dir)
#         scan_dir(input_dir, clean_tmp_dir, noise_tmp_dir)
#         logger.info(f"Expanding output to {expand} hours")
#         expand_audio(clean_tmp_dir, clean_output_dir, expected_total_duration_h=expand)
#         expand_audio(noise_tmp_dir, noise_output_dir, expected_total_duration_h=expand)
#     else:
#         scan_dir(input_dir, clean_output_dir, noise_output_dir)


# def main():
#     app()


# if __name__ == "__main__":
#     main()
