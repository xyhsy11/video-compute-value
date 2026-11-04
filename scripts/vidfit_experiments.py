"""Launch the project-local experiment browser using VidFit's optional UI runtime."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_compute_value.vidfit_view import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
