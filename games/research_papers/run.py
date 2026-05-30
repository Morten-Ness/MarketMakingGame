from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from games.research_papers.cli import main


RESEARCH_SUBJECT = "attention"
# Options: "attention", "embeddings"


if __name__ == "__main__":
    raise SystemExit(main(research_subject=RESEARCH_SUBJECT))
