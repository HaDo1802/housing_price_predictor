from pathlib import Path
import sys

# Ensure project root is importable when running in Vercel's /var/task/api context
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_main import app
