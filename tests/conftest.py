import os

# Settings require DATABASE_URL. Tests never connect to it; a placeholder is enough.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
