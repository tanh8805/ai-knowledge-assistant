import re

from app.documents.text import TextLoader

# YAML front matter ("---" block at the top of the file) is metadata, not content.
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


class MarkdownLoader(TextLoader):
    def load(self, data: bytes) -> str:
        return FRONT_MATTER.sub("", super().load(data))
