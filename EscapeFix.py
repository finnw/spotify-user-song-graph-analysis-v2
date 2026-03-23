from collections.abc import Iterable, Iterator


class EscapeFixer:
    """
    Fixes unescaped double quotes in CSV lines.
    This is a workaround for the fact that some song titles in the dataset contain unescaped double quotes, which causes issues with CSV parsing.
    The fix is to replace unescaped double quotes with a placeholder, and then restore them after parsing. This is not a general solution, but it works for the specific dataset we are working with.
    """
    def __init__(self, source: Iterable[bytes]):
        self.source = source

    @staticmethod
    def fix_line(line_str: str) -> str:
        if line_str.count('"') != 8:
            # Song titles with double quotes cause CSV parsing issues as they have not been escaped.
            # Re-encode them to make the CSV parser happy.
            # This will not work in the general case, but is sufficient for this dataset.
            line_str = line_str[1:-2]  # Remove leading and trailing quotes
            line_str = line_str.replace('@', '@0')
            line_str = line_str.replace('","', '@1') # Unambiguously mark the field separators
            line_str = line_str.replace('"', '@2')   # Remaining quotes are part of the title
            # Reassemble the line but double up any quotes in the title
            line_str = line_str.replace('@2', '""')
            line_str = line_str.replace('@1', '","')
            line_str = line_str.replace('@0', '@')
            line_str = f'"{line_str}"'  # Restore leading and trailing quotes
        return line_str

    def __iter__(self) -> Iterator[str]:
        line: bytes
        line_str: str
        for line in self.source:
            line_str = line.decode('utf-8')
            yield EscapeFixer.fix_line(line_str)
