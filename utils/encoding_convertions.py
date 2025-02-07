import re


class gtParser:
    def __init__(self, single_line_data: bool = True, no_sep_tok=True) -> None:
        self.single_line_data = single_line_data
        self.no_sep_tok = no_sep_tok

    def convert(self, src_file: str):
        # read file and get lines

        with open(src_file, "r") as f:
            if self.single_line_data:
                # separate tokens by space
                lines = f.read().split()
                if self.no_sep_tok:
                    for t in lines:
                        if t in [":", "-", ";", ","]:
                            lines.remove(t)
                        if re.match(r"\[", t):
                            lines.remove(t)
            else:
                lines = f.read().splitlines()

        return lines
