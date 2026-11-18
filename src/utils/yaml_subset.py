"""
A small, strict reader for the YAML subset smaLLMs' config files use.

Supported: block mappings and block sequences, plain, single- and double-quoted
scalars on one line, comments, flow sequences and mappings on one line
(``[a, b]``, ``{a: 1}``), and a leading ``---``. Plain scalars resolve the way
PyYAML's ``safe_load`` resolves them (YAML 1.1): null, booleans (including
yes/no/on/off), ints (decimal, octal, hex, binary) and floats.

Anything else (anchors, aliases, tags, block scalars ``|``/``>``, multi-line
scalars, several documents, timestamps, merge keys, tabs in indentation) raises
YAMLSubsetError naming the file and line. The reader never guesses.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple, Union


class YAMLSubsetError(ValueError):
    """Raised for input outside the supported subset, with file and line."""

    def __init__(self, source: str, line: int, message: str) -> None:
        self.source = source
        self.line = line
        self.message = message
        super().__init__(f"{source}, line {line}: {message}")


_BOOL_RE = re.compile(r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$")
_TRUE_WORDS = {"yes", "true", "on"}
_FLOAT_RE = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
        |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_INT_RE = re.compile(
    r"""^(?:[-+]?0b[0-1_]+
        |[-+]?0[0-7_]+
        |[-+]?(?:0|[1-9][0-9_]*)
        |[-+]?0x[0-9a-fA-F_]+
        |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$""",
    re.X,
)
_NULL_RE = re.compile(r"^(?:~|null|Null|NULL|)$")
_TIMESTAMP_RE = re.compile(
    r"""^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]
        |[0-9][0-9][0-9][0-9] -[0-9][0-9]? -[0-9][0-9]?
         (?:[Tt]|[ \t]+)[0-9][0-9]?
         :[0-9][0-9] :[0-9][0-9] (?:\.[0-9]*)?
         (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$""",
    re.X,
)
# Characters a plain scalar may not start with in this subset.
_REFUSED_START = {
    "&": "anchors",
    "*": "aliases",
    "!": "tags",
    "|": "block scalars",
    ">": "block scalars",
    "%": "directives",
    "@": "reserved indicators",
    "`": "reserved indicators",
    "?": "complex mapping keys",
}
_DOUBLE_ESCAPES = {
    "0": "\0", "a": "\x07", "b": "\x08", "t": "\t", "\t": "\t", "n": "\n", "v": "\x0b",
    "f": "\x0c", "r": "\r", "e": "\x1b", " ": " ", '"': '"', "/": "/", "\\": "\\",
    "N": "\x85", "_": "\xa0", "L": " ", "P": " ",
}
_HEX_ESCAPES = {"x": 2, "u": 4, "U": 8}


class _Line(NamedTuple):
    number: int
    indent: int
    text: str  # content after the indentation, comment not yet removed


class _Quoted(str):
    """A scalar that was quoted, so it is never resolved to another type."""


def load(text: str, source: str = "<string>") -> Any:
    """Parse YAML text in the supported subset and return plain Python data."""
    return _Parser(text, source).parse()


def load_file(path: Union[str, Path]) -> Any:
    """Read and parse one YAML file in the supported subset."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        return load(handle.read(), str(path))


def _resolve_plain(value: str, source: str, line: int) -> Any:
    """Resolve a plain scalar like PyYAML's SafeLoader, refusing what it would not."""
    if _NULL_RE.match(value):
        return None
    if _BOOL_RE.match(value):
        return value.lower() in _TRUE_WORDS
    if _FLOAT_RE.match(value):
        number = value.replace("_", "").lower()
        if ":" in number:
            raise YAMLSubsetError(source, line, f"sexagesimal numbers are not supported: {value!r}")
        sign = -1 if number.startswith("-") else 1
        if number[0] in "+-":
            number = number[1:]
        if number == ".inf":
            return sign * float("inf")
        if number == ".nan":
            return float("nan")
        return sign * float(number)
    if _INT_RE.match(value):
        number = value.replace("_", "")
        if ":" in number:
            raise YAMLSubsetError(source, line, f"sexagesimal numbers are not supported: {value!r}")
        sign = -1 if number.startswith("-") else 1
        if number[0] in "+-":
            number = number[1:]
        if number == "0":
            return 0
        if number.startswith("0b"):
            return sign * int(number[2:], 2)
        if number.startswith("0x"):
            return sign * int(number[2:], 16)
        if number.startswith("0"):
            return sign * int(number, 8)
        return sign * int(number)
    if _TIMESTAMP_RE.match(value):
        raise YAMLSubsetError(source, line, f"timestamps are not supported, quote the value: {value!r}")
    if value in ("<<", "="):
        raise YAMLSubsetError(source, line, f"merge and value keys are not supported: {value!r}")
    return value


class _Parser:
    def __init__(self, text: str, source: str) -> None:
        self.source = source
        self.lines: List[_Line] = []
        self.pos = 0
        if text.startswith("﻿"):
            text = text[1:]
        seen_content = False
        for number, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.lstrip(" ")
            if not stripped.strip() or stripped.startswith("#"):
                continue
            if stripped.startswith("\t"):
                raise YAMLSubsetError(source, number, "tabs in indentation are not allowed")
            indent = len(raw) - len(stripped)
            if indent == 0 and (stripped.startswith("---") and stripped[3:4] in ("", " ", "\t")):
                rest = _strip_comment(stripped[3:], source, number).strip()
                if seen_content or rest:
                    raise YAMLSubsetError(source, number, "only one document is supported, with '---' alone on the first line")
                seen_content = True
                continue
            if indent == 0 and (stripped.startswith("...") and stripped[3:4] in ("", " ", "\t")):
                raise YAMLSubsetError(source, number, "document end markers are not supported")
            if stripped.startswith("%"):
                raise YAMLSubsetError(source, number, "directives are not supported")
            seen_content = True
            self.lines.append(_Line(number, indent, stripped.rstrip()))

    # -- helpers ---------------------------------------------------------

    def error(self, line: int, message: str) -> YAMLSubsetError:
        return YAMLSubsetError(self.source, line, message)

    def current(self) -> Optional[_Line]:
        return self.lines[self.pos] if self.pos < len(self.lines) else None

    # -- structure ---------------------------------------------------------

    def parse(self) -> Any:
        first = self.current()
        if first is None:
            return None
        value = self.parse_block(first.indent)
        leftover = self.current()
        if leftover is not None:
            raise self.error(leftover.number, "unexpected content (check the indentation)")
        return value

    def parse_block(self, indent: int) -> Any:
        line = self.current()
        assert line is not None
        if _is_sequence_item(line.text):
            return self.parse_sequence(indent)
        if self.split_key(line) is not None:
            return self.parse_mapping(indent)
        # A lone scalar (a document that is just one value).
        self.pos += 1
        value = self.parse_value_text(line.text, line.number)
        following = self.current()
        if following is not None and following.indent > indent:
            raise self.error(following.number, "multi-line scalars are not supported")
        return value

    def parse_mapping(self, indent: int) -> dict:
        result: dict = {}
        while True:
            line = self.current()
            if line is None or line.indent < indent:
                return result
            if line.indent > indent:
                raise self.error(line.number, "unexpected indentation")
            if _is_sequence_item(line.text):
                raise self.error(line.number, "a sequence item where a mapping key was expected")
            split = self.split_key(line)
            if split is None:
                raise self.error(line.number, "expected 'key: value'")
            key, value_text = split
            self.pos += 1
            if not _strip_comment(value_text, self.source, line.number).strip():
                following = self.current()
                if following is not None and following.indent > indent:
                    value = self.parse_block(following.indent)
                elif following is not None and following.indent == indent and _is_sequence_item(following.text):
                    value = self.parse_sequence(indent)
                else:
                    value = None
            else:
                value = self.parse_value_text(value_text, line.number)
                following = self.current()
                if following is not None and following.indent > indent:
                    raise self.error(following.number, "multi-line scalars are not supported (or the indentation is off)")
            if isinstance(key, (dict, list)):
                raise self.error(line.number, "complex mapping keys are not supported")
            result[key] = value

    def parse_sequence(self, indent: int) -> list:
        result: list = []
        while True:
            line = self.current()
            if line is None or line.indent < indent:
                return result
            if line.indent > indent:
                raise self.error(line.number, "unexpected indentation")
            if not _is_sequence_item(line.text):
                return result
            rest = line.text[1:]
            stripped_rest = rest.lstrip(" ")
            if not _strip_comment(stripped_rest, self.source, line.number).strip():
                self.pos += 1
                following = self.current()
                if following is not None and following.indent > indent:
                    result.append(self.parse_block(following.indent))
                else:
                    result.append(None)
                continue
            item_indent = indent + 1 + (len(rest) - len(stripped_rest))
            if _is_sequence_item(stripped_rest):
                raise self.error(line.number, "nested sequences on one line ('- - x') are not supported")
            inner = _Line(line.number, item_indent, stripped_rest)
            if self.split_key(inner) is not None:
                # "- key: value" starts a mapping whose keys line up with "key".
                self.lines[self.pos] = inner
                result.append(self.parse_mapping(item_indent))
                continue
            self.pos += 1
            result.append(self.parse_value_text(stripped_rest, line.number))
            following = self.current()
            if following is not None and following.indent > indent:
                raise self.error(following.number, "multi-line scalars are not supported (or the indentation is off)")

    # -- keys and scalars ------------------------------------------------

    def split_key(self, line: _Line) -> Optional[Tuple[Any, str]]:
        """Return (key, rest) when the line is 'key: ...', else None."""
        text = line.text
        if not text or text[0] in "[{":
            return None
        if text[0] in "'\"":
            value, end = _read_quoted(text, 0, self.source, line.number)
            rest = text[end:]
            stripped = rest.lstrip(" ")
            if stripped.startswith(":") and (len(stripped) == 1 or stripped[1] == " "):
                return value, stripped[1:]
            return None
        if text[0] in _REFUSED_START and not text.startswith("? ") and text[0] != "?":
            return None
        if text.startswith("?") and (len(text) == 1 or text[1] == " "):
            raise self.error(line.number, "complex mapping keys ('? key') are not supported")
        index = 0
        while index < len(text):
            char = text[index]
            if char == "#" and index > 0 and text[index - 1] == " ":
                return None
            if char == ":" and (index + 1 == len(text) or text[index + 1] == " "):
                key_text = text[:index].rstrip(" ")
                if not key_text:
                    raise self.error(line.number, "empty mapping key")
                return _resolve_plain(key_text, self.source, line.number), text[index + 1 :]
            index += 1
        return None

    def parse_value_text(self, text: str, number: int) -> Any:
        """Parse a scalar or one-line flow collection that fills the rest of a line."""
        text = text.strip(" ")
        if not text:
            return None
        first = text[0]
        if first in "'\"":
            value, end = _read_quoted(text, 0, self.source, number)
            rest = _strip_comment(text[end:], self.source, number).strip()
            if rest:
                raise self.error(number, f"unexpected text after a quoted scalar: {rest!r}")
            return str(value)
        if first in "[{":
            value, end = _FlowReader(text, self.source, number).read_top()
            rest = _strip_comment(text[end:], self.source, number).strip()
            if rest:
                raise self.error(number, f"unexpected text after a flow collection: {rest!r}")
            return value
        if first in _REFUSED_START or (first == "-" and text[1:2] in ("", " ")):
            raise self.error(number, f"{_REFUSED_START.get(first, 'this value')} are not supported: {text!r}")
        plain = _strip_comment(text, self.source, number).rstrip(" ")
        if ": " in plain or plain.endswith(":"):
            raise self.error(number, f"a ':' inside a plain value is ambiguous, quote it: {plain!r}")
        return _resolve_plain(plain, self.source, number)


def _is_sequence_item(text: str) -> bool:
    return text == "-" or text.startswith("- ")


def _strip_comment(text: str, source: str, number: int) -> str:
    """Remove a trailing ' # comment' from plain text (quotes handled by callers)."""
    if text.startswith("#"):
        return ""
    index = text.find(" #")
    return text if index == -1 else text[:index]


def _read_quoted(text: str, start: int, source: str, number: int) -> Tuple[_Quoted, int]:
    """Read a quoted scalar starting at text[start]; return (value, index after it)."""
    quote = text[start]
    index = start + 1
    out: List[str] = []
    while index < len(text):
        char = text[index]
        if quote == "'":
            if char == "'":
                if text[index + 1 : index + 2] == "'":
                    out.append("'")
                    index += 2
                    continue
                return _Quoted("".join(out)), index + 1
            out.append(char)
            index += 1
            continue
        if char == '"':
            return _Quoted("".join(out)), index + 1
        if char == "\\":
            code = text[index + 1 : index + 2]
            if code in _DOUBLE_ESCAPES:
                out.append(_DOUBLE_ESCAPES[code])
                index += 2
                continue
            if code in _HEX_ESCAPES:
                width = _HEX_ESCAPES[code]
                digits = text[index + 2 : index + 2 + width]
                if len(digits) != width or not all(c in "0123456789abcdefABCDEF" for c in digits):
                    raise YAMLSubsetError(source, number, f"bad escape '\\{code}{digits}' in a double-quoted scalar")
                out.append(chr(int(digits, 16)))
                index += 2 + width
                continue
            raise YAMLSubsetError(source, number, f"unknown escape '\\{code}' in a double-quoted scalar")
        out.append(char)
        index += 1
    raise YAMLSubsetError(source, number, "unclosed quoted scalar (multi-line quoted scalars are not supported)")


class _FlowReader:
    """Read a one-line flow sequence or mapping: [a, 'b', {c: 1}]."""

    def __init__(self, text: str, source: str, number: int) -> None:
        self.text = text
        self.source = source
        self.number = number
        self.index = 0

    def error(self, message: str) -> YAMLSubsetError:
        return YAMLSubsetError(self.source, self.number, message)

    def skip_spaces(self) -> None:
        while self.index < len(self.text) and self.text[self.index] == " ":
            self.index += 1

    def peek(self) -> str:
        return self.text[self.index : self.index + 1]

    def read_top(self) -> Tuple[Any, int]:
        value = self.read_node(in_mapping=False)
        return value, self.index

    def read_node(self, in_mapping: bool) -> Any:
        self.skip_spaces()
        char = self.peek()
        if char == "[":
            return self.read_sequence()
        if char == "{":
            return self.read_mapping()
        if char in ("'", '"'):
            value, self.index = _read_quoted(self.text, self.index, self.source, self.number)
            return str(value)
        return self.read_plain(in_mapping)

    def read_plain(self, in_mapping: bool) -> Any:
        start = self.index
        char = self.peek()
        if not char:
            raise self.error("unclosed flow collection (flow collections must fit on one line)")
        if char in _REFUSED_START:
            raise self.error(f"{_REFUSED_START[char]} are not supported")
        while self.index < len(self.text):
            char = self.text[self.index]
            if char in ",[]{}":
                break
            if char == "#" and self.index > start and self.text[self.index - 1] == " ":
                raise self.error("a comment inside a flow collection is not supported")
            if char == ":" and (self.index + 1 == len(self.text) or self.text[self.index + 1] in " ,]}"):
                if in_mapping:
                    break
                raise self.error("mappings inside a flow sequence need braces: [{a: 1}]")
            self.index += 1
        value = self.text[start : self.index].strip(" ")
        return _resolve_plain(value, self.source, self.number)

    def read_sequence(self) -> list:
        self.index += 1
        items: list = []
        self.skip_spaces()
        if self.peek() == "]":
            self.index += 1
            return items
        while True:
            items.append(self.read_node(in_mapping=False))
            self.skip_spaces()
            char = self.peek()
            if char == ",":
                self.index += 1
                self.skip_spaces()
                if self.peek() == "]":
                    self.index += 1
                    return items
                continue
            if char == "]":
                self.index += 1
                return items
            raise self.error("expected ',' or ']' in a flow sequence (flow collections must fit on one line)")

    def read_mapping(self) -> dict:
        self.index += 1
        result: dict = {}
        self.skip_spaces()
        if self.peek() == "}":
            self.index += 1
            return result
        while True:
            self.skip_spaces()
            key = self.read_node(in_mapping=True)
            if isinstance(key, (dict, list)):
                raise self.error("complex mapping keys are not supported")
            self.skip_spaces()
            if self.peek() == ":":
                self.index += 1
                self.skip_spaces()
                if self.peek() in (",", "}"):
                    value = None
                else:
                    value = self.read_node(in_mapping=True)
            else:
                value = None
            result[key] = value
            self.skip_spaces()
            char = self.peek()
            if char == ",":
                self.index += 1
                self.skip_spaces()
                if self.peek() == "}":
                    self.index += 1
                    return result
                continue
            if char == "}":
                self.index += 1
                return result
            raise self.error("expected ',' or '}' in a flow mapping (flow collections must fit on one line)")
