"""Parsing and validation for Clue Chain Hunt QR payloads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Tuple


CLUE_RE = re.compile(
    r"^HUNT:(?P<id>\d+):(?P<token>[0-9A-F]{4}):(?P<body>.+)$"
)


class ClueError(ValueError):
    """Raised when a QR payload is not a valid hunt clue."""


@dataclass(frozen=True)
class Clue:
    text: str
    board_id: int
    token: str
    command: str
    args: Tuple[object, ...]
    treasure: bool = False


def chain_token(previous_text: str) -> str:
    """Return the four-character chain token for ``previous_text``."""
    return hashlib.sha1(previous_text.encode("utf-8")).hexdigest()[:4].upper()


def _number(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ClueError(f"not a number: {value}") from exc
    if not math.isfinite(result):
        raise ClueError(f"number is not finite: {value}")
    return result


def parse_clue(text: str) -> Clue:
    """Parse one complete QR payload using the challenge's strict grammar."""
    text = text.strip()
    match = CLUE_RE.fullmatch(text)
    if not match:
        raise ClueError("payload does not match HUNT:<id>:<token>:<command>")

    board_id = int(match.group("id"))
    token = match.group("token")
    body = match.group("body").strip()
    treasure = False
    if body.startswith("TREASURE "):
        treasure = True
        body = body[len("TREASURE "):].strip()

    parts = body.split()
    if not parts:
        raise ClueError("missing command")
    command = parts[0]

    if command == "GOTO" and len(parts) == 3 and not treasure:
        args: Tuple[object, ...] = (_number(parts[1]), _number(parts[2]))
    elif command == "PILLAR" and len(parts) == 2 and not treasure:
        colour = parts[1].upper()
        if colour not in {"RED", "GREEN", "BLUE"}:
            raise ClueError(f"unknown pillar colour: {colour}")
        args = (colour,)
    elif command == "BETWEEN" and len(parts) == 4 and not treasure:
        colour_a, colour_b = parts[1].upper(), parts[2].upper()
        if colour_a not in {"RED", "GREEN", "BLUE"}:
            raise ClueError(f"unknown pillar colour: {colour_a}")
        if colour_b not in {"RED", "GREEN", "BLUE"}:
            raise ClueError(f"unknown pillar colour: {colour_b}")
        fraction = _number(parts[3])
        if not 0.0 <= fraction <= 1.0:
            raise ClueError("BETWEEN fraction must be between zero and one")
        args = (colour_a, colour_b, fraction)
    elif command == "REL" and len(parts) == 3:
        args = (_number(parts[1]), _number(parts[2]))
    else:
        raise ClueError(f"invalid command: {body}")

    if treasure and command != "REL":
        raise ClueError("TREASURE only supports REL")
    return Clue(text, board_id, token, command, args, treasure)


def validate_clue(text: str, expected_id: int, previous_text: str) -> Clue:
    """Parse a clue and enforce both its next id and SHA-1 chain token."""
    clue = parse_clue(text)
    if clue.board_id != expected_id:
        raise ClueError(f"expected board {expected_id}, saw {clue.board_id}")
    expected_token = chain_token(previous_text)
    if clue.token != expected_token:
        raise ClueError(f"expected token {expected_token}, saw {clue.token}")
    if expected_id != 1 and clue.command == "GOTO":
        raise ClueError("GOTO is only allowed on board 1")
    return clue
