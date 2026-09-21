"""
core/cli.py — Dataclass-driven sys.argv parsing.

Goals:
  - Every training script exposes every config field as a `--flag`.
  - `--help` is auto-generated from the dataclass, so the interface can be
    discovered without reading source.
  - JSON config files still work (`--config path.json`), but every field
    is also overridable from the command line for ad-hoc bash sweeps.
  - Errors point at the offending flag with allowed values.

Usage in a training script:

    from firecomp.core.cli import Cli
    from firecomp.next_day.config import NextDayConfig

    def main():
        cli = Cli(NextDayConfig, prog="next_day.dl_2d",
                  description="Next-day fire spread segmentation.")
        cli.command("train", train)
        cli.command("eval", eval, positional=("checkpoint",))
        cli.run()

For one-off use without subcommands:

    cfg = parse_config(NextDayConfig, sys.argv[1:])

Bash sweep:
    for lr in 1e-3 5e-4 1e-4; do
      python -m firecomp.next_day.implementations.dl_2d train --lr $lr --tag lr_$lr
    done
"""

import argparse
import json
import sys
import types
import typing
from dataclasses import MISSING, fields as dc_fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin


# ---------------------------------------------------------------------------
# parse_config — the workhorse
# ---------------------------------------------------------------------------

def parse_config(cfg_cls: type, argv: list[str] | None = None,
                 prog: str = None, description: str = None,
                 extra_args: list[dict] | None = None,
                 ) -> tuple:
    """
    Parse argv into a `cfg_cls` instance.

    Resolution order (later wins):
        1. Dataclass defaults
        2. --config <path.json>      (whole-file override)
        3. Per-field --flag values

    Returns (cfg, remaining_positional_args). Most callers ignore the
    second element; subcommand dispatchers use it for trailing positionals.

    Raises SystemExit on --help or parse error (argparse standard).

    Args:
        cfg_cls:    A @dataclass type. Each field becomes a --flag.
        argv:       List like sys.argv[1:]. None → uses sys.argv[1:].
        prog:       Program name shown in --help.
        description: One-line description shown in --help.
        extra_args: Optional list of dicts with argparse.add_argument kwargs
                    for trailing positionals (e.g. {"name": "checkpoint",
                    "help": "Path to best.pt"}).
    """
    if not is_dataclass(cfg_cls):
        raise TypeError(f"parse_config requires a @dataclass, got {cfg_cls!r}")

    parser = _build_parser(cfg_cls, prog=prog, description=description,
                            extra_args=extra_args or [])
    ns = parser.parse_args(argv)

    # 1. start from defaults
    overrides: dict[str, Any] = {}

    # 2. JSON config file (replaces defaults)
    if getattr(ns, "config", None):
        with open(ns.config) as f:
            raw = json.load(f)
        valid = {f.name for f in dc_fields(cfg_cls)}
        unknown = set(raw) - valid
        if unknown:
            parser.error(f"--config {ns.config}: unknown fields {sorted(unknown)}")
        overrides.update({k: v for k, v in raw.items() if k in valid})

    # 3. CLI flags — only those the user explicitly set (others stay default
    #    or inherit from --config). We detect "explicit" by comparing against
    #    a sentinel default we install in the parser.
    for f in dc_fields(cfg_cls):
        val = getattr(ns, f.name, _UNSET)
        if val is _UNSET:
            continue
        overrides[f.name] = val

    cfg = cfg_cls(**overrides) if overrides else cfg_cls()

    # collect any extra positionals
    rest = {ea["name"]: getattr(ns, ea["name"]) for ea in (extra_args or [])
            if getattr(ns, ea["name"], None) is not None}

    return cfg, rest


# ---------------------------------------------------------------------------
# Cli — subcommand dispatcher (train / eval / etc.)
# ---------------------------------------------------------------------------

class Cli:
    """
    Subcommand dispatcher built on parse_config.

    Each command has its own argparse subparser populated from the same
    config dataclass. The subcommand handler is called with the parsed
    config plus any extra positional args declared at registration time.

    Usage:
        cli = Cli(NextDayConfig, prog="next_day.dl_2d",
                  description="Next-day fire spread segmentation.")
        cli.command("train", train)
        cli.command("eval", eval, positional=("checkpoint",))
        cli.run()

    Then:
        # next_day.dl_2d train --lr 1e-3 --tag exp1
        # next_day.dl_2d train --config base.json --lr 1e-3
        # next_day.dl_2d eval runs/exp1/best.pt --device cpu
        # next_day.dl_2d --help
        # next_day.dl_2d train --help        ← lists every config field
    """

    def __init__(self, cfg_cls: type, prog: str = None, description: str = None):
        if not is_dataclass(cfg_cls):
            raise TypeError(f"Cli requires a @dataclass, got {cfg_cls!r}")
        self.cfg_cls = cfg_cls
        self._handlers: dict[str, _Command] = {}
        self.prog = prog
        self.description = description

    def command(self, name: str, handler: Callable,
                positional: tuple[str, ...] = ()):
        """
        Register a subcommand.

        Args:
            name:       subcommand keyword (first argv after the script)
            handler:    callable invoked as handler(cfg, **positional_kwargs).
                        For commands with no positionals, handler(cfg).
            positional: names of additional positional args (after the
                        subcommand keyword). Each becomes a required arg
                        for that subcommand.
        """
        self._handlers[name] = _Command(name=name, handler=handler,
                                         positional=positional)

    def run(self, argv: list[str] | None = None) -> Any:
        """Parse argv, dispatch to the matching command, return its result."""
        if argv is None:
            argv = sys.argv[1:]

        if not self._handlers:
            raise RuntimeError("Cli has no commands; call .command() first")

        if not argv or argv[0] in ("-h", "--help"):
            self._print_top_help()
            sys.exit(0)

        cmd_name, rest = argv[0], argv[1:]
        if cmd_name not in self._handlers:
            self._print_top_help(error=f"unknown command: {cmd_name!r}")
            sys.exit(2)

        cmd = self._handlers[cmd_name]
        prog = f"{self.prog or sys.argv[0]} {cmd_name}"

        extra_args = [{"name": p, "help": f"<{p}>"} for p in cmd.positional]
        cfg, pos = parse_config(self.cfg_cls, rest, prog=prog,
                                 description=self.description,
                                 extra_args=extra_args)
        return cmd.handler(cfg, **pos)

    def _print_top_help(self, error: str = None):
        if error:
            print(f"error: {error}\n", file=sys.stderr)
        prog = self.prog or sys.argv[0]
        print(f"usage: {prog} <command> [options]")
        if self.description:
            print(f"\n{self.description}")
        print("\nCommands:")
        for name, cmd in self._handlers.items():
            pos = " ".join(f"<{p}>" for p in cmd.positional)
            print(f"  {name}{(' ' + pos) if pos else '':24s}  {_first_line(cmd.handler.__doc__)}")
        print(f"\nRun `{prog} <command> --help` to see config flags for a command.")


# ---------------------------------------------------------------------------
# _build_parser — turn dataclass fields into argparse arguments
# ---------------------------------------------------------------------------

def _build_parser(cfg_cls: type, prog: str = None, description: str = None,
                   extra_args: list[dict] = ()) -> argparse.ArgumentParser:
    """Construct an ArgumentParser populated from dataclass fields."""
    # We format the real default into each --flag's help string ourselves,
    # so we deliberately don't use ArgumentDefaultsHelpFormatter (which would
    # append `(default: <UNSET>)` to every entry — the _UNSET sentinel is
    # an internal implementation detail).
    parser = argparse.ArgumentParser(prog=prog, description=description)

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file. Field overrides on the command line "
             "take precedence over file values.",
    )

    type_hints = typing.get_type_hints(cfg_cls)

    cfg_group = parser.add_argument_group(f"{cfg_cls.__name__} fields")
    for f in dc_fields(cfg_cls):
        _add_field(cfg_group, f, type_hints.get(f.name, f.type))

    for ea in extra_args:
        kw = {k: v for k, v in ea.items() if k != "name"}
        parser.add_argument(ea["name"], **kw)

    return parser


def _add_field(group, field, type_):
    """Add one dataclass field as a --flag on the parser group."""
    flag = "--" + field.name.replace("_", "-")
    default = field.default if field.default is not MISSING else (
        field.default_factory() if field.default_factory is not MISSING else None
    )
    help_text = (field.metadata.get("help") if field.metadata else None) \
        or _describe_type(type_)

    origin = get_origin(type_)
    args = get_args(type_)

    # ---- Literal[...] → choices ----
    if origin is Literal:
        choices = list(args)
        item_type = type(choices[0]) if choices else str
        group.add_argument(flag, dest=field.name, type=item_type,
                           choices=choices, default=_UNSET,
                           help=f"{help_text} (default: {default})")
        return

    # ---- Optional[X] / Union[X, None] → unwrap ----
    if origin in (typing.Union,) or _is_union_with_none(type_):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _add_field(group, field, non_none[0])

    # ---- bool → BooleanOptionalAction (--flag / --no-flag) ----
    if type_ is bool:
        group.add_argument(flag, dest=field.name,
                           action=argparse.BooleanOptionalAction,
                           default=_UNSET,
                           help=f"{help_text} (default: {default})")
        return

    # ---- list[X] or bare list → nargs='+' ----
    if origin in (list, typing.List) or type_ is list:
        item_type = args[0] if args else str
        group.add_argument(flag, dest=field.name, nargs="+",
                           type=_coerce(item_type),
                           default=_UNSET,
                           help=f"{help_text} (default: {default})")
        return

    # ---- scalar (str, int, float) ----
    group.add_argument(flag, dest=field.name, type=_coerce(type_),
                       default=_UNSET,
                       help=f"{help_text} (default: {default})")


# ---------------------------------------------------------------------------
# Type coercion + introspection helpers
# ---------------------------------------------------------------------------

def _coerce(type_):
    """Return a callable str→T for argparse, falling back to str."""
    if type_ in (int, float, str):
        return type_
    if type_ is bool:
        return _str_to_bool
    return str


def _str_to_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):  return True
    if s in ("0", "false", "no", "n", "off"): return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {s!r}")


def _is_union_with_none(t) -> bool:
    """True for `X | None` / `Optional[X]` (both typing.Union and PEP 604 `|`)."""
    if get_origin(t) is typing.Union or isinstance(t, types.UnionType):
        return type(None) in get_args(t)
    return False


def _describe_type(type_) -> str:
    """Short human description of a type for help text."""
    if get_origin(type_) is Literal:
        return "{" + "|".join(str(a) for a in get_args(type_)) + "}"
    if get_origin(type_) in (list, typing.List):
        item = get_args(type_)[0] if get_args(type_) else str
        return f"list of {getattr(item, '__name__', repr(item))}"
    if hasattr(type_, "__name__"):
        return type_.__name__
    return repr(type_)


def _first_line(s: str | None) -> str:
    if not s:
        return ""
    for line in s.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _Unset:
    """Sentinel: distinguishes "user didn't pass --flag" from "user passed --flag <default>".

    parse_config uses this to know whether a CLI flag was explicitly set,
    so it can give --config values precedence when the flag is absent.
    """
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self): return "<UNSET>"
    def __bool__(self): return False


_UNSET = _Unset()


class _Command:
    """One registered subcommand."""
    __slots__ = ("name", "handler", "positional")

    def __init__(self, name: str, handler: Callable, positional: tuple[str, ...]):
        self.name = name
        self.handler = handler
        self.positional = positional
