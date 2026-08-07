from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

NANO_DIR = Path.home() / ".nano"
CONFIG_PATH = NANO_DIR / "config.toml"
SKILLS_DIR = NANO_DIR / "skills"

CONFIG_TEMPLATE = """\
# Nano personal assistant — ~/.nano/config.toml

[assistant]
name = "Nano"

# Override the default system prompt (uncomment to customize):
# system_prompt = \"\"\"
# You are a helpful assistant specializing in ...
# \"\"\"

[routing]
# "auto" — LLM-based difficulty routing (cost-efficient, recommended)
# "T0" / "T1" / "T2" / "T3" — force a specific model tier
tier = "auto"

[defaults]
# Activate a skill on startup (uncomment to enable):
# skill = "researcher"
"""


@dataclass
class NanoConfig:
    name: str = "Nano"
    system_prompt: str = ""     # empty → use DEFAULT_SYSTEM_PROMPT
    tier: str = "auto"          # "auto" or T0/T1/T2/T3
    default_skill: str = ""

    @classmethod
    def load(cls) -> "NanoConfig":
        """从 ~/.nano/config.toml 加载配置；文件不存在时返回默认值 / Load config; returns defaults if file absent."""
        if not CONFIG_PATH.exists():
            return cls()
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        assistant = data.get("assistant", {})
        routing = data.get("routing", {})
        defaults = data.get("defaults", {})
        return cls(
            name=str(assistant.get("name", "Nano")),
            system_prompt=str(assistant.get("system_prompt", "")),
            tier=str(routing.get("tier", "auto")),
            default_skill=str(defaults.get("skill", "")),
        )
