from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UseImageARM:
    pass

class UseSynthesis:
    pass

SubmoduleConfig = UseImageARM | UseSynthesis
