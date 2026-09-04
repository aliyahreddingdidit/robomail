"""Robot capability profile (CLAUDE.md architecture item 6).

Hard physical constraints of the Franka Emika Panda cell, injected into the
High-Level Planner's and Step Planner's system prompts so that implausible plans
(the classic one from the booklet: "pour the red and blue cups in at the same
time") are reasoned around *autonomously at plan time*.

This is deliberately NOT a review gate: nothing here blocks a plan or asks a
human. It changes what the planner produces, and every time it does, the planner
reports a workaround note which the Logging Agent records (item 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotProfile:
    """Hard capability constraints, embedded at plan time."""

    name: str = "Franka Emika Panda (single-arm wet-chemistry cell)"

    #: Number of independently controllable arms.
    num_arms: int = 1
    #: Number of end effectors / grippers.
    num_grippers: int = 1
    gripper_type: str = "parallel-plate, two-finger"
    #: How many objects can be held at once (one gripper -> one object).
    max_simultaneously_held_objects: int = 1

    payload_kg: float = 3.0
    gripper_max_opening_cm: float = 8.0

    hard_constraints: tuple[str, ...] = (
        "There is exactly ONE arm with ONE parallel-plate gripper.",
        "Only ONE object may be held at any moment. To use a second object, the "
        "first must be PLACEd down first.",
        "The robot CANNOT perform two spatially-separate actions simultaneously. "
        "Any instruction of the form 'do A and B at the same time' must be "
        "re-expressed as a strictly sequential plan.",
        "The robot CANNOT exert two independent forces at once (e.g. it cannot "
        "hold a container steady with one effector while stirring it with "
        "another). Containers must be stable on the bench unaided.",
        "The gripper has no wrist-mounted force/torque control fine enough to "
        "meter a pour by weight; quantities are metered by scoop or pipette.",
        "Containers themselves are not relocated unless the task requires it; "
        "prefer transferring contents into a stationary container.",
    )

    #: Directives the planner must follow when a constraint bites.
    workaround_directives: tuple[str, ...] = (
        "Serialise simultaneous requirements into an explicit ordered sequence.",
        "Insert an explicit PLACE before any step that needs a different tool.",
        "If serialising changes the observable outcome (e.g. a reaction that the "
        "source text expects to happen on simultaneous contact), say so plainly "
        "in the workaround note rather than silently pretending it is equivalent.",
    )

    #: Fixed, semantically-named workspace positions (PLATO convention).
    workspace_positions: tuple[str, ...] = (
        "Home Pose",
        "Original Position of beaker",
        "Original Position of clear cup 1",
        "Original Position of clear cup 2",
        "Original Position of clear cup 3",
        "Original Position of paper cup",
        "Original Position of red cabbage powder container",
        "Original Position of sodium bicarbonate container",
        "Original Position of citric acid container",
        "Original Position of instant snow powder container",
        "Original Position of hydrophobic sand container",
        "Original Position of measuring scoop",
        "Original Position of pipette",
        "Original Position of water container",
        "Verification Pose",
    )

    def as_prompt_block(self) -> str:
        """Render the profile for injection into an LLM system prompt."""
        lines = [
            "ROBOT CAPABILITY PROFILE (hard constraints -- plan around these yourself):",
            f"  Platform: {self.name}",
            f"  Arms: {self.num_arms}   Grippers: {self.num_grippers} ({self.gripper_type})",
            f"  Max objects held at once: {self.max_simultaneously_held_objects}",
            f"  Payload: {self.payload_kg} kg   Max gripper opening: {self.gripper_max_opening_cm} cm",
            "",
            "  Constraints:",
        ]
        lines += [f"    - {c}" for c in self.hard_constraints]
        lines += ["", "  When a constraint makes the literal instruction impossible:"]
        lines += [f"    - {d}" for d in self.workaround_directives]
        lines += [
            "",
            "  You must NOT ask a human for help, approval or clarification. Decide",
            "  yourself and record what you changed.",
            "",
            "  Whenever a constraint above changes your plan relative to a literal",
            '  reading of the goal, add an entry to "capability_workarounds" naming the',
            "  constraint, what the literal instruction asked for, and what you did instead.",
        ]
        return "\n".join(lines)

    def positions_prompt_block(self) -> str:
        joined = ", ".join(f'"{p}"' for p in self.workspace_positions)
        return (
            "Available workspace positions (use these names exactly; you may express "
            f"offsets relative to them): {joined}"
        )


#: The profile used across the pipeline.
DEFAULT_PROFILE = RobotProfile()


@dataclass
class CapabilityWorkaround:
    """A single planner-reported adjustment forced by the profile (logged, item 8)."""

    constraint: str
    literal_instruction: str
    adopted_approach: str
    outcome_differs: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "constraint": self.constraint,
            "literal_instruction": self.literal_instruction,
            "adopted_approach": self.adopted_approach,
            "outcome_differs": self.outcome_differs,
            "note": self.note,
        }
