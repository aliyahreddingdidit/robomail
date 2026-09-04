"""Scene Understanding Agent (CLAUDE.md architecture item 1).

Adapted from PLATO/PLATO/scene_comprehension.py. Changes from upstream:

  * Object vocabulary extended to the wet-chemistry kit (beaker, pipette,
    measuring scoop, clear cups, paper cups, red cabbage powder, sodium
    bicarbonate, citric acid, instant snow powder, hydrophobic sand, pH
    indicator paper).
  * Structured JSON output instead of ``ast.literal_eval`` on a free-text
    "two python lists" reply, which upstream could not parse robustly.
  * The handle flag is kept -- downstream affordance/grasping consumes it.
  * ``model='gpt-4o'`` replaced by the env-driven cheap tier (config.models).

The agent still has NO hard-coded scene knowledge: it is told the kit's object
*vocabulary* so it can name things consistently, not which objects are present
or where they are. That is exactly PLATO's design and the paper's claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agents import llm_client
from config import models
from config.robot_profile import DEFAULT_PROFILE, RobotProfile

AGENT_NAME = "SceneUnderstandingAgent"

#: The chemistry-kit vocabulary this project adds on top of PLATO's generic
#: tabletop objects. Supplied to the model as *candidate* names for consistency.
KIT_VOCABULARY: tuple[str, ...] = (
    "beaker",
    "pipette",
    "measuring scoop",
    "small scoop",
    "big scoop",
    "clear cup",
    "paper cup",
    "red cabbage powder",
    "sodium bicarbonate",
    "citric acid",
    "instant snow powder",
    "hydrophobic sand",
    "pH indicator paper",
    "water container",
    "stirring rod",
    "tray",
)

#: Categories drive downstream affordance reasoning (what can be poured, scooped...).
OBJECT_CATEGORIES: tuple[str, ...] = (
    "container",     # beaker, cups -- hold contents, generally not relocated
    "reagent",       # powders and solutions
    "tool",          # pipette, scoop, stirring rod
    "indicator",     # pH paper, prepared indicator solution
    "distractor",    # in-frame but irrelevant to the goal
)

_SYSTEM_PROMPT_TEMPLATE = """You are the Scene Understanding Agent of an autonomous robotic chemist.

You are given an overhead image of the robot's workspace and the natural-language
goal that downstream agents must achieve in that workspace.

List the objects actually visible on the bench. For each object report:
  name             -- a brief noun phrase. Prefer a name from the kit vocabulary
                      below when the object plausibly matches one, so downstream
                      agents and the goal text use the same words. If an object is
                      clearly not in the vocabulary, name it briefly anyway.
  category         -- one of: {categories}
  has_handle       -- true if the object has a graspable handle/stem (a pipette
                      and a scoop do; a cup and a beaker generally do not),
                      false otherwise. This drives grasp selection downstream.
  contents         -- what the object appears to contain, or "empty", or "unknown".
                      Report what you SEE (e.g. "dark purple liquid"), not what
                      you infer the chemistry to be.
  relevant_to_goal -- true if this object is plausibly needed for the goal.

Kit vocabulary (candidate names, NOT a list of what is present):
{vocabulary}

Rules:
  * Report only what is visible. Do not invent objects because the goal mentions
    them, and do not omit objects because the goal does not.
  * Include irrelevant objects too, categorised as "distractor" -- the evaluation
    protocol deliberately adds distractors to the scene.
  * Ignore markings, tape and fiducials on the bench surface itself.
  * Do not plan. Do not suggest actions. Only describe what is there.
  * Sort objects alphabetically by name.

{robot_block}
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string", "enum": list(OBJECT_CATEGORIES)},
                    "has_handle": {"type": "boolean"},
                    "contents": {"type": "string"},
                    "relevant_to_goal": {"type": "boolean"},
                },
                "required": [
                    "name",
                    "category",
                    "has_handle",
                    "contents",
                    "relevant_to_goal",
                ],
                "additionalProperties": False,
            },
        },
        "scene_notes": {"type": "string"},
    },
    "required": ["objects", "scene_notes"],
    "additionalProperties": False,
}


@dataclass
class SceneObject:
    """One object grounded in the workspace image."""

    name: str
    category: str
    has_handle: bool
    contents: str
    relevant_to_goal: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "has_handle": self.has_handle,
            "contents": self.contents,
            "relevant_to_goal": self.relevant_to_goal,
        }


@dataclass
class Scene:
    """The Scene Understanding Agent's output."""

    objects: list[SceneObject] = field(default_factory=list)
    scene_notes: str = ""
    source_image: str = ""

    @property
    def names(self) -> list[str]:
        return [o.name for o in self.objects]

    @property
    def handle_flags(self) -> list[int]:
        """PLATO-compatible binary handle flags, aligned with :attr:`names`."""
        return [int(o.has_handle) for o in self.objects]

    def relevant(self) -> list[SceneObject]:
        return [o for o in self.objects if o.relevant_to_goal]

    def as_dict(self) -> dict:
        return {
            "objects": [o.as_dict() for o in self.objects],
            "scene_notes": self.scene_notes,
            "source_image": self.source_image,
        }

    def as_prompt_block(self) -> str:
        """Render for injection into the Planner's prompt."""
        if not self.objects:
            return "Objects visible in the workspace: (none detected)"
        lines = ["Objects visible in the workspace:"]
        for o in self.objects:
            handle = "has handle" if o.has_handle else "no handle"
            flag = "" if o.relevant_to_goal else "  [likely distractor]"
            lines.append(f"  - {o.name} ({o.category}, {handle}, contains: {o.contents}){flag}")
        if self.scene_notes:
            lines.append(f"Scene notes: {self.scene_notes}")
        return "\n".join(lines)


def comprehend_scene(
    workspace_image: str | Path,
    goal: str,
    *,
    profile: RobotProfile = DEFAULT_PROFILE,
    model: str | None = None,
    client=None,
) -> Scene:
    """Ground the objects in a workspace image, in the context of ``goal``."""
    path = Path(workspace_image)
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        categories=", ".join(OBJECT_CATEGORIES),
        vocabulary="\n".join(f"  - {v}" for v in KIT_VOCABULARY),
        robot_block=profile.as_prompt_block(),
    )
    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or models.cheap_model(),
        system_prompt=system_prompt,
        user_content=[
            llm_client.text_block(
                f"What objects are present in this workspace image? "
                f"The goal that downstream agents must achieve here is: <{goal}>"
            ),
            llm_client.image_block(path, detail="high"),
        ],
        schema=_SCHEMA,
        schema_name="scene_description",
        client=client,
    )
    objects = [
        SceneObject(
            name=str(o["name"]).strip(),
            category=str(o.get("category", "distractor")),
            has_handle=bool(o.get("has_handle", False)),
            contents=str(o.get("contents", "unknown")),
            relevant_to_goal=bool(o.get("relevant_to_goal", True)),
        )
        for o in response.get("objects", [])
    ]
    objects.sort(key=lambda o: o.name.lower())
    return Scene(
        objects=objects,
        scene_notes=str(response.get("scene_notes", "")).strip(),
        source_image=str(path),
    )
