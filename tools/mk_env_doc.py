from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import yaml
from docstring_parser import parse

if TYPE_CHECKING:
    import ap_gym


def render_md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Prints a markdown table with the given headers and rows."""
    col_widths = [
        max(len(header), max(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]
    header_row = (
        "| "
        + " | ".join(
            f"{header:<{col_width}}" for header, col_width in zip(headers, col_widths)
        )
        + " |"
    )
    separator_row = (
        "|-" + "-|-".join("-" * col_width for col_width in col_widths) + "-|"
    )
    return "\n".join(
        [header_row, separator_row]
        + [
            "| "
            + " | ".join(
                f"{str(item):<{col_width}}" for item, col_width in zip(row, col_widths)
            )
            + " |"
            for row in rows
        ]
    )


def render_shape_latex(shape: tuple[int, ...]) -> str:
    return r" \times ".join(str(s) for s in shape)


def render_shape(shape: tuple[int, ...]) -> str:
    if len(shape) == 0:
        return "scalar"
    if len(shape) == 1:
        return f"{shape[0]}-element"
    else:
        return f"${render_shape_latex(shape)}$"


def render_bounds(space: gym.spaces.Box) -> str:
    low_scalar = np.min(space.low)
    if np.any(space.low != low_scalar):
        print(f"Warning: lower bound of space {space} is not scalar. Skipping.")
        return ""
    high_scalar = np.max(space.high)
    if np.any(space.high != high_scalar):
        print(f"Warning: upper bound of space {space} is not scalar. Skipping.")
        return ""

    if low_scalar == -np.inf:
        low = f"(-\infty"
    else:
        if np.abs(np.round(low_scalar) - low_scalar) < 1e-5:
            v = f"{int(round(low_scalar))}"
        else:
            v = f"{low_scalar:0.2f}"
        low = f"[{v}"
    if high_scalar == np.inf:
        high = f"\infty)"
    else:
        if np.abs(np.round(high_scalar) - high_scalar) < 1e-5:
            v = f"{int(round(high_scalar))}"
        else:
            v = f"{high_scalar:0.2f}"
        high = f"{v}]"
    set_repr = f"{low}, {high}"
    if space.shape != ():
        set_repr += f"^{{{render_shape_latex(space.shape)}}}"

    return rf" $\in {set_repr}$"


def render_box(space: gym.spaces.Box) -> str:
    if hasattr(space, "__idoc__"):
        doc = f" that {space.__idoc__}"
    else:
        doc = "."
    if space.shape == ():
        arr = f"`{space.dtype}`"
    else:
        arr = f"`np.ndarray[{space.dtype}]`"
    return f"{render_shape(space.shape).capitalize()} {arr}{render_bounds(space)}{doc}"


def determine_space_type(space: gym.spaces.Space) -> str:
    sample = space.sample()
    if isinstance(sample, np.ndarray):
        if sample.ndim == 0:
            return str(sample.dtype)
        else:
            return "np.ndarray"
    else:
        return str(type(sample))


def render_dict(space: gym.spaces.Dict) -> str:
    entries = [
        (k, determine_space_type(v), render_space(v)) for k, v in space.spaces.items()
    ]
    entries.sort(key=lambda x: x[0])
    table = render_md_table(["Key", "Type", "Description"], entries)
    return f"dictionary with the following keys:\n\n{table}"


def render_discrete(space: gym.spaces.Discrete) -> str:
    raise NotImplementedError()


def render_space(space: gym.spaces.Space) -> str:
    if isinstance(space, gym.spaces.Box):
        return render_box(space)
    elif isinstance(space, gym.spaces.Dict):
        return render_dict(space)
    elif isinstance(space, gym.spaces.Discrete):
        return render_discrete(space)
    else:
        raise NotImplementedError(f"Rendering for {type(space)} is not implemented.")


def _render_dict_compact(space: gym.spaces.Dict, indentation: int = 0) -> str:
    entries = [
        (k, _render_space_compact(v, indentation=indentation + 2))
        for k, v in space.spaces.items()
    ]
    entries.sort(key=lambda x: x[0])
    l = max(len(k) for k, v in entries)
    inner = "</code><br/><code>".join(
        f'{"&nbsp;" * (indentation + 2)}"{k}"{" " * (l - len(k))}: {v}'
        for k, v in entries
    )
    return (
        f"Dict({{</code><br/><code>{inner}</code><br/><code>{'&nbsp;' * indentation}}})"
    )


def _render_space_compact(space: gym.spaces.Space, indentation: int = 0) -> str:
    if isinstance(space, gym.spaces.Dict):
        return _render_dict_compact(space, indentation=indentation)
    else:
        return str(space)


def render_space_compact(space: gym.spaces.Space) -> str:
    return f"<code>{_render_space_compact(space)}</code>"


def is_marker_line(line: str, target: str) -> bool:
    return line.replace(" ", "").replace("\t", "").lower() == f"#!{target.lower()}"


def is_base_env(env: ap_gym.ActivePerceptionEnv) -> bool:
    """Checks if the environment is a base environment."""
    if env.__doc__ is None:
        return False
    doc = inspect.cleandoc(env.__doc__)
    lines = doc.splitlines()
    return any(is_marker_line(l, "ap_gym_base_env") for l in lines)


def is_wrapper(env: ap_gym.ActivePerceptionEnv) -> bool:
    """Checks if the environment is a base environment."""
    if env.__doc__ is None:
        return False
    doc = inspect.cleandoc(env.__doc__)
    lines = doc.splitlines()
    return any(is_marker_line(l, "ap_gym_wrapper") for l in lines)


def render_sections(sections: dict[str | None, str | dict], level: int = 0) -> str:
    output = []
    for k, v in sections.items():
        if isinstance(v, dict):
            v = render_sections(v, level=level + 1)
        if k is None:
            output.append(v)
        else:
            output.append(f"{'#' * (level + 1)} {k}\n\n{v}")
    return "\n\n".join(output) + "\n"


def doc_extract_yaml(doc: str) -> str:
    lines = doc.splitlines()
    marker_lines = [
        i
        for i, l in enumerate(lines)
        if is_marker_line(l, "ap_gym_base_env") or is_marker_line(l, "ap_gym_wrapper")
    ]
    return "\n".join(lines[marker_lines[0] + 1 :])


def get_loss_fn_text(loss_fn: ap_gym.LossFn) -> str:
    import ap_gym

    if isinstance(loss_fn, ap_gym.CrossEntropyLossFn):
        return "cross entropy"
    elif isinstance(loss_fn, ap_gym.MSELossFn):
        return "mean squared error"
    else:
        raise NotImplementedError(f"Rendering for {type(loss_fn)} is not implemented.")


def get_loss_fn_name(loss_fn: ap_gym.LossFn) -> str:
    name = type(loss_fn).__name__
    module = type(loss_fn).__module__
    if module.startswith("ap_gym."):
        module = "ap_gym"
    return f"{module}.{name}"


def render_html_cell(content: str) -> str:
    return f"    <td>{content}</td>\n"


def render_html_row(row: list[str]) -> str:
    return f"  <tr>\n{''.join(render_html_cell(c) for c in row)}  </tr>\n"


def render_html_table(rows: list[list[str]]) -> str:
    return f"<table>\n{''.join(render_html_row(r) for r in rows)}</table>\n"


def render_env(env_name: str) -> str:
    import ap_gym

    all_versions_and_variants = sorted(
        [e[len(env_name) + 1 :] for e in gym.registry if e.startswith(f"{env_name}-")]
    )
    all_versions = sorted(
        {int(e.split("-")[-1][1:]) for e in all_versions_and_variants}
    )
    latest_version = max(all_versions)
    full_env_name = f"{env_name}-v{latest_version}"

    env = ap_gym.make(full_env_name)
    base_env = env
    wrappers = []
    while not is_base_env(base_env):
        if not hasattr(base_env, "env"):
            raise ValueError(
                f"Could not unwrap environment {base_env} further and environment is not a base environment."
            )
        if is_wrapper(base_env):
            wrappers.append(base_env)
        base_env = base_env.env
    doc = inspect.cleandoc(base_env.__doc__)
    doc_parsed = yaml.safe_load(doc_extract_yaml(doc))

    version_info = {0: "Initial version"}
    version_info.update([(i, p) for i, p in enumerate(doc_parsed.get("patches", []))])
    if len(version_info) != len(all_versions):
        raise ValueError(
            f"Length of version info ({len(version_info)}) does not match the number of versions ({len(all_versions)})."
        )
    version_history = "\n".join(
        f"- `v{i}`: {version_info[i]}" for i in range(len(all_versions))
    )

    example_usage = inspect.cleandoc(
        f"""
        ```python
        import ap_gym
        
        env = ap_gym.make("{full_env_name}")

        # Or for the vectorized version with 4 environments:
        envs = ap_gym.make_vec("{full_env_name}", num_envs=4)
        ```
    """
    )

    arg_descriptions = {
        p.arg_name: p.description for p in parse(base_env.__init__.__doc__).params
    }
    args = inspect.signature(base_env.__init__).parameters
    arguments_lst = [
        (
            f"`{name}`",
            f"`{arg.annotation}`",
            f"`{repr(arg.default)}`",
            arg_descriptions[name],
        )
        for name, arg in args.items()
    ]
    arguments = render_md_table(
        ["Name", "Type", "Default", "Description"], arguments_lst
    )

    end_conditions = doc_parsed.get("end_conditions", {})
    terminate_conditions = end_conditions.get("terminate", [])
    truncate_conditions = end_conditions.get("truncate", [])
    for wrapper in reversed(wrappers):
        wrapper_doc_parsed = yaml.safe_load(
            doc_extract_yaml(inspect.cleandoc(wrapper.__doc__))
        )
        wrapper_end_conditions = wrapper_doc_parsed.get("end_conditions", {})
        terminate_conditions += wrapper_end_conditions.get("terminate", [])
        truncate_conditions += wrapper_end_conditions.get("truncate", [])
    episode_end = ""
    for name, conditions in [
        ("terminate", terminate_conditions),
        ("truncate", truncate_conditions),
    ]:
        if len(conditions) > 0:
            l = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(conditions)])
            if len(episode_end) > 0:
                episode_end += "\n\n"
            episode_end += (
                f"The episode ends with the {name} flag set if one of the following conditions is met:\n "
                f"{l}"
            )
    if len(episode_end) == 0:
        episode_end = "The episode ends when the environment is reset."

    rewards_lst = doc_parsed.get("rewards", [])
    for wrapper in reversed(wrappers):
        wrapper_doc_parsed = yaml.safe_load(
            doc_extract_yaml(inspect.cleandoc(wrapper.__doc__))
        )
        rewards_lst += wrapper_doc_parsed.get("rewards", [])
    rewards_lst.append(
        f"The negative {get_loss_fn_text(env.loss_fn)} between the agent's prediction and the target."
    )

    rewards = "The reward at each timestep is "
    if len(rewards_lst) == 1:
        rewards_lst[0] = rewards_lst[0][0].lower() + rewards_lst[0][1:]
        rewards += rewards_lst[0]
    else:
        inner = "\n".join(f"- {e}" for e in rewards_lst)
        rewards += f" the sum of:\n{inner}"

    spaces_raw = [
        ("Action Space", env.inner_action_space),
        ("Prediction Space", env.prediction_space),
        ("Prediction Target Space", env.prediction_target_space),
        ("Observation Space", env.observation_space),
    ]
    properties = render_html_table(
        [[f"<strong>{n}</strong>", render_space_compact(s)] for n, s in spaces_raw]
        + [
            [
                "<strong>Loss Function</strong>",
                f"<code>{get_loss_fn_name(env.loss_fn)}()</code>",
            ]
        ]
    )

    sections = {
        env_name: {
            None: f'<p align="center"><img src="img/{full_env_name}.gif" alt="{full_env_name}" width="200px"/></p>',
            "Description": doc_parsed["description"],
            "Properties": properties,
            "Action Space": f"The action is a {render_space(env.inner_action_space)}",
            "Prediction Space": f"The prediction is a {render_space(env.prediction_space)}",
            "Prediction Target Space": f"The prediction target is a {render_space(env.prediction_target_space)}",
            "Observation Space": f"The observation is a {render_space(env.observation_space)}",
            "Rewards": rewards,
            "Starting State": doc_parsed["starting_state"],
            "Episode End": episode_end,
            "Arguments": arguments,
            "Example Usage": example_usage,
            "Version History": version_history,
        }
    }
    return render_sections(sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for the documentation."
    )
    parser.add_argument(
        "modules", type=str, nargs="+", help="Modules to create docs for."
    )
    parser.add_argument(
        "-i", "--include", nargs="+", type=str, help="Specific environment to include."
    )
    args = parser.parse_args()

    exclude_list = set(gym.registry)

    for module in args.modules:
        __import__(module)

    if args.include is not None:
        env_names = set(args.include)
    else:
        env_names_with_versions = set(gym.registry) - exclude_list
        env_names = set("-".join(e.split("-")[:-1]) for e in env_names_with_versions)

    args.output_dir.mkdir(exist_ok=True)

    for env_name in sorted(env_names):
        print(f"Rendering {env_name}...")
        rendered = render_env(env_name)

        with (args.output_dir / f"{env_name}.md").open("w") as f:
            f.write(rendered)
