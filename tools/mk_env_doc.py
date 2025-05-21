from __future__ import annotations

import argparse
import copy
import inspect
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from docstring_parser import parse
from gymnasium.envs.registration import parse_env_id

PURE_GYM_ENVS = set(gym.registry)

import ap_gym

AP_GYM_ENVS = [e for e in gym.registry if e not in PURE_GYM_ENVS]


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


def render_shape_latex(
    shape: tuple[int, ...], var_spec: dict[int, str] | None = None
) -> str:
    if var_spec is None:
        var_spec = {}
    return r" \times ".join(str(var_spec.get(i, s)) for i, s in enumerate(shape))


def render_shape(shape: tuple[int, ...], var_spec: dict[int, str] | None = None) -> str:
    if var_spec is None:
        var_spec = {}
    if len(shape) == 0:
        return "scalar"
    if len(shape) == 1:
        return f"{var_spec.get(0, shape[0])}-element"
    else:
        return f"${render_shape_latex(shape, var_spec)}$"


def render_bounds(space: gym.spaces.Box, var_spec: dict[int, str] | None = None) -> str:
    low_scalar = np.min(space.low)
    if np.any(space.low != low_scalar):
        print(f"Warning: lower bound of space {space} is not scalar.")
    high_scalar = np.max(space.high)
    if np.any(space.high != high_scalar):
        print(f"Warning: upper bound of space {space} is not scalar.")

    if low_scalar == -np.inf and high_scalar == np.inf:
        set_repr = r"\mathbb{R}"
    else:
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
        set_repr += f"^{{{render_shape_latex(space.shape, var_spec=var_spec)}}}"

    return rf" $\in {set_repr}$"


def render_box(space: gym.spaces.Box) -> str:
    text, var_spec = get_text_var_idoc(space)
    if text is not None:
        doc = f" that {text}"
    else:
        doc = "."
    if space.shape == ():
        arr = f"`{space.dtype}`"
    else:
        arr = f"`np.ndarray[{space.dtype}]`"
    return f"{arr}{render_bounds(space, var_spec)}{doc}"


def determine_space_type(space: gym.spaces.Space) -> str:
    sample = space.sample()
    if isinstance(sample, np.ndarray):
        if sample.ndim == 0:
            return str(sample.dtype)
        else:
            return "np.ndarray"
    else:
        return str(type(sample))


def get_extra_text_entries(
    space: gym.spaces.Space,
) -> tuple[str, dict[str, gym.spaces.Space]]:
    idoc = get_idoc(space)
    extra_text = ""
    extra_entries = {}
    if isinstance(idoc, dict):
        extra_text = idoc.get("extra_text", "")
        extra_entries = idoc.get("extra_entries", {})
    if extra_text != "":
        extra_text = f"\n{extra_text}"
    return extra_text, extra_entries


def render_dict(space: gym.spaces.Dict) -> str:
    extra_text, extra_entries = get_extra_text_entries(space)
    entries = [
        (k, f"`{determine_space_type(v)}`", render_space(v))
        for k, v in chain(space.spaces.items(), extra_entries.items())
    ]
    entries.sort(key=lambda x: x[0])
    table = render_md_table(["Key", "Type", "Description"], entries)
    return f"dictionary with the following keys:\n\n{table}{extra_text}"


def render_integer_set(max_val: int | str) -> str:
    if isinstance(max_val, int) and max_val <= 2:
        vals = ", ".join(str(i) for i in range(max_val + 1))
        return rf"$\{{{vals}\}}$"
    return rf"$\{{0, \dots{{}}, {max_val}\}}$"


def render_discrete(space: gym.spaces.Discrete) -> str:
    text, limit = get_text_var_idoc(space)
    if text is not None:
        doc = f" that {text}"
    else:
        doc = "."
    if limit is None:
        limit = space.n - 1
    return f"scalar integer in {render_integer_set(limit)}{doc}"


def get_idoc(obj: Any) -> Any | None:
    if hasattr(obj, "__idoc__"):
        return obj.__idoc__
    return None


class ReprString(str):
    def __repr__(self):
        return str(self)


def get_text_var_idoc(space: gym.spaces.Space) -> tuple[str | None, Any]:
    var_spec = text = None
    if hasattr(space, "__idoc__"):
        idoc = space.__idoc__
        if isinstance(idoc, dict):
            text = idoc["text"]
            var_spec = idoc.get("var")
        else:
            text = idoc
    return text, var_spec


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
    extra_text, extra_entries = get_extra_text_entries(space)
    entries = [
        (k, _render_space_compact(v, indentation=indentation + 2))
        for k, v in chain(space.spaces.items(), extra_entries.items())
    ]
    entries.sort(key=lambda x: x[0])
    l = max(len(k) for k, v in entries)
    inner = "</code><br/><code>".join(
        f'{"&nbsp;" * (indentation + 2)}"{k}"{" " * (l - len(k))}: {v}'
        for k, v in entries
    )
    return f"Dict({{</code><br/><code>{inner}</code><br/><code>{'&nbsp;' * indentation}}}){extra_text}"


def _render_space_compact(space: gym.spaces.Space, indentation: int = 0) -> str:
    if isinstance(space, gym.spaces.Dict):
        return _render_dict_compact(space, indentation=indentation)
    else:
        if isinstance(space, gym.spaces.Box):
            _, var_spec = get_text_var_idoc(space)
            if var_spec is None:
                var_spec = {}
            var_spec = {
                i: ReprString(v) for i, v in var_spec.items()
            }  # To ensure that we get no quotes
            space = copy.copy(space)
            space._shape = tuple(var_spec.get(i, e) for i, e in enumerate(space.shape))
        elif isinstance(space, gym.spaces.Discrete):
            _, limit = get_text_var_idoc(space)
            if limit is None:
                limit = space.n - 1
            else:
                limit = ReprString(limit)  # To ensure that we get no quotes
            space = copy.copy(space)
            space.n = limit
        return str(space)


def render_space_compact(space: gym.spaces.Space) -> str:
    return f"<code>{_render_space_compact(space)}</code>"


def is_marker_line(line: str, target: str) -> bool:
    return line.replace(" ", "").replace("\t", "").lower() == f"#!{target.lower()}"


def is_base_env(env: Any) -> bool:
    if env.__doc__ is None:
        return False
    lines = inspect.cleandoc(env.__doc__).splitlines()
    return any(is_marker_line(l, "ap_gym_base_env") for l in lines)


def is_wrapper(env: Any) -> bool:
    if env.__doc__ is None:
        return False
    lines = inspect.cleandoc(env.__doc__).splitlines()
    return any(is_marker_line(l, "ap_gym_wrapper") for l in lines)


def has_idoc(env: Any) -> bool:
    if not hasattr(env, "__idoc__") or env.__idoc__ is None:
        return False
    return isinstance(env.__idoc__, dict)


SectionType = "dict[str | None, str | SectionType]"


def render_sections(sections: SectionType, level: int = 0) -> str:
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


@dataclass(frozen=True)
class RenderOutput:
    class_name: str
    sections: SectionType
    properties: dict[str, str] | None = None
    has_idoc: bool = False
    full_env_name: str | None = None


def unwrap(env: gym.Env) -> tuple[gym.Env, list[gym.Wrapper], dict, dict | None]:
    base_env = env
    wrappers = []
    concrete_doc_parsed = None
    while not is_base_env(base_env):
        if has_idoc(base_env) and concrete_doc_parsed is None:
            concrete_doc_parsed = base_env.__idoc__
        if isinstance(base_env, gym.Wrapper) or isinstance(
            base_env, gym.vector.VectorWrapper
        ):
            if is_wrapper(base_env):
                wrappers.append(base_env)
            base_env = base_env.env
        elif isinstance(base_env, ap_gym.VectorToSingleWrapper):
            base_env = base_env.vector_env
        else:
            raise ValueError(
                f"Could not unwrap environment {base_env} further and environment is not a base environment."
            )
    if has_idoc(base_env) and concrete_doc_parsed is None:
        concrete_doc_parsed = base_env.__idoc__
    base_doc = inspect.cleandoc(base_env.__doc__)
    doc_parsed = yaml.safe_load(doc_extract_yaml(base_doc))

    return base_env, wrappers, doc_parsed, concrete_doc_parsed


def render_env(
    env: ap_gym.ActivePerceptionEnv,
    concrete: bool = False,
    base_title: str | None = None,
    base_name: str | None = None,
) -> RenderOutput:
    base_env, wrappers, doc_parsed, concrete_doc_parsed = unwrap(env)

    if concrete:
        doc_parsed.update(concrete_doc_parsed)

    all_versions = sorted(
        {
            parse_env_id(e)[2]
            for e in gym.registry
            if parse_env_id(e)[1] == env.spec.name
        }
    )
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

        env = ap_gym.make("{env.spec.id}")

        # Or for the vectorized version with 4 environments:
        envs = ap_gym.make_vec("{env.spec.id}", num_envs=4)
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
            "" if arg.default is arg.empty else f"`{repr(arg.default)}`",
            arg_descriptions[name].replace("\n", " "),
        )
        for name, arg in args.items()
        if name != "num_envs"
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
            if len(episode_end) > 0:
                episode_end += "\n\n"
            if len(conditions) > 1:
                l = "\n".join(
                    [
                        f"{i + 1}. {c[0].capitalize()}{c[1:]}"
                        for i, c in enumerate(conditions)
                    ]
                )
                episode_end += (
                    f"The episode ends with the {name} flag set if one of the following conditions is met:\n "
                    f"{l}"
                )
            else:
                episode_end += (
                    f"The episode ends with the {name} flag set if {conditions[0]}"
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
        ],
    )

    space_vars = doc_parsed.get("space_variables", [])
    if len(space_vars) > 0:
        properties += "\n\n where "
        if len(space_vars) > 1:
            properties += ", ".join(space_vars[:-1])
            if len(space_vars) > 2:
                properties += ", and "
            else:
                properties += " and "
        properties += space_vars[-1] + "."

    introduction = f'<p align="center"><img src="img/{env.spec.id}.gif" alt="{env.spec.id}" width="200px"/></p>'

    concrete_properties = None
    if concrete:
        introduction += (
            f"\n\n This environment is part of the {base_title}. Refer to the [{base_title} overview]({base_name}.md) "
            f"for a general description of these environments."
        )
        concrete_properties = list(doc_parsed.get("properties", {}).items())
        concrete_properties.insert(0, ("Environment ID", env.spec.id))
        introduction += f"\n\n{render_md_table(['', ''], [[f'**{n}**', p] for n, p in concrete_properties])}"

    body = {
        None: introduction,
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

    variant_full_names = [n for n in gym.registry if n.startswith(f"{env_name}-")]
    # Ensure every key is present only once while preserving order
    variant_names = list(
        dict.fromkeys(["-".join(n.split("-")[:-1]) for n in variant_full_names])
    )
    variant_names.remove(env.spec.name)
    variant_entries = []
    for n in variant_names:
        all_versions = sorted(
            {parse_env_id(e)[2] for e in gym.registry if parse_env_id(e)[1] == n}
        )
        latest_version = max(all_versions)
        full_name = f"{n}-v{latest_version}"
        variant_env = ap_gym.make(full_name)

        _, _, variant_doc_parsed, variant_concrete_doc_parsed = unwrap(variant_env)
        variant_doc_parsed.update(variant_concrete_doc_parsed)

        variant_entries.append(
            (
                full_name,
                variant_doc_parsed["description"],
                f'<img src="img/{full_name}.gif" alt="{full_name}" width="200px"/>',
            )
        )

    if len(variant_entries) > 0:
        body["Variants"] = render_md_table(
            ["Environment ID", "Description", "Preview"], variant_entries
        )

    sections = {env.spec.name if concrete else doc_parsed["title"]: body}

    return RenderOutput(
        type(base_env).__name__,
        sections,
        properties=concrete_properties,
        has_idoc=concrete_doc_parsed is not None,
        full_env_name=env.spec.id if concrete else None,
    )


def render_env_with_base(env_name: str) -> dict[str, RenderOutput]:
    all_versions = sorted(
        {parse_env_id(e)[2] for e in gym.registry if parse_env_id(e)[1] == env_name}
    )
    latest_version = max(all_versions)
    full_env_name = f"{env_name}-v{latest_version}"

    env = ap_gym.make(full_env_name)
    render_output_base = render_env(env)

    if render_output_base.has_idoc:
        render_output_concrete = render_env(
            env,
            concrete=True,
            base_name=render_output_base.class_name,
            base_title=list(render_output_base.sections.keys())[0],
        )
        output = {
            env.spec.name: render_output_concrete,
            render_output_base.class_name: render_output_base,
        }
    else:
        output = {
            env.spec.name: render_output_base,
        }

    return output


def dict_cut_recursive(dicts: list[SectionType]) -> SectionType | None:
    if len(dicts) == 0:
        raise ValueError("Cannot cut empty list of dicts.")
    if isinstance(dicts[0], dict):
        output = {k: dict_cut_recursive([d[k] for d in dicts]) for k in dicts[0]}
        return {k: v for k, v in output.items() if v is not None}
    else:
        if all(d == dicts[0] for d in dicts):
            return dicts[0]
        return None


def dict_diff_recursive(dict_a: SectionType, dict_b: SectionType) -> SectionType | None:
    if isinstance(dict_a, dict):
        output = {
            k: dict_diff_recursive(dict_a[k], dict_b[k]) if k in dict_b else dict_a[k]
            for k in dict_a
        }
        return {k: v for k, v in output.items() if v is not None}
    else:
        if dict_a == dict_b:
            return None
        return dict_a


def write_file(path: Path, content: str) -> None:
    while content.endswith("\n\n"):
        content = content[:-1]
    with path.open("w") as f:
        f.write(content)


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

    exclude_list = PURE_GYM_ENVS

    if "ap_gym" not in args.modules:
        exclude_list = exclude_list.union(AP_GYM_ENVS)

    for module in args.modules:
        if module != "ap_gym":
            __import__(module)

    if args.include is not None:
        env_names = set(args.include)
    else:
        env_names_with_versions = set(gym.registry) - exclude_list
        env_names = set(parse_env_id(e)[1] for e in env_names_with_versions)

    variants = []
    for env_name in env_names:
        if any(env_name.startswith(f"{e}-") for e in env_names):
            variants.append(env_name)
    env_names -= set(variants)

    args.output_dir.mkdir(exist_ok=True)

    bases = defaultdict(list)
    envs = {}

    for env_name in sorted(env_names):
        print(f"Rendering {env_name}...")
        rendered_dict = render_env_with_base(env_name)
        if len(rendered_dict) == 1:
            envs[env_name] = (None, rendered_dict[env_name])
        else:
            base_name = list(set(rendered_dict.keys()) - {env_name})[0]
            envs[env_name] = (base_name, rendered_dict[env_name])
            bases[base_name].append(rendered_dict[base_name])

    aggregated_bases = {}

    for base_name, rendered_lst in bases.items():
        aggregated = dict_cut_recursive([r.sections for r in rendered_lst])
        base_title = list(aggregated.keys())[0]
        if "Example Usage" in aggregated[base_title]:
            del aggregated[base_title]["Example Usage"]
        if "Version History" in aggregated[base_title]:
            del aggregated[base_title]["Version History"]

        concrete_envs = [
            (env_name, rendered.full_env_name, rendered.properties)
            for env_name, (bn, rendered) in envs.items()
            if bn == base_name
        ]

        if len(concrete_envs) > 0:
            full_properties = [
                [("Environment ID", f"[{full_env_name}]({env_name}.md)")] + props[1:]
                for env_name, full_env_name, props in concrete_envs
            ]
            headers = [e for e, _ in full_properties[0]]
            full_properties_dict = list(map(dict, full_properties))
            aggregated[base_title]["Overview of Implemented Environments"] = (
                render_md_table(
                    headers, [[l[e] for e in headers] for l in full_properties_dict]
                )
            )

        aggregated_bases[base_name] = aggregated

    for base_name, rendered in aggregated_bases.items():
        write_file(args.output_dir / f"{base_name}.md", render_sections(rendered))

    for env_name, (base_name, rendered) in envs.items():
        sections = rendered.sections
        if base_name is not None:
            base_title = list(aggregated_bases[base_name].keys())[0]
            sections[env_name] = dict_diff_recursive(
                sections[env_name],
                aggregated_bases[base_name][base_title],
            )
        write_file(args.output_dir / f"{env_name}.md", render_sections(sections))
