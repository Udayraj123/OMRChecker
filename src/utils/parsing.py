import json

from deepmerge import Merger

OVERRIDE_MERGER = Merger(
    # pass in a list of tuples,with the
    # strategies you are looking to apply
    # to each type.
    [
        # (list, ["prepend"]),
        (dict, ["merge"])
    ],
    # next, choose the fallback strategies,
    # applied to all other types:
    ["override"],
    # finally, choose the strategies in
    # the case where the types conflict:
    ["override"],
)


def load_json(path, **rest):
    with open(path, "r") as f:
        loaded = json.load(f, **rest)
    return loaded


def get_concatenated_response(omr_response, template):
    concatenated_response = {}

    # TODO: get correct local/global emptyVal here for each question
    # symbol for absent response
    unmarked_symbol = ""

    # Multi-column/multi-row questions which need to be concatenated
    for q_no, resp_keys in template.concatenations.items():
        concatenated_response[q_no] = "".join(
            [omr_response.get(k, unmarked_symbol) for k in resp_keys]
        )

    # Single-column/single-row questions
    for q_no in template.singles:
        concatenated_response[q_no] = omr_response.get(q_no, unmarked_symbol)

    # Note: concatenations and singles together should be mutually exclusive
    # and should cover all questions in the template(exhaustive)
    # TODO: ^add a warning if omr_response has unused keys remaining
    return concatenated_response


def evaluate_concatenated_response(
    concatenated_response, evaluation_config, should_explain_scoring=False
):
    if should_explain_scoring:
        pass
    # todo: code for evaluation

    return 0
