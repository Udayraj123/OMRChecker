# https://docs.python.org/3/tutorial/modules.html#:~:text=The%20__init__.py,on%20the%20module%20search%20path.
from jsonschema import Draft202012Validator

from src.schemas.config_schema import CONFIG_SCHEMA
from src.schemas.evaluation_schema import EVALUATION_SCHEMA
from src.schemas.template_schema import TEMPLATE_SCHEMA

SCHEMA_JSONS = {
    "config": CONFIG_SCHEMA,
    "evaluation": EVALUATION_SCHEMA,
    "template": TEMPLATE_SCHEMA,
}

SCHEMA_VALIDATORS = {
    "config": Draft202012Validator(CONFIG_SCHEMA),
    "evaluation": Draft202012Validator(EVALUATION_SCHEMA),
    "template": Draft202012Validator(TEMPLATE_SCHEMA),
}
