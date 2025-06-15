import yaml
import re

def get_file_content(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def preprocess_yaml_content(content):

    # Preprocess by replacing 'on:' with 'on': 'to avoid parsing to Boolean values
    content = re.sub(r'(?<!\w)on:(?!\w)', "'on':", content)

    return content

def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten nested dictionaries and lists into a unary pattern"""
    items = []
    if not isinstance(d, dict):
        return {}

    for k, v in d.items():
        k = str(k).replace(".", "-")
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):  # processing list
            # If all elements in the list are single values (not nested values)
            if all(not isinstance(item, (dict, list)) for item in v):
                items.append((new_key, v))
            else:
                for index, item in enumerate(v):
                    list_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, list_key, sep=sep).items())
                    else:
                        items.append((list_key, item))
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_config_data(content):
    """Read YAML files and return flattened data"""
    if content=="No content found":
        return {}
    try:
        content = preprocess_yaml_content(content)
        data = yaml.safe_load(content)
        data = flatten_dict(data)
        return data
    except Exception as e:
        return None

def get_config_data_from_file(filename):
    """Read YAML files and return flattened data"""
    content = get_file_content(filename)
    data = get_config_data(content)
    return data

def focus_config_data(flattened_config, AI_reviewer_mention,):
    """
    Focus only on changes in specific configuration items among all configuration items:
    1. Trigger conditions related to 'on'
    2. Important configurations related to AI review actions:
    2.1 Version of action used;
    2.2 Model used for actions;
    2.3 Prompt used for actions;

    : paramflattened_config: flattened configuration dictionary
    : return: filtered configuration dictionary
    """
    focused_config = {}

    # Keep key value pairs related to 'on'
    for key, value in flattened_config.items():
        if key.startswith("on.") or key.startswith("on:") or key=='on':
            focused_config[key] = value

    # Keep key value pairs related to AI review actions
    is_illegal = False
    for key, value in flattened_config.items():
        if isinstance(value, str) and AI_reviewer_mention in value:
            job_path = key.split(".")
            job_prefix = job_path[0]
            if job_prefix!="jobs":
                is_illegal = True
                break
            job_name = job_path[1]
            step_index = job_path[2]
            old_prefix = job_prefix + "." + job_name
            new_prefix = job_prefix + "." + "jobName" # Not paying attention to the specific name of this action
            parent_key = key.rsplit(".", 1)[0]  # Find parent key
            for sub_key, sub_value in flattened_config.items():
                if sub_key.startswith(parent_key) or sub_key==old_prefix+".if":
                    sub_key = sub_key.replace(old_prefix,new_prefix)
                    sub_key = sub_key.replace(step_index, "steps[i]")
                    focused_config[sub_key] = sub_value
    if is_illegal:
        return None
    else:
        return focused_config

def compare_configs(flat_config1, flat_config2):
    """
    Compare two flattened configuration dictionaries and identify the differences.
    : paramflat_comfig1: The first flattened configuration dictionary
    : paramflat_comfig2: The second flattened configuration dictionary
    : return: List of difference results, each item is (recursive key name, pre modification value, modified value, modification type)
    """
    differences = []

    all_keys = set(flat_config1.keys()) | set(flat_config2.keys())  # The union of all keys

    for key in all_keys:
        value1 = flat_config1.get(key, None)
        value2 = flat_config2.get(key, None)

        if key not in flat_config1:
            differences.append((key, "Not_Exist", value2, 1))  # Added key
        elif key not in flat_config2:
            differences.append((key, value1, "Not_Exist", -1))  # deleted key
        elif value1 != value2:
            differences.append((key, value1, value2, 0))  # modified key

    return differences

def compare_config_content(config_content_1, config_content_2, ai_reviewer_mention):
    config_data_1 = get_config_data(config_content_1)
    config_data_2 = get_config_data(config_content_2)
    if config_data_1!=None and config_data_2!=None:
        config1 = focus_config_data(config_data_1, ai_reviewer_mention)
        config2 = focus_config_data(config_data_2, ai_reviewer_mention)
        if config1!=None and config2!=None:
            differences = compare_configs(config1, config2)
            return differences
    return None


if __name__ == "__main__":

    AI_reviewer_mention = "anc95/ChatGPT-CodeReview"
    filename_1 = "test_case/cr1.yml"
    filename_2 = "test_case/cr2.yml"

    config1 = focus_config_data(get_config_data_from_file(filename_1),AI_reviewer_mention)
    print(get_config_data_from_file(filename_1))
    print(config1)
    config2 = focus_config_data(get_config_data_from_file(filename_2),AI_reviewer_mention)


    differences = compare_configs(config1, config2)

    print("Differences between the two configurations:")
    for diff in differences:
        print(f"Path: {diff[0]}, Before: {diff[1]}, After: {diff[2]}, Change Type: {diff[3]}")
