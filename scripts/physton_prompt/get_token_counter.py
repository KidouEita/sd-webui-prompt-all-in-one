from modules import script_callbacks, extra_networks, prompt_parser, sd_models
from functools import reduce

# 嘗試載入 sd_hijack / model_hijack（A1111 / Forge Classic 才有）
try:
    from modules.sd_hijack import model_hijack
except (ImportError, ModuleNotFoundError):
    model_hijack = None


def get_token_counter(text, steps):
    try:
        try:
            text, _ = extra_networks.parse_prompt(text)
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
            prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)
        except Exception:
            prompt_schedules = [[[steps, text]]]

        # 判斷是否 Forge
        try:
            from modules_forge import forge_version
            forge = True
        except:
            forge = False

        flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules)
        prompts = [prompt_text for step, prompt_text in flat_prompts]

        # 🚨 Forge Neo / 沒有 hijack：直接停用 token counter
        if model_hijack is None:
            return {"token_count": 0, "max_length": 0}

        # A1111 / Forge Classic
        if forge:
            cond_stage_model = sd_models.model_data.sd_model.cond_stage_model
            token_count, max_length = max(
                [model_hijack.get_prompt_lengths(prompt, cond_stage_model) for prompt in prompts],
                key=lambda args: args[0]
            )
        else:
            token_count, max_length = max(
                [model_hijack.get_prompt_lengths(prompt) for prompt in prompts],
                key=lambda args: args[0]
            )

        return {"token_count": token_count, "max_length": max_length}

    except Exception:
        return {"token_count": 0, "max_length": 0}
