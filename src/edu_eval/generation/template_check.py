from typing import Any


def template_report(tokenizer: Any, enable_thinking: bool | None) -> dict:
    """Render the generation prompt exactly as the runner will and inspect its tail."""
    template = getattr(tokenizer, "chat_template", None) or ""
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "halo"}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )

    return {
        "template_mentions_enable_thinking": "enable_thinking" in template,
        "open_think_block": prompt.rfind("<think>") > prompt.rfind("</think>"),
        "prompt_tail": prompt[-80:],
    }


def require_thinking_off(tokenizer: Any, enable_thinking: bool | None) -> dict:
    """Fail loudly when thinking should be off but the prompt still opens <think>.

    Unknown chat_template kwargs are silently ignored by jinja, so passing
    enable_thinking=False proves nothing until the rendered prompt is checked.
    """
    report = template_report(tokenizer, enable_thinking)

    if enable_thinking is False and report["open_think_block"]:
        raise RuntimeError(
            "enable_thinking=False tapi prompt masih membuka <think> "
            f"(ujung prompt: {report['prompt_tail']!r}). Template model ini tidak "
            "menghormati flag tersebut; cari mekanisme lain sebelum generate."
        )

    return report
