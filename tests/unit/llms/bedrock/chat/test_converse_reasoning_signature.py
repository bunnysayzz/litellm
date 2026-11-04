"""
Regression test for https://github.com/BerriAI/litellm/issues/49969

Bedrock Converse must not replay reasoningText.signature to a non-Anthropic target model.
A signature produced by Claude is only meaningful to Claude; sending it to GLM, Nova, Llama
or any other Bedrock model raises:
    BedrockException - This model doesn't support the reasoningContent.reasoningText.signature field.
"""
from litellm.litellm_core_utils.prompt_templates.factory import (
    BedrockConverseMessagesProcessor,
)
from litellm.types.llms.openai import ChatCompletionThinkingBlock


def test_signature_dropped_for_non_claude_target():
    """A thinking block with a signature must not carry the signature to a non-Claude target."""
    thinking_block = ChatCompletionThinkingBlock(
        type="thinking",
        thinking="Let me think about this carefully.",
        signature="EqoBCkgIARABGAIiQKlDOqwIJ...",
    )

    result = BedrockConverseMessagesProcessor._to_reasoning_content_block(
        thinking_block, to_claude=False
    )

    assert "reasoningText" in result
    assert result["reasoningText"].get("signature") is None, (
        "signature must not be forwarded to a non-Anthropic Converse target"
    )
    assert result["reasoningText"]["text"] == "Let me think about this carefully."


def test_signature_kept_for_claude_target():
    """A thinking block with a signature must still carry the signature to a Claude target."""
    thinking_block = ChatCompletionThinkingBlock(
        type="thinking",
        thinking="Let me think about this carefully.",
        signature="EqoBCkgIARABGAIiQKlDOqwIJ...",
    )

    result = BedrockConverseMessagesProcessor._to_reasoning_content_block(
        thinking_block, to_claude=True
    )

    assert "reasoningText" in result
    assert result["reasoningText"].get("signature") == "EqoBCkgIARABGAIiQKlDOqwIJ..."


def test_no_signature_block_unaffected_by_to_claude():
    """A thinking block without a signature must behave the same regardless of to_claude."""
    thinking_block = ChatCompletionThinkingBlock(
        type="thinking",
        thinking="No signature here.",
    )

    for to_claude in (True, False):
        result = BedrockConverseMessagesProcessor._to_reasoning_content_block(
            thinking_block, to_claude=to_claude
        )
        assert "reasoningText" in result
        assert result["reasoningText"].get("signature") is None
        assert result["reasoningText"]["text"] == "No signature here."
