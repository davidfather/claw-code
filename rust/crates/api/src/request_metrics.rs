use serde::{Deserialize, Serialize};

use crate::types::{MessageRequest, ToolChoice};

/// Request-side metrics that can be measured before a provider answers.
///
/// These values intentionally describe the request envelope, not the final
/// model result. For example, `estimated_input_tokens` is calculated before the
/// request is sent, while `usage.input_tokens` is provider-reported after a
/// response. Keeping both lets callers compare the local estimate against the
/// provider's actual accounting when the provider exposes usage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageRequestMetrics {
    /// Number of conversation messages inside one `MessageRequest`.
    ///
    /// This is not the number of API requests. A single request can contain
    /// multiple messages when history, assistant turns, or tool results need to
    /// be replayed for context.
    pub message_count: usize,
    /// Byte length of the optional system prompt text.
    ///
    /// Fast lanes such as DirectFinal/DirectLlm should normally keep this at 0
    /// unless the turn explicitly needs instructions or memory.
    pub system_prompt_bytes: usize,
    /// Number of tool definitions attached to this request.
    pub tool_count: usize,
    /// Total serialized size of all tool schemas attached to this request.
    ///
    /// This is the aggregate `tools` array size, not a per-tool measurement.
    /// It is the first-order signal for "tool schema payload is making local
    /// LLM calls slow"; per-tool breakdown can be added later if needed.
    pub tool_schema_bytes: usize,
    /// Tool selection policy sent with the request, such as `auto` or `any`.
    pub tool_choice: Option<String>,
    /// Whether the provider was asked to stream output chunks as they are made.
    ///
    /// Streaming requests can report first-token latency. Non-streaming
    /// requests usually only have total request latency because the full answer
    /// arrives at once.
    pub stream: bool,
    /// Size of the provider-facing serialized request payload.
    ///
    /// This captures metadata plus prompt/tool payload, so it is useful for
    /// detecting accidental large requests even when token estimates are rough.
    pub serialized_request_bytes: usize,
    /// Approximate input tokens computed locally before sending the request.
    ///
    /// This is a stable comparison metric, not provider truth. It estimates the
    /// prompt-bearing parts of the request: messages, system prompt, tools, and
    /// tool choice.
    pub estimated_input_tokens: u32,
    /// Maximum output tokens requested from the provider.
    ///
    /// This is the configured upper bound (`max_tokens`), not the number of
    /// tokens the model eventually generated (`usage.output_tokens`).
    pub requested_output_tokens: u32,
}

impl MessageRequestMetrics {
    #[must_use]
    pub fn from_request(request: &MessageRequest) -> Self {
        Self {
            message_count: request.messages.len(),
            system_prompt_bytes: request.system.as_ref().map_or(0, |value| value.len()),
            tool_count: request.tools.as_ref().map_or(0, Vec::len),
            tool_schema_bytes: request.tools.as_ref().map_or(0, serialized_json_bytes),
            tool_choice: request.tool_choice.as_ref().map(tool_choice_label),
            stream: request.stream,
            serialized_request_bytes: serialized_json_bytes(request),
            estimated_input_tokens: estimate_message_request_input_tokens(request),
            requested_output_tokens: request.max_tokens,
        }
    }
}

#[must_use]
pub fn estimate_message_request_input_tokens(request: &MessageRequest) -> u32 {
    let mut estimate = estimate_serialized_tokens(&request.messages);
    estimate = estimate.saturating_add(estimate_serialized_tokens(&request.system));
    estimate = estimate.saturating_add(estimate_serialized_tokens(&request.tools));
    estimate = estimate.saturating_add(estimate_serialized_tokens(&request.tool_choice));
    estimate
}

fn estimate_serialized_tokens<T: Serialize>(value: &T) -> u32 {
    serialized_json_bytes(value)
        .checked_div(4)
        .and_then(|value| u32::try_from(value + 1).ok())
        .unwrap_or(u32::MAX)
}

fn serialized_json_bytes<T: Serialize>(value: &T) -> usize {
    serde_json::to_vec(value).map_or(0, |bytes| bytes.len())
}

fn tool_choice_label(tool_choice: &ToolChoice) -> String {
    match tool_choice {
        ToolChoice::Auto => "auto".to_string(),
        ToolChoice::Any => "any".to_string(),
        ToolChoice::Tool { name } => format!("tool:{name}"),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::types::{InputMessage, MessageRequest, ToolChoice, ToolDefinition};

    use super::MessageRequestMetrics;

    #[test]
    fn request_metrics_describe_request_payload_not_response_usage() {
        let request = MessageRequest {
            model: "ollama/llama3.1:8b".to_string(),
            max_tokens: 128,
            messages: vec![InputMessage::user_text("Reply with exactly: pong")],
            system: Some("system prompt".to_string()),
            tools: Some(vec![ToolDefinition {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    }
                }),
            }]),
            tool_choice: Some(ToolChoice::Auto),
            response_format: None,
            stream: true,
        };

        let metrics = MessageRequestMetrics::from_request(&request);

        assert_eq!(metrics.message_count, 1);
        assert_eq!(metrics.system_prompt_bytes, "system prompt".len());
        assert_eq!(metrics.tool_count, 1);
        assert!(metrics.tool_schema_bytes > 0);
        assert_eq!(metrics.tool_choice.as_deref(), Some("auto"));
        assert!(metrics.stream);
        assert!(metrics.serialized_request_bytes > metrics.tool_schema_bytes);
        assert!(metrics.estimated_input_tokens > 0);
        assert_eq!(metrics.requested_output_tokens, 128);
    }

    #[test]
    fn request_metrics_keep_absent_tool_schema_at_zero_bytes() {
        let request = MessageRequest {
            model: "ollama/llama3.1:8b".to_string(),
            max_tokens: 128,
            messages: vec![InputMessage::user_text("pong")],
            system: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            stream: false,
        };

        let metrics = MessageRequestMetrics::from_request(&request);

        assert_eq!(metrics.tool_count, 0);
        assert_eq!(metrics.tool_schema_bytes, 0);
        assert_eq!(metrics.system_prompt_bytes, 0);
        assert!(!metrics.stream);
    }
}
