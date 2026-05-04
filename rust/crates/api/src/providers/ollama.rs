use std::collections::VecDeque;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest,
    MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock, ResponseFormat,
    StreamEvent, ToolDefinition, ToolResultContentBlock, Usage,
};

use super::{
    model_name_for_provider_request, preflight_message_request, Provider, ProviderFuture,
    ProviderKind,
};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";
const REQUEST_ID_HEADER: &str = "request-id";
const ALT_REQUEST_ID_HEADER: &str = "x-request-id";
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(200);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(2);
const DEFAULT_MAX_RETRIES: u32 = 2;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    api_key: Option<String>,
    base_url: String,
    max_retries: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl OllamaClient {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: Some(api_key.into()),
            base_url: read_base_url(),
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    #[must_use]
    pub fn without_auth() -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: None,
            base_url: read_base_url(),
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    pub fn from_env() -> Result<Self, ApiError> {
        Ok(read_env_non_empty("OLLAMA_API_KEY")?.map_or_else(Self::without_auth, Self::new))
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    #[must_use]
    pub fn with_retry_policy(
        mut self,
        max_retries: u32,
        initial_backoff: Duration,
        max_backoff: Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.initial_backoff = initial_backoff;
        self.max_backoff = max_backoff;
        self
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let request = MessageRequest {
            stream: false,
            ..request.clone()
        };
        preflight_message_request(&request)?;
        let response = self.send_with_retry(&request).await?;
        let request_id = request_id_from_headers(response.headers());
        let payload = response.json::<OllamaChatResponse>().await?;
        let mut normalized = normalize_response(&request.model, payload);
        if normalized.request_id.is_none() {
            normalized.request_id = request_id;
        }
        Ok(normalized)
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        preflight_message_request(request)?;
        let response = self
            .send_with_retry(&request.clone().with_streaming())
            .await?;
        Ok(MessageStream {
            request_id: request_id_from_headers(response.headers()),
            response,
            parser: NdjsonParser::new(),
            pending: VecDeque::new(),
            done: false,
            state: StreamState::new(request.model.clone()),
        })
    }

    async fn send_with_retry(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;

        let last_error = loop {
            attempts += 1;
            let retryable_error = match self.send_raw_request(request).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                Err(error) => return Err(error),
            };

            if attempts > self.max_retries {
                break retryable_error;
            }

            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        };

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error),
        })
    }

    async fn send_raw_request(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let request_url = chat_endpoint(&self.base_url);
        let request = self
            .http
            .post(&request_url)
            .header("content-type", "application/json")
            .json(&build_chat_request(request));
        let request =
            if let Some(api_key) = self.api_key.as_deref().filter(|value| !value.is_empty()) {
                request.bearer_auth(api_key)
            } else {
                request
            };
        request.send().await.map_err(ApiError::from)
    }

    fn backoff_for_attempt(&self, attempt: u32) -> Result<Duration, ApiError> {
        let Some(multiplier) = 1_u32.checked_shl(attempt.saturating_sub(1)) else {
            return Err(ApiError::BackoffOverflow {
                attempt,
                base_delay: self.initial_backoff,
            });
        };
        Ok(self
            .initial_backoff
            .checked_mul(multiplier)
            .map_or(self.max_backoff, |delay| delay.min(self.max_backoff)))
    }
}

impl Provider for OllamaClient {
    type Stream = MessageStream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse> {
        Box::pin(async move { self.send_message(request).await })
    }

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream> {
        Box::pin(async move { self.stream_message(request).await })
    }
}

#[derive(Debug)]
pub struct MessageStream {
    request_id: Option<String>,
    response: reqwest::Response,
    parser: NdjsonParser,
    pending: VecDeque<StreamEvent>,
    done: bool,
    state: StreamState,
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            if self.done {
                self.pending.extend(self.state.finish());
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }
                return Ok(None);
            }

            match self.response.chunk().await? {
                Some(chunk) => {
                    for parsed in self.parser.push(&chunk)? {
                        self.pending.extend(self.state.ingest_chunk(parsed));
                    }
                }
                None => {
                    for parsed in self.parser.finish()? {
                        self.pending.extend(self.state.ingest_chunk(parsed));
                    }
                    self.done = true;
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct NdjsonParser {
    buffer: Vec<u8>,
}

impl NdjsonParser {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, chunk: &[u8]) -> Result<Vec<OllamaChatResponse>, ApiError> {
        self.buffer.extend_from_slice(chunk);
        let mut responses = Vec::new();

        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.buffer.drain(..=position).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line).trim().to_string();
            if !line.is_empty() {
                responses.push(serde_json::from_str(&line)?);
            }
        }

        Ok(responses)
    }

    fn finish(&mut self) -> Result<Vec<OllamaChatResponse>, ApiError> {
        let trailing = String::from_utf8_lossy(&self.buffer).trim().to_string();
        self.buffer.clear();
        if trailing.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![serde_json::from_str(&trailing)?])
        }
    }
}

#[derive(Debug)]
struct StreamState {
    model: String,
    message_started: bool,
    text_started: bool,
    text_finished: bool,
    finished: bool,
    content: String,
    tool_calls: Vec<OllamaToolCall>,
    usage: Usage,
    stop_reason: Option<String>,
}

impl StreamState {
    const fn new(model: String) -> Self {
        Self {
            model,
            message_started: false,
            text_started: false,
            text_finished: false,
            finished: false,
            content: String::new(),
            tool_calls: Vec::new(),
            usage: Usage {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 0,
            },
            stop_reason: None,
        }
    }

    fn ingest_chunk(&mut self, chunk: OllamaChatResponse) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if !self.message_started {
            self.message_started = true;
            events.push(StreamEvent::MessageStart(MessageStartEvent {
                message: MessageResponse {
                    id: message_id_for_chunk(&chunk),
                    kind: "message".to_string(),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    model: chunk.model.clone().unwrap_or_else(|| self.model.clone()),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: self.usage.clone(),
                    request_id: None,
                },
            }));
        }

        if let Some(text) = chunk.message.content.filter(|value| !value.is_empty()) {
            if !self.text_started {
                self.text_started = true;
                events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    index: 0,
                    content_block: OutputContentBlock::Text {
                        text: String::new(),
                    },
                }));
            }
            self.content.push_str(&text);
            events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: 0,
                delta: ContentBlockDelta::TextDelta { text },
            }));
        }

        self.tool_calls.extend(chunk.message.tool_calls);
        if let Some(prompt_eval_count) = chunk.prompt_eval_count {
            self.usage.input_tokens = prompt_eval_count;
        }
        if let Some(eval_count) = chunk.eval_count {
            self.usage.output_tokens = eval_count;
        }
        if chunk.done {
            self.stop_reason = Some(normalize_done_reason(chunk.done_reason.as_deref()));
            self.finished = true;
        }

        events
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        if !self.message_started {
            return Vec::new();
        }
        let mut events = Vec::new();
        if self.text_started && !self.text_finished {
            self.text_finished = true;
            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: 0,
            }));
        }

        for (index, tool_call) in self.tool_calls.iter().enumerate() {
            let block_index = index as u32 + 1;
            let id = format!("ollama_call_{index}");
            let name = tool_call.function.name.clone();
            let arguments = normalize_tool_arguments(&tool_call.function.arguments);
            events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                index: block_index,
                content_block: OutputContentBlock::ToolUse {
                    id,
                    name,
                    input: json!({}),
                },
            }));
            events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: block_index,
                delta: ContentBlockDelta::InputJsonDelta {
                    partial_json: arguments.to_string(),
                },
            }));
            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: block_index,
            }));
        }

        events.push(StreamEvent::MessageDelta(MessageDeltaEvent {
            delta: MessageDelta {
                stop_reason: Some(
                    self.stop_reason
                        .clone()
                        .unwrap_or_else(|| "end_turn".to_string()),
                ),
                stop_sequence: None,
            },
            usage: self.usage.clone(),
        }));
        events.push(StreamEvent::MessageStop(MessageStopEvent {}));
        self.message_started = false;
        events
    }
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OllamaTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    num_predict: u32,
}

#[derive(Debug, Serialize)]
struct OllamaChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Debug, Serialize)]
struct OllamaTool {
    r#type: &'static str,
    function: OllamaToolFunction,
}

#[derive(Debug, Serialize)]
struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OllamaToolCallFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    message: OllamaResponseMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

fn build_chat_request(request: &MessageRequest) -> OllamaChatRequest {
    let model = model_name_for_provider_request(&request.model, ProviderKind::Ollama);
    let tools = request
        .tools
        .as_deref()
        .map_or_else(Vec::new, translate_tools);
    OllamaChatRequest {
        model,
        messages: translate_messages(&request.messages, request.system.as_deref()),
        stream: request.stream,
        tools,
        format: request.response_format.and_then(|format| match format {
            ResponseFormat::JsonObject => Some("json"),
        }),
        options: (request.max_tokens > 0).then_some(OllamaOptions {
            num_predict: request.max_tokens,
        }),
    }
}

fn translate_messages(messages: &[InputMessage], system: Option<&str>) -> Vec<OllamaChatMessage> {
    let mut translated = Vec::new();
    if let Some(system) = system.filter(|value| !value.is_empty()) {
        translated.push(OllamaChatMessage {
            role: "system".to_string(),
            content: system.to_string(),
            tool_calls: Vec::new(),
        });
    }

    for message in messages {
        let text = text_from_blocks(&message.content);
        let tool_calls = tool_calls_from_blocks(&message.content);
        let tool_results = tool_result_texts_from_blocks(&message.content);
        if !text.is_empty() || !tool_calls.is_empty() {
            translated.push(OllamaChatMessage {
                role: if message.role == "assistant" {
                    "assistant".to_string()
                } else {
                    "user".to_string()
                },
                content: text,
                tool_calls,
            });
        }
        for tool_result in tool_results {
            translated.push(OllamaChatMessage {
                role: "tool".to_string(),
                content: tool_result,
                tool_calls: Vec::new(),
            });
        }
    }

    translated
}

fn text_from_blocks(blocks: &[InputContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            InputContentBlock::Text { text } => Some(text.as_str()),
            InputContentBlock::ToolUse { .. } | InputContentBlock::ToolResult { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn tool_calls_from_blocks(blocks: &[InputContentBlock]) -> Vec<OllamaToolCall> {
    blocks
        .iter()
        .filter_map(|block| match block {
            InputContentBlock::ToolUse { name, input, .. } => Some(OllamaToolCall {
                function: OllamaToolCallFunction {
                    name: name.clone(),
                    arguments: input.clone(),
                },
            }),
            InputContentBlock::Text { .. } | InputContentBlock::ToolResult { .. } => None,
        })
        .collect()
}

fn tool_result_texts_from_blocks(blocks: &[InputContentBlock]) -> Vec<String> {
    blocks
        .iter()
        .filter_map(|block| match block {
            InputContentBlock::ToolResult { content, .. } => {
                Some(flatten_tool_result_content(content))
            }
            InputContentBlock::Text { .. } | InputContentBlock::ToolUse { .. } => None,
        })
        .filter(|value| !value.is_empty())
        .collect()
}

fn flatten_tool_result_content(content: &[ToolResultContentBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            ToolResultContentBlock::Text { text } => text.clone(),
            ToolResultContentBlock::Json { value } => value.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn translate_tools(tools: &[ToolDefinition]) -> Vec<OllamaTool> {
    tools
        .iter()
        .map(|tool| OllamaTool {
            r#type: "function",
            function: OllamaToolFunction {
                name: tool.name.clone(),
                description: tool.description.clone().unwrap_or_default(),
                parameters: tool.input_schema.clone(),
            },
        })
        .collect()
}

fn normalize_response(model: &str, response: OllamaChatResponse) -> MessageResponse {
    let mut content = Vec::new();
    if let Some(text) = response.message.content.filter(|value| !value.is_empty()) {
        content.push(OutputContentBlock::Text { text });
    }
    for (index, tool_call) in response.message.tool_calls.into_iter().enumerate() {
        content.push(OutputContentBlock::ToolUse {
            id: format!("ollama_call_{index}"),
            name: tool_call.function.name,
            input: normalize_tool_arguments(&tool_call.function.arguments),
        });
    }

    MessageResponse {
        id: message_id_for_parts(response.created_at.as_deref(), response.model.as_deref()),
        kind: "message".to_string(),
        role: response
            .message
            .role
            .unwrap_or_else(|| "assistant".to_string()),
        content,
        model: response.model.unwrap_or_else(|| model.to_string()),
        stop_reason: Some(normalize_done_reason(response.done_reason.as_deref())),
        stop_sequence: None,
        usage: Usage {
            input_tokens: response.prompt_eval_count.unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: response.eval_count.unwrap_or(0),
        },
        request_id: None,
    }
}

fn normalize_tool_arguments(arguments: &Value) -> Value {
    match arguments {
        Value::String(value) => {
            serde_json::from_str(value).unwrap_or_else(|_| json!({ "raw": value }))
        }
        other => other.clone(),
    }
}

fn normalize_done_reason(done_reason: Option<&str>) -> String {
    match done_reason {
        Some("stop") | None => "end_turn".to_string(),
        Some("length") => "max_tokens".to_string(),
        Some("tool_calls") => "tool_use".to_string(),
        Some(other) => other.to_string(),
    }
}

fn message_id_for_chunk(chunk: &OllamaChatResponse) -> String {
    message_id_for_parts(chunk.created_at.as_deref(), chunk.model.as_deref())
}

fn message_id_for_parts(created_at: Option<&str>, model: Option<&str>) -> String {
    let suffix = created_at
        .filter(|value| !value.is_empty())
        .or(model.filter(|value| !value.is_empty()))
        .unwrap_or("message");
    format!("ollama_{suffix}")
}

fn chat_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    let without_v1 = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
    if without_v1.ends_with("/api/chat") {
        without_v1.to_string()
    } else {
        format!("{without_v1}/api/chat")
    }
}

#[must_use]
pub fn read_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
}

fn read_env_non_empty(key: &str) -> Result<Option<String>, ApiError> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Ok(Some(value)),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(ApiError::from(error)),
    }
}

fn request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_ID_HEADER)
        .or_else(|| headers.get(ALT_REQUEST_ID_HEADER))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let request_id = request_id_from_headers(response.headers());
    let body = response.text().await.unwrap_or_default();
    let message = serde_json::from_str::<ErrorEnvelope>(&body)
        .ok()
        .and_then(|envelope| envelope.error.message);
    Err(ApiError::Api {
        status,
        error_type: Some("ollama_error".to_string()),
        message,
        request_id,
        body,
        retryable: matches!(status.as_u16(), 408 | 409 | 429 | 500..=599),
    })
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{
        build_chat_request, chat_endpoint, normalize_response, normalize_tool_arguments,
        NdjsonParser, OllamaChatResponse, OllamaResponseMessage, OllamaToolCall,
        OllamaToolCallFunction,
    };
    use crate::types::{
        InputContentBlock, InputMessage, MessageRequest, OutputContentBlock, ResponseFormat,
        ToolDefinition, ToolResultContentBlock,
    };
    use serde_json::json;

    #[test]
    fn request_translation_uses_native_ollama_shape() {
        let payload = serde_json::to_value(build_chat_request(&MessageRequest {
            model: "ollama/llama3.2".to_string(),
            max_tokens: 64,
            messages: vec![InputMessage {
                role: "user".to_string(),
                content: vec![
                    InputContentBlock::Text {
                        text: "hello".to_string(),
                    },
                    InputContentBlock::ToolResult {
                        tool_use_id: "tool_1".to_string(),
                        content: vec![ToolResultContentBlock::Json {
                            value: json!({"ok": true}),
                        }],
                        is_error: false,
                    },
                ],
            }],
            system: Some("be helpful".to_string()),
            tools: Some(vec![ToolDefinition {
                name: "weather".to_string(),
                description: Some("Get weather".to_string()),
                input_schema: json!({"type": "object"}),
            }]),
            tool_choice: None,
            response_format: Some(ResponseFormat::JsonObject),
            stream: false,
        }))
        .expect("serialize ollama payload");

        assert_eq!(payload["model"], json!("llama3.2"));
        assert_eq!(payload["messages"][0]["role"], json!("system"));
        assert_eq!(payload["messages"][1]["role"], json!("user"));
        assert_eq!(payload["messages"][2]["role"], json!("tool"));
        assert_eq!(payload["tools"][0]["type"], json!("function"));
        assert_eq!(payload["format"], json!("json"));
        assert_eq!(payload["options"]["num_predict"], json!(64));
    }

    #[test]
    fn endpoint_builder_accepts_base_urls_and_full_endpoint() {
        assert_eq!(
            chat_endpoint("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/api/chat"
        );
        assert_eq!(
            chat_endpoint("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/api/chat"
        );
        assert_eq!(
            chat_endpoint("http://127.0.0.1:11434/api/chat"),
            "http://127.0.0.1:11434/api/chat"
        );
    }

    #[test]
    fn normalizes_response_text_tool_calls_usage_and_stop_reason() {
        let response = normalize_response(
            "ollama/llama3.2",
            OllamaChatResponse {
                model: Some("llama3.2".to_string()),
                created_at: Some("2026-05-02T12:00:00Z".to_string()),
                message: OllamaResponseMessage {
                    role: Some("assistant".to_string()),
                    content: Some("hello".to_string()),
                    tool_calls: vec![OllamaToolCall {
                        function: OllamaToolCallFunction {
                            name: "weather".to_string(),
                            arguments: json!("{\"city\":\"Paris\"}"),
                        },
                    }],
                },
                done: true,
                done_reason: Some("tool_calls".to_string()),
                prompt_eval_count: Some(11),
                eval_count: Some(7),
            },
        );

        assert_eq!(response.id, "ollama_2026-05-02T12:00:00Z");
        assert_eq!(response.model, "llama3.2");
        assert_eq!(response.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(response.usage.input_tokens, 11);
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(
            response.content,
            vec![
                OutputContentBlock::Text {
                    text: "hello".to_string()
                },
                OutputContentBlock::ToolUse {
                    id: "ollama_call_0".to_string(),
                    name: "weather".to_string(),
                    input: json!({"city": "Paris"}),
                },
            ]
        );
    }

    #[test]
    fn parses_split_ndjson_chunks() {
        let mut parser = NdjsonParser::new();
        let first = parser
            .push(br#"{"model":"llama3.2","message":{"content":"hel"},"done":false}"#)
            .expect("partial chunk should parse");
        assert!(first.is_empty());

        let parsed = parser
            .push(
                br#"
{"model":"llama3.2","message":{"content":"lo"},"done":true}
"#,
            )
            .expect("complete chunks should parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].message.content.as_deref(), Some("hel"));
        assert_eq!(parsed[1].message.content.as_deref(), Some("lo"));
        assert!(parser.finish().expect("empty trailing buffer").is_empty());
    }

    #[test]
    fn tool_argument_normalization_preserves_objects_and_wraps_invalid_strings() {
        assert_eq!(
            normalize_tool_arguments(&json!({"city": "Paris"})),
            json!({"city": "Paris"})
        );
        assert_eq!(
            normalize_tool_arguments(&json!("not-json")),
            json!({"raw": "not-json"})
        );
    }
}
