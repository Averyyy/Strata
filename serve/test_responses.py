"""Responses HTTP and history replay against the existing mock engine."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate, responses_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, responses_events, serve

ROOT = Path(__file__).resolve().parents[1]


class Responses(unittest.TestCase):
    def service(self, answer):
        tok = ByteTokenizer()
        return Service(MockEngine(tok, answer, max_context=50000), tok,
                       ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def events(self, answer, **fields):
        svc = self.service(answer)
        req = {"input": "hello", "reasoning": {"effort": "none"}, **fields}
        messages, tools, kw = responses_to_messages(req)
        ids, thinking, limit = svc.prepare(messages, tools, kw, 500)
        return [e[1] for e in responses_events(svc, req, ids, thinking, tools, limit, threading.Event()) if e]

    def test_text_reasoning_and_lifecycle(self):
        events = self.events("Thought.</think>\n\nHello.", reasoning={"effort": "low"})
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        final = events[-1]["response"]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(final["output"][0]["summary"][0]["text"], "Thought.")
        self.assertEqual(final["output"][1]["content"][0]["text"], "Hello.")
        self.assertEqual(final["usage"]["total_tokens"], final["usage"]["input_tokens"] + final["usage"]["output_tokens"])

    def test_function_stream_arguments_and_replay(self):
        tools = [{"type": "function", "name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
        events = self.events("<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n"
                             "</function>\n</tool_call>", tools=tools)
        final = events[-1]["response"]["output"][0]
        self.assertEqual(final["type"], "function_call")
        self.assertEqual(json.loads(final["arguments"]), {"path": "a.txt"})
        deltas = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
        self.assertEqual(deltas, final["arguments"])
        messages, _, _ = responses_to_messages({"tools": tools, "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "read a.txt"}]}, final,
            {"type": "function_call_output", "call_id": final["call_id"], "output": "file contents"}]})
        self.assertEqual(messages[-2]["tool_calls"][0]["function"]["arguments"], {"path": "a.txt"})
        self.assertEqual(messages[-1], {"role": "tool", "content": "file contents"})

    def test_custom_tool_preserves_freeform_and_replay(self):
        patch = "*** Begin Patch\n*** Add File: hi.txt\n+hello\n*** End Patch"
        tools = [{"type": "custom", "name": "apply_patch", "description": "Apply patch"}]
        events = self.events("<tool_call>\n<function=apply_patch>\n<parameter=input>\n" + patch +
                             "\n</parameter>\n</function>\n</tool_call>", tools=tools)
        final = events[-1]["response"]["output"][0]
        self.assertEqual(final["type"], "custom_tool_call")
        self.assertEqual(final["input"], patch)
        messages, _, _ = responses_to_messages({"tools": tools, "input": [
            {"role": "user", "content": "write"}, final,
            {"type": "custom_tool_call_output", "call_id": final["call_id"],
             "output": [{"type": "input_text", "text": "Success"}]}]})
        self.assertEqual(messages[-2]["tool_calls"][0]["function"]["arguments"], {"input": patch})
        self.assertEqual(messages[-1]["content"], "Success")

    def test_message_output_text_replay(self):
        messages, _, _ = responses_to_messages({"input": [
            {"role": "user", "content": "one"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "two"}]},
            {"role": "user", "content": "three"}]})
        self.assertEqual([m["content"] for m in messages], ["one", "two", "three"])

    def test_namespace_function_call_and_replay(self):
        tools = [{"type": "namespace", "name": "multi_agent_v1", "tools": [
            {"type": "function", "name": "spawn_agent", "parameters": {
                "type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}]}]
        events = self.events("<tool_call>\n<function=multi_agent_v1.spawn_agent>\n<parameter=message>\nhi\n"
                             "</parameter>\n</function>\n</tool_call>", tools=tools)
        final = events[-1]["response"]["output"][0]
        self.assertEqual((final["namespace"], final["name"]), ("multi_agent_v1", "spawn_agent"))
        messages, normalized, _ = responses_to_messages({"tools": tools, "input": [
            {"role": "user", "content": "delegate"}, final,
            {"type": "function_call_output", "call_id": final["call_id"], "output": "agent started"}]})
        self.assertEqual(normalized[0]["name"], "multi_agent_v1.spawn_agent")
        self.assertEqual(messages[-2]["tool_calls"][0]["function"]["name"], "multi_agent_v1.spawn_agent")

    def test_http_stream_auth_monitor_and_output_limit(self):
        svc = self.service("Hello.")
        svc.api_monitor = True
        svc.api_key = "test-key"
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        def post(body, key="test-key"):
            return urllib.request.urlopen(urllib.request.Request(base + "/v1/responses", json.dumps(body).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + key}), timeout=10)
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                post({"input": "hi"}, "wrong")
            self.assertEqual(caught.exception.code, 401)
            with post({"input": "hi", "reasoning": {"effort": "none"}, "max_output_tokens": 3}) as r:
                final = json.load(r)
            self.assertEqual(final["status"], "incomplete")
            self.assertEqual(final["incomplete_details"], {"reason": "max_output_tokens"})
            self.assertEqual(final["output"][0]["content"][0]["text"], "Hel")
            with post({"input": "hi", "reasoning": {"effort": "none"}, "stream": True}) as r:
                raw = r.read().decode()
            self.assertIn("event: response.completed", raw)
            self.assertIn("event: response.output_text.delta", raw)
            self.assertNotIn("[DONE]", raw)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_provider_bound_history_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "originating provider"):
            responses_to_messages({"input": [{"type": "reasoning", "encrypted_content": "opaque"}]})
        with self.assertRaisesRegex(ValueError, "full input history"):
            responses_to_messages({"input": "hi", "previous_response_id": "resp_other"})


if __name__ == "__main__":
    unittest.main()
