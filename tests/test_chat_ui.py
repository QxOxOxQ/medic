from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
NODE = shutil.which("node")


def test_chat_ui_runs_submit_render_and_error_behaviors(tmp_path: Path) -> None:
    if NODE is None:
        pytest.skip("Node.js is required to execute chat.js behavior tests")

    script_path = tmp_path / "chat_ui_test.mjs"
    script_path.write_text(_node_test_script(), encoding="utf-8")

    result = subprocess.run(
        [NODE, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _node_test_script() -> str:
    return textwrap.dedent(
        r"""
        import assert from "node:assert/strict";
        import fs from "node:fs";
        import vm from "node:vm";

        class FakeElement {
          constructor({ className = "", value = "", textContent = "" } = {}) {
            this.children = [];
            this.className = className;
            this.disabled = false;
            this._innerHTML = "";
            this.parent = null;
            this.removed = false;
            this.scrollHeight = 0;
            this.scrolled = false;
            this.scrollTop = 0;
            this.tabIndex = null;
            this.textContent = textContent;
            this.value = value;
            this.listeners = {};
          }

          set innerHTML(value) {
            this._innerHTML = value;
            this.children = [];
          }

          get innerHTML() {
            return this._innerHTML;
          }

          appendChild(child) {
            child.parent = this;
            this.children.push(child);
            this.scrollHeight = this.children.length;
          }

          addEventListener(eventName, handler) {
            this.listeners[eventName] = handler;
          }

          dispatchEvent(eventName, event) {
            this.listeners[eventName](event);
          }

          focus() {
            this.focused = true;
          }

          querySelector(selector) {
            if (selector !== ".chat-empty") {
              return null;
            }
            return this.children.find(
              (child) => child.className === "chat-empty" && !child.removed,
            ) || null;
          }

          querySelectorAll() {
            return [];
          }

          scrollIntoView() {
            this.scrolled = true;
          }

          setAttribute(name, value) {
            this[name] = value;
          }

          remove() {
            this.removed = true;
            if (this.parent) {
              this.parent.children = this.parent.children.filter(
                (child) => child !== this,
              );
            }
          }
        }

        function createHarness({ question = "  Is LDL high?  ", apiJson }) {
          const chatHistory = new FakeElement();
          chatHistory.appendChild(new FakeElement({ className: "chat-empty" }));
          const elements = {
            chatHistory,
            chatQuestion: new FakeElement({ value: question }),
            chatSources: new FakeElement(),
            chatState: new FakeElement({ textContent: "ready" }),
            chatSubmit: new FakeElement(),
            conversationList: new FakeElement(),
            sourceDrawer: new FakeElement(),
            sourceDrawerTitle: new FakeElement(),
            sourceDrawerBody: new FakeElement(),
            sourceDrawerClose: new FakeElement(),
          };
          const context = {
            console,
            document: {
              createElement() {
                return new FakeElement();
              },
              querySelector() {
                return null;
              },
            },
            window: {
              MedicDashboard: {
                api: { json: apiJson },
                elements,
                processDetails: {
                  async load() {},
                  showTab() {},
                },
              },
            },
          };
          context.globalThis = context;
          return { context, elements };
        }

        function loadChat(context) {
          vm.createContext(context);
          vm.runInContext(
            fs.readFileSync("dashboard/static/i18n.js", "utf8"),
            context,
          );
          vm.runInContext(
            fs.readFileSync("dashboard/static/formatting.js", "utf8"),
            context,
          );
          vm.runInContext(
            fs.readFileSync("dashboard/static/chat.js", "utf8"),
            context,
          );
        }

        async function successfulSubmitRendersAnswerAndSources() {
          const calls = [];
          let pendingResolve;
          const pendingResponse = new Promise((resolve) => {
            pendingResolve = resolve;
          });
          const { context, elements } = createHarness({
            apiJson: async (path, options) => {
              calls.push({ path, options });
              return pendingResponse;
            },
          });
          loadChat(context);

          const askPromise = context.window.MedicDashboard.chat.ask();

          assert.equal(calls.length, 1);
          assert.equal(calls[0].path, "/api/chat/conversations");
          assert.deepEqual(JSON.parse(calls[0].options.body), {
            question: "Is LDL high?",
            limit: 5,
          });
          assert.equal(elements.chatQuestion.disabled, true);
          assert.equal(elements.chatSubmit.disabled, true);
          assert.equal(elements.chatState.textContent, "answering");
          assert.equal(elements.chatHistory.children.length, 1);
          assert.equal(elements.chatHistory.children[0].className, "chat-message user");
          assert.match(elements.chatHistory.children[0].innerHTML, /You/);
          assert.match(elements.chatHistory.children[0].innerHTML, /Is LDL high\?/);

          pendingResolve({
            conversation: conversationPayload({
              answer: "LDL is elevated [S1].",
              insufficientContext: false,
              sources: [sourcePayload()],
              traceEvents: [coordinatorTrace()],
            }),
          });
          await askPromise;

          assert.equal(calls.length, 2);
          assert.equal(calls[1].path, "/api/chat/conversations");
          assert.equal(elements.chatQuestion.disabled, false);
          assert.equal(elements.chatSubmit.disabled, false);
          assert.equal(elements.chatQuestion.value, "");
          assert.equal(elements.chatState.textContent, "answer ready");
          assert.equal(elements.chatHistory.children.length, 2);
          const answerHtml = elements.chatHistory.children[1].innerHTML;
          assert.match(answerHtml, /Agent/);
          assert.match(answerHtml, /cardiometabolic internist/);
          assert.match(answerHtml, /LDL is elevated/);
          assert.match(answerHtml, /data-source-id="S1"/);
          assert.match(answerHtml, /Answer trace/);
          assert.match(elements.chatSources.innerHTML, /Sources/);
          assert.match(elements.chatSources.innerHTML, /1 fragment/);
          assert.match(elements.chatSources.innerHTML, /S1/);
          assert.match(elements.chatSources.innerHTML, /report\.md/);
          assert.match(elements.chatSources.innerHTML, /score 0\.823/);
          assert.match(elements.chatSources.innerHTML, /1234567890abcdef\.\.\./);
          assert.match(elements.chatSources.innerHTML, /LDL cholesterol is elevated\./);

          const clicked = { prevented: false };
          elements.chatHistory.dispatchEvent("click", {
            preventDefault() {
              clicked.prevented = true;
            },
            target: {
              closest(selector) {
                assert.equal(selector, "[data-source-id]");
                return { dataset: { sourceId: "S1" }, textContent: "[S1]" };
              },
            },
          });
          assert.equal(clicked.prevented, true);
          assert.equal(elements.sourceDrawer.hidden, false);
          assert.equal(elements.sourceDrawerTitle.textContent, "S1");
          assert.equal(elements.sourceDrawer.scrolled, true);
          assert.equal(elements.sourceDrawer.focused, true);
          assert.match(elements.sourceDrawerBody.innerHTML, /LDL cholesterol is elevated\./);
          assert.equal(elements.chatState.textContent, "source S1");
        }

        async function insufficientContextRendersWarningAndNoSources() {
          const { context, elements } = createHarness({
            question: "Missing data?",
            apiJson: async () => ({
              conversation: conversationPayload({
                answer: "I could not find enough context.",
                insufficientContext: true,
                sources: [],
                traceEvents: [coordinatorTrace()],
              }),
            }),
          });
          loadChat(context);

          await context.window.MedicDashboard.chat.ask();

          assert.equal(elements.chatState.textContent, "missing context");
          assert.match(
            elements.chatHistory.children[1].innerHTML,
            /Not enough data in the documentation\./,
          );
          assert.match(elements.chatSources.innerHTML, /No sources to display\./);
        }

        async function rejectionRendersErrorAndReenablesControls() {
          const { context, elements } = createHarness({
            question: "Is there an error?",
            apiJson: async () => {
              throw new Error("Agent execution failed");
            },
          });
          loadChat(context);

          await context.window.MedicDashboard.chat.ask();

          assert.equal(elements.chatState.textContent, "error");
          assert.equal(elements.chatQuestion.disabled, false);
          assert.equal(elements.chatSubmit.disabled, false);
          assert.equal(elements.chatHistory.children[1].className, "chat-message assistant");
          assert.match(elements.chatHistory.children[1].innerHTML, /Agent execution failed/);
        }

        function conversationPayload({
          answer,
          insufficientContext,
          sources,
          traceEvents,
        }) {
          return {
            id: "conversation-1",
            title: "Is LDL high?",
            messages: [
              {
                id: "message-user",
                role: "user",
                content: "Is LDL high?",
                insufficient_context: false,
                sources: [],
                trace_events: [],
              },
              {
                id: "message-assistant",
                role: "assistant",
                content: answer,
                insufficient_context: insufficientContext,
                sources,
                trace_events: traceEvents,
              },
            ],
          };
        }

        function sourcePayload() {
          return {
            source_id: "S1",
            source: "report.md",
            content_hash: "1234567890abcdefXYZ",
            score: 0.82345,
            excerpt: "LDL cholesterol is elevated.",
            document_name: "report.md",
            relative_raw_path: "raw/report.pdf",
            retrieval_query: "LDL",
          };
        }

        function coordinatorTrace() {
          return {
            event_type: "coordinator",
            title: "Coordinator selected specialists",
            status: "succeeded",
            agent_name: "coordinator",
            payload: {
              selected_agents: ["cardiometabolic_internist"],
            },
          };
        }

        async function emptyQuestionDoesNotCallApi() {
          let calls = 0;
          const { context, elements } = createHarness({
            question: "   ",
            apiJson: async () => {
              calls += 1;
              return {};
            },
          });
          loadChat(context);

          await context.window.MedicDashboard.chat.ask();

          assert.equal(calls, 0);
          assert.equal(elements.chatState.textContent, "enter a question");
          assert.equal(elements.chatHistory.children.length, 1);
        }

        async function englishOnlyRendersDynamicLabels() {
          const { context, elements } = createHarness({
            question: "Is LDL high?",
            apiJson: async () => ({
              conversation: conversationPayload({
                answer: "LDL is elevated [S1].",
                insufficientContext: false,
                sources: [sourcePayload()],
                traceEvents: [coordinatorTrace()],
              }),
            }),
          });
          loadChat(context);

          await context.window.MedicDashboard.chat.ask();

          assert.equal(elements.chatState.textContent, "answer ready");
          assert.match(elements.chatHistory.children[0].innerHTML, /You/);
          assert.match(elements.chatHistory.children[1].innerHTML, /Answer trace/);
          assert.match(elements.chatSources.innerHTML, /Sources/);
          assert.match(elements.chatSources.innerHTML, /1 fragment/);
        }

        await successfulSubmitRendersAnswerAndSources();
        await insufficientContextRendersWarningAndNoSources();
        await rejectionRendersErrorAndReenablesControls();
        await emptyQuestionDoesNotCallApi();
        await englishOnlyRendersDynamicLabels();
        """
    )
