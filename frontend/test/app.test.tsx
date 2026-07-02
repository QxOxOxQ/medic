import { cleanup, render, screen } from "@testing-library/preact";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { App } from "../app/App";


beforeEach(() => {
  window.history.replaceState({}, "", "/retrieval");
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.startsWith("/api/documents")) {
        return Response.json({
          ok: true,
          documents: [],
          page: 1,
          page_size: 25,
          total: 0,
          pages: 1,
          status_counts: {},
          qdrant_error: null,
        });
      }
      if (path === "/api/workspace/overview") {
        return Response.json({
          ok: true,
          status: {
            raw_pdf_count: 0,
            parsed_markdown_count: 0,
            document_count: 0,
            last_processed_at: null,
            qdrant: {
              available: true,
              collection_name: "medic",
              collection_exists: true,
              points_count: 0,
              error: null,
            },
          },
          postgres: { available: true, error: null },
          latest_pipeline_run: null,
          latest_conversation: null,
        });
      }
      if (path === "/api/status") {
        return Response.json({
          raw_pdf_count: 0,
          parsed_markdown_count: 0,
          document_count: 0,
          last_processed_at: null,
          qdrant: {
            available: true,
            collection_name: "medic",
            collection_exists: true,
            points_count: 0,
            error: null,
          },
        });
      }
      if (path === "/api/admin/llm-providers") {
        return Response.json({
          ok: true,
          generated_at: "2026-07-02T00:00:00Z",
          configuration: {
            chat_provider: "openrouter",
            chat_model: "openai/gpt-4.1-mini",
            embedding_provider: "openrouter",
            embedding_model: "openai/text-embedding-3-small",
            agent_models: [
              { agent_name: "professor", model_id: "openai/gpt-4.1-mini" },
            ],
            selectable_models: [],
          },
          providers: [
            {
              provider_key: "openrouter",
              provider_name: "OpenRouter",
              status: "available",
              message: null,
              issues: [],
              credits: {
                total_credits: { amount: "100.50", currency: "USD" },
                total_usage: { amount: "25.75", currency: "USD" },
                remaining_credits: { amount: "74.75", currency: "USD" },
              },
              api_key: {
                label: "sk-or-v1-test",
                usage: { amount: "25.50", currency: "USD" },
                usage_daily: { amount: "1.50", currency: "USD" },
                usage_weekly: { amount: "7.50", currency: "USD" },
                usage_monthly: { amount: "12.50", currency: "USD" },
                byok_usage: { amount: "0", currency: "USD" },
                byok_usage_daily: { amount: "0", currency: "USD" },
                byok_usage_weekly: { amount: "0", currency: "USD" },
                byok_usage_monthly: { amount: "0", currency: "USD" },
                include_byok_in_limit: false,
                is_free_tier: false,
                is_management_key: false,
                is_provisioning_key: false,
                limit: { amount: "100", currency: "USD" },
                limit_remaining: { amount: "74.50", currency: "USD" },
                limit_reset: "monthly",
                expires_at: "2027-12-31T23:59:59Z",
              },
              activity: {
                window_label: "Last 30 completed UTC days",
                completed_utc_days: 30,
                totals: {
                  usage: { amount: "0.015", currency: "USD" },
                  byok_usage: { amount: "0", currency: "USD" },
                  requests: 5,
                  prompt_tokens: 50,
                  completion_tokens: 125,
                  reasoning_tokens: 0,
                },
                top_models: [
                  {
                    model: "openai/gpt-4.1-mini",
                    provider_name: "OpenAI",
                    totals: {
                      usage: { amount: "0.015", currency: "USD" },
                      byok_usage: { amount: "0", currency: "USD" },
                      requests: 5,
                      prompt_tokens: 50,
                      completion_tokens: 125,
                      reasoning_tokens: 0,
                    },
                    last_activity_date: "2026-07-01",
                  },
                ],
                top_providers: [],
              },
            },
          ],
        });
      }
      return Response.json({});
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("app shell exposes primary workflows and admin navigation", async () => {
  const user = userEvent.setup();
  render(<App username="operator" isAdmin />);

  expect(
    screen.getByRole("heading", { level: 2, name: "Retrieval inspector" }),
  ).toBeVisible();
  expect(screen.getByRole("link", { name: /Admin/ })).toHaveAttribute(
    "href",
    "/admin",
  );

  await user.click(screen.getByRole("button", { name: /Documents/ }));

  expect(
    await screen.findByRole("heading", { name: "Document workspace" }),
  ).toBeVisible();
  expect(window.location.pathname).toBe("/documents");
});

test("app shell exposes provider stats to admins", async () => {
  window.history.replaceState({}, "", "/llm-providers");
  render(<App username="operator" isAdmin />);

  expect(
    await screen.findByRole("heading", { level: 2, name: "LLM providers" }),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: /LLM providers/ })).toBeVisible();
  expect(screen.getByText("$74.75")).toBeVisible();
  expect(screen.getAllByText("openai/gpt-4.1-mini")[0]).toBeVisible();
});

test("app shell hides admin provider stats from non-admin users", () => {
  window.history.replaceState({}, "", "/overview");
  render(<App username="operator" isAdmin={false} />);

  expect(screen.queryByRole("button", { name: /LLM providers/ })).toBeNull();
  expect(screen.queryByRole("link", { name: /Admin/ })).toBeNull();
});
