import { render, screen } from "@testing-library/preact";
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
      return Response.json({});
    }),
  );
});

afterEach(() => {
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
