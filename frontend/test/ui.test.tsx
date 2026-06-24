import { render, screen } from "@testing-library/preact";
import userEvent from "@testing-library/user-event";
import type { JSX } from "preact";
import { useState } from "preact/hooks";
import { expect, test, vi } from "vitest";
import { Dialog, StatusBadge, Tabs } from "../shared/ui";


function TabFixture(): JSX.Element {
  const [tab, setTab] = useState<"overview" | "chunks">("overview");
  return (
    <Tabs
      value={tab}
      onChange={setTab}
      tabs={[
        { value: "overview", label: "Overview" },
        { value: "chunks", label: "Chunks" },
      ]}
    />
  );
}

test("tabs support arrow-key navigation", async () => {
  const user = userEvent.setup();
  render(<TabFixture />);

  const overview = screen.getByRole("tab", { name: "Overview" });
  overview.focus();
  await user.keyboard("{ArrowRight}");

  expect(screen.getByRole("tab", { name: "Chunks" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

test("dialog closes with Escape and status badges expose text", async () => {
  const user = userEvent.setup();
  const close = vi.fn();
  render(
    <>
      <Dialog title="Delete documents" close={close}>
        Confirm deletion
      </Dialog>
      <StatusBadge status="interrupted" />
    </>,
  );

  expect(screen.getByRole("dialog", { name: "Delete documents" })).toBeVisible();
  expect(screen.getByText("interrupted")).toBeVisible();
  await user.keyboard("{Escape}");
  expect(close).toHaveBeenCalledOnce();
});
