import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";


test.beforeEach(async ({ page }) => {
  await page.goto("/");
  if (page.url().includes("/login")) {
    await page.getByLabel("Username").fill(
      process.env.MEDIC_DASHBOARD_USERNAME ?? "admin",
    );
    await page.getByLabel("Password").fill(
      process.env.MEDIC_DASHBOARD_PASSWORD ?? "secret",
    );
    await page.getByRole("button", { name: /log in/i }).click();
  }
  await expect(page.locator("#medic-app")).toBeVisible();
});

test("dashboard has no critical accessibility or horizontal overflow issues", async ({
  page,
}) => {
  const accessibility = await new AxeBuilder({ page }).analyze();
  const blocking = accessibility.violations.filter(
    (violation) =>
      violation.impact === "critical" || violation.impact === "serious",
  );
  expect(blocking).toEqual([]);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

test("primary workflows are keyboard reachable", async ({ page }) => {
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toBeVisible();

  if ((page.viewportSize()?.width ?? 0) <= 767) {
    const menu = page.getByRole("button", { name: "Open navigation" });
    await expect(menu).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(
      page.getByRole("button", { name: "Workflow overview" }),
    ).toBeFocused();
  }

  for (let index = 0; index < 8; index += 1) {
    const focused = page.locator(":focus");
    const label = await focused.getAttribute("aria-label");
    const text = await focused.textContent();
    if (`${label ?? ""} ${text ?? ""}`.includes("Documents")) {
      await page.keyboard.press("Enter");
      break;
    }
    await page.keyboard.press("Tab");
  }

  await expect(page).toHaveURL(/\/documents$/);
});
