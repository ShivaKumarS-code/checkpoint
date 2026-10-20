import { expect, test } from "@playwright/test";

test("dark UI renders runs, expands payloads, and shows the experiment DAG", async ({
  page,
}) => {
  const dialogs: string[] = [];
  page.on("dialog", (dialog) => {
    dialogs.push(dialog.message());
    void dialog.dismiss();
  });

  await page.goto("/");
  await expect(page.getByTestId("app-shell")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByTestId("runs-page")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Runs" })).toBeVisible();

  const background = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(background).not.toBe("rgb(255, 255, 255)");

  await page.locator('[data-run-name="refund-agent"] a').first().click();
  await expect(page.getByTestId("run-page")).toBeVisible();
  await expect(page.getByTestId("run-name")).toHaveText("refund-agent");

  await page.getByTestId("step-toggle-0").click();
  const payload = page.getByTestId("step-payload-0");
  await expect(payload).toBeVisible();
  await expect(payload).toContainText("<script>window.__xss=1</script>");
  await expect(payload).toContainText("refund planner");

  expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();
  expect(dialogs).toEqual([]);
  await expect(page.locator("script", { hasText: "window.__xss" })).toHaveCount(0);

  await page.getByRole("link", { name: "Runs" }).first().click();
  await expect(page.getByTestId("experiment-card")).toHaveCount(1);
  await page.getByTestId("experiment-card").click();

  await expect(page.getByTestId("experiment-page")).toBeVisible();
  await expect(page.getByTestId("dag")).toBeVisible();
  await expect(page.getByTestId("dag-node-root")).toBeVisible();
  await expect(page.getByTestId("dag-node-fork")).toContainText("fork at step 2");
  await expect(page.getByTestId("dag-node-baseline")).toBeVisible();
  await expect(page.getByTestId("dag-node-guard_prior_refund")).toBeVisible();
  await expect(page.getByTestId("dag-node-guard_prior_refund")).toHaveAttribute(
    "data-variant-status",
    "COMPLETED",
  );

  await expect(
    page.getByTestId("variant-row-baseline").getByText("refund issued twice"),
  ).toBeVisible();
  await page.getByTestId("compare-guard_prior_refund").click();
  await expect(page.getByTestId("diff-page")).toBeVisible();
  await expect(page.getByTestId("diff-title")).toHaveText("First divergence at step 2");
  await expect(page.getByTestId("diff-left")).toBeVisible();
  await expect(page.getByTestId("diff-right")).toBeVisible();
  await expect(page.getByTestId("diff-left-input")).toContainText("refund decision engine");
  await expect(page.getByTestId("diff-right-input")).toContainText("support reply writer");
});

test("live websocket events add runs and steps without a reload", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await expect(page.getByTestId("runs-page")).toBeVisible();
  await expect(page.getByTestId("runs-page").getByText("live", { exact: true })).toBeVisible();

  const created = await request.post("/api/sdk/run/start", {
    data: { name: "live-e2e", mode: "RECORD" },
  });
  expect(created.ok()).toBeTruthy();
  const runId = (await created.json()).run_id as string;
  await expect(page.locator(`[data-run-id="${runId}"]`)).toBeVisible();

  await page.locator(`[data-run-id="${runId}"] a`).first().click();
  await expect(page.getByTestId("run-page")).toBeVisible();
  await expect(page.getByTestId("run-page").getByText("live", { exact: true })).toBeVisible();

  const step = await request.post("/api/sdk/step", {
    data: {
      run_id: runId,
      name: "live_tool",
      step_type: "TOOL_CALL",
      input: { query: "live" },
      output: { ok: true },
      source: "RECORDED",
    },
  });
  expect(step.ok()).toBeTruthy();
  await expect(page.getByTestId("step-row-0")).toBeVisible();
  await page.getByTestId("step-toggle-0").click();
  await expect(page.getByTestId("step-payload-0")).toContainText('"query": "live"');
});
