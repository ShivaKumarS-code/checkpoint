import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.CHECKPOINT_E2E_PORT ?? 4173);
const external = process.env.CHECKPOINT_E2E_EXTERNAL === "1";
const baseURL = process.env.CHECKPOINT_E2E_URL ?? `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 45_000,
  expect: { timeout: 10_000 },
  reporter: [["list"]],
  use: {
    baseURL,
    colorScheme: "dark",
    trace: "retain-on-failure",
  },
  webServer: external
    ? undefined
    : {
        command: "uv run --project ../backend python e2e/serve.py",
        url: `http://127.0.0.1:${port}/api/health`,
        timeout: 120_000,
        reuseExistingServer: false,
        env: { CHECKPOINT_E2E_PORT: String(port) },
        stdout: "pipe",
        stderr: "pipe",
      },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
