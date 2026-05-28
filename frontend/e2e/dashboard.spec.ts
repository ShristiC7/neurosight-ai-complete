import { test, expect, type Page } from "@playwright/test";

const BASE_URL = "http://localhost:3000";
const API_URL = "http://localhost:8000";

// ── Helpers ────────────────────────────────────────────────────────────────

async function registerAndLogin(page: Page): Promise<{ accessToken: string; user: Record<string, unknown> }> {
  const email = `e2e_${Date.now()}@neurosight.test`;
  const res = await page.request.post(`${API_URL}/api/v1/auth/register`, {
    data: { name: "E2E Tester", email, password: "SecurePass123!" },
  });
  const data = await res.json();

  await page.goto(BASE_URL);
  await page.evaluate((authData) => {
    localStorage.setItem("neurosight-auth", JSON.stringify({
      state: {
        user: authData.user,
        accessToken: authData.access_token,
        expiresAt: authData.expires_at,
      },
    }));
  }, data);

  return { accessToken: data.access_token, user: data.user };
}

// ── Dashboard Tests ────────────────────────────────────────────────────────

test.describe("Dashboard — Core Layout", () => {
  test.beforeEach(async ({ page }) => {
    await registerAndLogin(page);
    await page.goto(`${BASE_URL}/dashboard`);
    await page.waitForLoadState("networkidle");
  });

  test("page has correct title", async ({ page }) => {
    await expect(page).toHaveTitle(/NeuroSight/);
  });

  test("sidebar renders with logo", async ({ page }) => {
    await expect(page.getByText("NeuroSight")).toBeVisible();
  });

  test("sidebar navigation items visible", async ({ page }) => {
    await expect(page.getByText("Dashboard")).toBeVisible();
    await expect(page.getByText("Analytics")).toBeVisible();
    await expect(page.getByText("Settings")).toBeVisible();
  });

  test("topbar renders", async ({ page }) => {
    await expect(page.getByText("Cognitive Dashboard")).toBeVisible();
    await expect(page.getByText("Real-time multimodal monitoring")).toBeVisible();
  });

  test("cognitive score ring renders", async ({ page }) => {
    await expect(page.getByText("Cognitive Score")).toBeVisible();
  });

  test("start session button is visible", async ({ page }) => {
    await expect(page.getByText("▶ Start Session")).toBeVisible();
  });

  test("sensor labels are visible", async ({ page }) => {
    await expect(page.getByText("Eye Tracking")).toBeVisible();
    await expect(page.getByText("Voice Analysis")).toBeVisible();
    await expect(page.getByText("Behavioral")).toBeVisible();
  });

  test("AI Coach panel is visible", async ({ page }) => {
    await expect(page.getByText("AI Coach")).toBeVisible();
  });

  test("timeline chart heading visible", async ({ page }) => {
    await expect(page.getByText("Cognitive Timeline")).toBeVisible();
  });

  test("burnout risk panel visible", async ({ page }) => {
    await expect(page.getByText("Burnout Risk")).toBeVisible();
  });

  test("focus heatmap visible", async ({ page }) => {
    await expect(page.getByText("Weekly Focus Heatmap")).toBeVisible();
  });
});

test.describe("Dashboard — Metric Panels", () => {
  test.beforeEach(async ({ page }) => {
    await registerAndLogin(page);
    await page.goto(`${BASE_URL}/dashboard`);
    await page.waitForLoadState("networkidle");
  });

  test("eye fatigue panel renders", async ({ page }) => {
    await expect(page.getByText("Eye Fatigue")).toBeVisible();
  });

  test("voice stress panel renders", async ({ page }) => {
    await expect(page.getByText("Voice Stress")).toBeVisible();
  });

  test("productivity panel renders", async ({ page }) => {
    await expect(page.getByText("Productivity")).toBeVisible();
  });

  test("empty coach shows prompt message", async ({ page }) => {
    await expect(page.getByText(/AI Coach is monitoring/i)).toBeVisible();
  });
});

test.describe("Dashboard — Navigation", () => {
  test.beforeEach(async ({ page }) => {
    await registerAndLogin(page);
  });

  test("navigates to analytics page", async ({ page }) => {
    await page.goto(`${BASE_URL}/dashboard`);
    await page.getByText("Analytics").click();
    await expect(page).toHaveURL(/analytics/);
  });

  test("navigates to settings page", async ({ page }) => {
    await page.goto(`${BASE_URL}/dashboard`);
    await page.getByText("Settings").click();
    await expect(page).toHaveURL(/settings/);
  });

  test("unauthenticated user is redirected", async ({ page }) => {
    // Clear auth and attempt dashboard access
    await page.goto(BASE_URL);
    await page.evaluate(() => localStorage.clear());
    await page.goto(`${BASE_URL}/dashboard`);
    // Should redirect to login or home
    await expect(page).not.toHaveURL(`${BASE_URL}/dashboard`);
  });
});

test.describe("Dashboard — WebSocket Indicator", () => {
  test.beforeEach(async ({ page }) => {
    await registerAndLogin(page);
    await page.goto(`${BASE_URL}/dashboard`);
  });

  test("shows a connection status badge", async ({ page }) => {
    const statusBadge = page.locator("text=/LIVE|OFFLINE|RECONNECTING/");
    await expect(statusBadge.first()).toBeVisible({ timeout: 5000 });
  });
});
