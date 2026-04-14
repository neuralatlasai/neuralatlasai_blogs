/* ==========================================================================
 * Neural Atlas AI — Article Page Client Runtime
 * ==========================================================================
 *
 * Responsibilities:
 *   1. Reading progress bar (scroll-driven, rAF-debounced)
 *   2. Current-section HUD (IntersectionObserver-free scroll detection)
 *   3. Copy-to-clipboard for code blocks
 *   4. Scroll-based reveal animations (IntersectionObserver)
 *   5. Dark mode toggle (localStorage-persisted)
 *   6. Sticky header scroll detection
 * ========================================================================== */

type SectionHeading = {
  element: HTMLElement;
  label: string;
};

/* ── Element References ──────────────────────────────────────────────────── */

const copyButtons = Array.from(document.querySelectorAll<HTMLButtonElement>(".code-frame__copy"));
const currentSection = document.querySelector<HTMLElement>("[data-current-section]");
const currentSectionTitle = document.querySelector<HTMLElement>("[data-current-section-title]");
const articleContent = document.querySelector<HTMLElement>(".article-content");
const progressBar = document.getElementById("reading-progress-bar") as HTMLElement | null;
const siteHeader = document.getElementById("site-header") as HTMLElement | null;
const themeToggle = document.getElementById("theme-toggle") as HTMLButtonElement | null;

const sectionHeadings: SectionHeading[] = Array.from(
  document.querySelectorAll<HTMLElement>(".article-content .article-heading[id]")
).map((heading) => {
  const clone = heading.cloneNode(true) as HTMLElement;
  clone.querySelector(".heading-anchor")?.remove();

  return {
    element: heading,
    label: clone.textContent?.trim() ?? ""
  };
});

let framePending = false;

/* ── Reading Progress ────────────────────────────────────────────────────── */

function syncProgress(): void {
  if (!progressBar || !articleContent) {
    return;
  }

  const rect = articleContent.getBoundingClientRect();
  const viewportHeight = window.innerHeight;
  const totalScrollable = rect.height - viewportHeight;

  if (totalScrollable <= 0) {
    progressBar.style.width = "100%";
    return;
  }

  const scrolled = Math.max(0, -rect.top);
  const progress = Math.min(1, scrolled / totalScrollable);
  progressBar.style.width = `${(progress * 100).toFixed(1)}%`;
}

/* ── Current Section HUD ─────────────────────────────────────────────────── */

function syncCurrentSection(): void {
  if (!currentSection || !currentSectionTitle || !articleContent || sectionHeadings.length === 0) {
    return;
  }

  const isVisible = articleContent.getBoundingClientRect().top <= 24;
  currentSection.dataset.visible = String(isVisible);
  currentSection.setAttribute("aria-hidden", String(!isVisible));

  if (!isVisible) {
    return;
  }

  const activationOffset = 92;
  let activeHeading = sectionHeadings[0];

  for (const heading of sectionHeadings) {
    if (heading.element.getBoundingClientRect().top > activationOffset) {
      break;
    }

    activeHeading = heading;
  }

  if (currentSectionTitle.textContent !== activeHeading.label) {
    currentSectionTitle.textContent = activeHeading.label;
  }
}

/* ── Sticky Header Detection ─────────────────────────────────────────────── */

function syncHeaderScroll(): void {
  if (!siteHeader) {
    return;
  }

  const scrolled = window.scrollY > 40;
  if (siteHeader.dataset.scrolled !== String(scrolled)) {
    siteHeader.dataset.scrolled = String(scrolled);
  }
}

/* ── Unified Scroll Sync (rAF debounced) ─────────────────────────────────── */

function scheduleSync(): void {
  if (framePending) {
    return;
  }

  framePending = true;
  window.requestAnimationFrame(() => {
    framePending = false;
    syncProgress();
    syncCurrentSection();
    syncHeaderScroll();
  });
}

/* ── Code Copy Buttons ───────────────────────────────────────────────────── */

function installCopyButtons(): void {
  for (const button of copyButtons) {
    button.addEventListener("click", async () => {
      const figure = button.closest(".code-frame");
      const code = figure?.querySelector("pre code");
      if (!code?.textContent) {
        return;
      }

      await navigator.clipboard.writeText(code.textContent);
      const original = button.textContent;
      button.textContent = "Copied ✓";
      button.style.color = "var(--page-accent)";
      window.setTimeout(() => {
        button.textContent = original;
        button.style.color = "";
      }, 1600);
    });
  }
}

/* ── Scroll Reveal Animations (IntersectionObserver) ─────────────────────── */

function installScrollReveals(): void {
  /* Article page: no staggered reveals needed since the paper is always
   * above-the-fold. This is intentionally minimal for article pages. */
}

/* ── Dark Mode Toggle ────────────────────────────────────────────────────── */

function installThemeToggle(): void {
  if (!themeToggle) {
    return;
  }

  themeToggle.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";

    /* Apply transient transition class to documentElement to smooth all colors */
    document.documentElement.classList.add("theme-transition");

    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);

    /* Update theme-color meta tags */
    const metaTags = document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]');
    metaTags.forEach((tag) => {
      if (newTheme === "dark") {
        tag.setAttribute("content", "#0e0d0b");
      } else {
        tag.setAttribute("content", "#f4eed6");
      }
    });

    /* Remove transition lock after animation completes (120ms token + 30ms buffer) */
    window.setTimeout(() => {
      document.documentElement.classList.remove("theme-transition");
    }, 150);
  });
}

/* ── Bootstrap ───────────────────────────────────────────────────────────── */

scheduleSync();
installCopyButtons();
installScrollReveals();
installThemeToggle();

window.addEventListener("scroll", scheduleSync, { passive: true });
window.addEventListener("resize", scheduleSync);
