/* ==========================================================================
 * Neural Atlas AI — Home / Collection Page Client Runtime
 * ==========================================================================
 *
 * Responsibilities:
 *   1. Fuzzy-free collection search filter with debounce
 *   2. Scroll-based reveal animations (IntersectionObserver)
 *   3. Dark mode toggle (localStorage-persisted)
 *   4. Sticky header scroll detection
 *   5. Archive entry hover micro-interactions
 * ========================================================================== */

/* ── Element References ──────────────────────────────────────────────────── */

const searchInput = document.querySelector<HTMLInputElement>("[data-search-input]");
const cards = Array.from(document.querySelectorAll<HTMLElement>("[data-filter-card]"));
const emptyState = document.querySelector<HTMLElement>("[data-empty-state]");
const siteHeader = document.getElementById("site-header") as HTMLElement | null;
const themeToggle = document.getElementById("theme-toggle") as HTMLButtonElement | null;

/* ── Search Filter ───────────────────────────────────────────────────────── */

let searchTimer: ReturnType<typeof setTimeout> | null = null;

function syncCollectionFilter(): void {
  const query = searchInput?.value.trim().toLowerCase() ?? "";
  let visibleCount = 0;

  for (const card of cards) {
    const haystack = card.dataset.search ?? "";
    const isVisible = query.length === 0 || haystack.includes(query);
    card.hidden = !isVisible;

    /* Re-trigger reveal animation on filter */
    if (isVisible) {
      visibleCount += 1;
      card.classList.remove("revealed");
      void card.offsetWidth; /* Force reflow */
      card.classList.add("revealed");
    }
  }

  if (emptyState) {
    emptyState.hidden = visibleCount !== 0;
  }
}

function debouncedFilter(): void {
  if (searchTimer) {
    clearTimeout(searchTimer);
  }

  searchTimer = setTimeout(syncCollectionFilter, 80);
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

/* ── Scroll Reveal Animations ────────────────────────────────────────────── */

function installScrollReveals(): void {
  /* Stagger archive entries with progressive enhancement */
  const entries = document.querySelectorAll<HTMLElement>(".archive-entry");

  if (entries.length === 0) {
    return;
  }

  /* Gate: add class to body to activate hidden-by-default CSS */
  document.body.classList.add("js-reveal-ready");

  entries.forEach((entry, index) => {
    entry.dataset.delay = String(Math.min(index + 1, 6));
  });

  const entryObserver = new IntersectionObserver(
    (observerEntries) => {
      for (const entry of observerEntries) {
        if (entry.isIntersecting) {
          (entry.target as HTMLElement).classList.add("revealed");
          entryObserver.unobserve(entry.target);
        }
      }
    },
    {
      threshold: 0.05,
      rootMargin: "0px 0px -20px 0px"
    }
  );

  entries.forEach((entry) => entryObserver.observe(entry));
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

if (searchInput) {
  searchInput.addEventListener("input", debouncedFilter);
  syncCollectionFilter();
}

installScrollReveals();
installThemeToggle();

/* Sticky header */
window.addEventListener("scroll", () => {
  window.requestAnimationFrame(syncHeaderScroll);
}, { passive: true });

syncHeaderScroll();
