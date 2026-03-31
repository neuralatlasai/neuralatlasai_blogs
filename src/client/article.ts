type SectionHeading = {
  element: HTMLElement;
  label: string;
};

const copyButtons = Array.from(document.querySelectorAll<HTMLButtonElement>(".code-frame__copy"));
const currentSection = document.querySelector<HTMLElement>("[data-current-section]");
const currentSectionTitle = document.querySelector<HTMLElement>("[data-current-section-title]");
const articleContent = document.querySelector<HTMLElement>(".article-content");
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

function syncProgress(): void {
  return;
}

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

function scheduleSync(): void {
  if (framePending) {
    return;
  }

  framePending = true;
  window.requestAnimationFrame(() => {
    framePending = false;
    syncProgress();
    syncCurrentSection();
  });
}

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
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = original;
      }, 1400);
    });
  }
}

scheduleSync();
installCopyButtons();
window.addEventListener("scroll", scheduleSync, { passive: true });
window.addEventListener("resize", scheduleSync);
