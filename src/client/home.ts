const searchInput = document.querySelector<HTMLInputElement>("[data-search-input]");
const cards = Array.from(document.querySelectorAll<HTMLElement>("[data-filter-card]"));
const emptyState = document.querySelector<HTMLElement>("[data-empty-state]");

function syncCollectionFilter(): void {
  const query = searchInput?.value.trim().toLowerCase() ?? "";
  let visibleCount = 0;

  for (const card of cards) {
    const haystack = card.dataset.search ?? "";
    const isVisible = query.length === 0 || haystack.includes(query);
    card.hidden = !isVisible;
    if (isVisible) {
      visibleCount += 1;
    }
  }

  if (emptyState) {
    emptyState.hidden = visibleCount !== 0;
  }
}

if (searchInput) {
  searchInput.addEventListener("input", syncCollectionFilter);
  syncCollectionFilter();
}
