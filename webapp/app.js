/* ==========================================================================
   Hackneyed Recommender — Frontend Application
   ========================================================================== */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  movies: [],           // Array of { id, title, genres[] }
  ratings: {},          // { movieId: 1-5 }  — the user's submitted ratings
  displayedCount: 0,    // How many movies are currently rendered (pagination)
  filteredMovies: [],   // Currently visible after search + genre filter
  activeGenre: null,    // Currently selected genre filter (null = all)
};

const PAGE_SIZE = 40;   // Movies loaded per "page" in the grid

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const views         = { rating: $("#rating-view"), loading: $("#loading-view"), results: $("#results-view") };
const searchInput   = $("#search-input");
const genreFilters  = $("#genre-filters");
const movieGrid     = $("#movie-grid");
const loadMoreBtn   = $("#load-more-btn");
const ratedCount    = $("#rated-count");
const algorithmSel  = $("#algorithm-select");
const submitBtn     = $("#submit-btn");
const loadingDetail = $("#loading-detail");
const loadingAlgo   = $("#loading-algorithm");
const resultsList   = $("#results-list");
const resultsAlgo   = $("#results-algorithm");
const backBtn       = $("#back-btn");

// ---------------------------------------------------------------------------
// View management
// ---------------------------------------------------------------------------
function showView(name) {
  Object.values(views).forEach((v) => v.classList.remove("active"));
  views[name].classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Movie data loading
// ---------------------------------------------------------------------------

async function loadMovies() {
  try {
    // Fetch from the API server
    const res = await fetch("/api/movies");
    if (res.ok) {
      state.movies = await res.json();
      return;
    }
  } catch { /* API not available — fall through */ }

  try {
    // Fallback: load the CSV directly (works when served by a plain HTTP server)
    const res = await fetch("../data/ml-latest-small/movies.csv");
    const text = await res.text();
    state.movies = parseMoviesCsv(text);
  } catch {
    // Last-resort: use an embedded sample so the UI is still demoable
    state.movies = getSampleMovies();
  }
}

/** Parse the MovieLens movies.csv format (handles quoted titles with commas). */
function parseMoviesCsv(csv) {
  const lines = csv.trim().split("\n").slice(1); // drop header
  return lines.map((line) => {
    // CSV fields: movieId,title,genres  — title may be quoted
    const match = line.match(/^(\d+),(\"[^\"]*\"|[^,]*),(.*)$/);
    if (!match) return null;
    const id = parseInt(match[1], 10);
    const title = match[2].replace(/^"|"$/g, "");
    const genres = match[3].split("|").filter(Boolean);
    return { id, title, genres };
  }).filter(Boolean);
}

/** Tiny fallback dataset so the UI works even without any server. */
function getSampleMovies() {
  return [
    { id: 1,   title: "Toy Story (1995)",                     genres: ["Adventure","Animation","Children","Comedy","Fantasy"] },
    { id: 2,   title: "Jumanji (1995)",                       genres: ["Adventure","Children","Fantasy"] },
    { id: 6,   title: "Heat (1995)",                          genres: ["Action","Crime","Thriller"] },
    { id: 10,  title: "GoldenEye (1995)",                     genres: ["Action","Adventure","Thriller"] },
    { id: 16,  title: "Casino (1995)",                        genres: ["Crime","Drama"] },
    { id: 32,  title: "Twelve Monkeys (1995)",                genres: ["Mystery","Sci-Fi","Thriller"] },
    { id: 47,  title: "Seven (a.k.a. Se7en) (1995)",         genres: ["Mystery","Thriller"] },
    { id: 50,  title: "Usual Suspects, The (1995)",           genres: ["Crime","Mystery","Thriller"] },
    { id: 110, title: "Braveheart (1995)",                    genres: ["Action","Drama","War"] },
    { id: 150, title: "Apollo 13 (1995)",                     genres: ["Adventure","Drama","IMAX"] },
    { id: 260, title: "Star Wars: Episode IV (1977)",         genres: ["Action","Adventure","Drama","Sci-Fi"] },
    { id: 296, title: "Pulp Fiction (1994)",                  genres: ["Comedy","Crime","Drama","Thriller"] },
    { id: 318, title: "Shawshank Redemption, The (1994)",     genres: ["Crime","Drama"] },
    { id: 356, title: "Forrest Gump (1994)",                  genres: ["Comedy","Drama","Romance","War"] },
    { id: 480, title: "Jurassic Park (1993)",                 genres: ["Action","Adventure","Sci-Fi","Thriller"] },
    { id: 527, title: "Schindler's List (1993)",              genres: ["Drama","War"] },
    { id: 589, title: "Terminator 2: Judgment Day (1991)",    genres: ["Action","Sci-Fi"] },
    { id: 593, title: "Silence of the Lambs, The (1991)",     genres: ["Crime","Horror","Thriller"] },
    { id: 858, title: "Godfather, The (1972)",                genres: ["Crime","Drama"] },
    { id: 1196,title: "Star Wars: Episode V (1980)",          genres: ["Action","Adventure","Drama","Sci-Fi"] },
    { id: 1210,title: "Star Wars: Episode VI (1983)",         genres: ["Action","Adventure","Fantasy","Sci-Fi"] },
    { id: 1270,title: "Back to the Future (1985)",            genres: ["Adventure","Comedy","Sci-Fi"] },
    { id: 2571,title: "Matrix, The (1999)",                   genres: ["Action","Sci-Fi","Thriller"] },
    { id: 4993,title: "Lord of the Rings: Fellowship (2001)", genres: ["Adventure","Fantasy"] },
    { id: 5952,title: "Lord of the Rings: Two Towers (2002)", genres: ["Adventure","Fantasy"] },
    { id: 7153,title: "Lord of the Rings: Return (2003)",     genres: ["Action","Adventure","Drama","Fantasy"] },
    { id: 58559, title: "Dark Knight, The (2008)",            genres: ["Action","Crime","Drama","IMAX"] },
    { id: 79132, title: "Inception (2010)",                   genres: ["Action","Crime","Drama","Mystery","Sci-Fi","Thriller","IMAX"] },
    { id: 91529, title: "Dark Knight Rises, The (2012)",      genres: ["Action","Crime","IMAX","Thriller"] },
    { id: 122886,title: "Star Wars: Episode VII (2015)",      genres: ["Action","Adventure","Fantasy","Sci-Fi","IMAX"] },
  ];
}

// ---------------------------------------------------------------------------
// Genre filter pills
// ---------------------------------------------------------------------------
function buildGenreFilters() {
  const genreSet = new Set();
  state.movies.forEach((m) => m.genres.forEach((g) => genreSet.add(g)));
  const genres = [...genreSet].sort();

  // "All" pill
  const allPill = document.createElement("span");
  allPill.className = "genre-pill active";
  allPill.textContent = "All";
  allPill.dataset.genre = "";
  genreFilters.appendChild(allPill);

  genres.forEach((g) => {
    if (g === "(no genres listed)") return;
    const pill = document.createElement("span");
    pill.className = "genre-pill";
    pill.textContent = g;
    pill.dataset.genre = g;
    genreFilters.appendChild(pill);
  });

  genreFilters.addEventListener("click", (e) => {
    const pill = e.target.closest(".genre-pill");
    if (!pill) return;
    $$(".genre-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    state.activeGenre = pill.dataset.genre || null;
    applyFilters();
  });
}

// ---------------------------------------------------------------------------
// Search + filter logic
// ---------------------------------------------------------------------------
function applyFilters() {
  const query = searchInput.value.trim().toLowerCase();
  state.filteredMovies = state.movies.filter((m) => {
    if (state.activeGenre && !m.genres.includes(state.activeGenre)) return false;
    if (query && !m.title.toLowerCase().includes(query) &&
        !m.genres.some((g) => g.toLowerCase().includes(query))) return false;
    return true;
  });
  state.displayedCount = 0;
  movieGrid.innerHTML = "";
  renderNextPage();
}

let searchDebounce = null;
searchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(applyFilters, 200);
});

// ---------------------------------------------------------------------------
// Movie card rendering (paginated)
// ---------------------------------------------------------------------------
function renderNextPage() {
  const slice = state.filteredMovies.slice(
    state.displayedCount,
    state.displayedCount + PAGE_SIZE
  );
  slice.forEach((movie) => movieGrid.appendChild(createMovieCard(movie)));
  state.displayedCount += slice.length;

  // Toggle "Load More" visibility
  loadMoreBtn.style.display =
    state.displayedCount < state.filteredMovies.length ? "" : "none";
}

function createMovieCard(movie) {
  const card = document.createElement("div");
  card.className = "movie-card" + (state.ratings[movie.id] ? " rated" : "");
  card.dataset.movieId = movie.id;

  const titleEl = document.createElement("div");
  titleEl.className = "movie-title";
  titleEl.textContent = movie.title;

  const genresEl = document.createElement("div");
  genresEl.className = "movie-genres";
  genresEl.textContent = movie.genres.join(" · ");

  const stars = document.createElement("div");
  stars.className = "star-rating";
  for (let i = 1; i <= 5; i++) {
    const star = document.createElement("span");
    star.className = "star" + (state.ratings[movie.id] >= i ? " filled" : "");
    star.textContent = "★";
    star.dataset.value = i;
    stars.appendChild(star);
  }

  card.append(titleEl, genresEl, stars);
  return card;
}

// ---------------------------------------------------------------------------
// Star rating interaction (event delegation)
// ---------------------------------------------------------------------------
movieGrid.addEventListener("click", (e) => {
  const star = e.target.closest(".star");
  if (!star) return;
  const card = star.closest(".movie-card");
  const movieId = parseInt(card.dataset.movieId, 10);
  const value = parseInt(star.dataset.value, 10);

  // Toggle: clicking same star again removes the rating
  if (state.ratings[movieId] === value) {
    delete state.ratings[movieId];
    card.classList.remove("rated");
  } else {
    state.ratings[movieId] = value;
    card.classList.add("rated");
  }

  // Update star visuals
  card.querySelectorAll(".star").forEach((s) => {
    const v = parseInt(s.dataset.value, 10);
    s.classList.toggle("filled", state.ratings[movieId] >= v);
  });

  updateRatedCount();
});

// Hover preview
movieGrid.addEventListener("mouseover", (e) => {
  const star = e.target.closest(".star");
  if (!star) return;
  const card = star.closest(".movie-card");
  const hoverVal = parseInt(star.dataset.value, 10);
  card.querySelectorAll(".star").forEach((s) => {
    s.classList.toggle("hovered", parseInt(s.dataset.value, 10) <= hoverVal);
  });
});
movieGrid.addEventListener("mouseout", (e) => {
  const star = e.target.closest(".star");
  if (!star) return;
  star.closest(".movie-card").querySelectorAll(".star").forEach((s) => {
    s.classList.remove("hovered");
  });
});

function updateRatedCount() {
  const count = Object.keys(state.ratings).length;
  ratedCount.textContent = count;
  submitBtn.disabled = count === 0;
}

// ---------------------------------------------------------------------------
// Load more button
// ---------------------------------------------------------------------------
loadMoreBtn.addEventListener("click", renderNextPage);

// ---------------------------------------------------------------------------
// Fitted model loading — populates the algorithm dropdown
// ---------------------------------------------------------------------------

async function loadModels() {
  const noModelsNotice = document.getElementById("no-models-notice");
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;
    const data = await res.json();
    const models = data.models;

    if (models.length === 0) {
      algorithmSel.style.display = "none";
      noModelsNotice.style.display = "";
      submitBtn.disabled = true;
      return;
    }

    algorithmSel.style.display = "";
    noModelsNotice.style.display = "none";
    algorithmSel.innerHTML = "";

    models.forEach((model) => {
      const opt = document.createElement("option");
      opt.value = model.name;
      opt.textContent = model.label + _formatParamsSuffix(model.params);
      opt.dataset.params = JSON.stringify(model.params);
      algorithmSel.appendChild(opt);
    });
  } catch { /* API not available */ }
}

function _formatParamsSuffix(params) {
  const entries = Object.entries(params);
  if (entries.length === 0) return "";
  return " (" + entries.map(([k, v]) => `${k}=${v}`).join(", ") + ")";
}

// ---------------------------------------------------------------------------
// Submit → get recommendations
// ---------------------------------------------------------------------------
submitBtn.addEventListener("click", async () => {
  const selectedOpt = algorithmSel.options[algorithmSel.selectedIndex];
  const algorithm = selectedOpt.value;
  const params = JSON.parse(selectedOpt.dataset.params || "{}");
  const algoLabel = selectedOpt.text;

  // Show loading view
  loadingAlgo.textContent = algoLabel;
  showView("loading");

  try {
    const recommendations = await fetchRecommendations(algorithm, params, state.ratings);
    renderResults(recommendations, algoLabel);
    showView("results");
  } catch (err) {
    console.error("Recommendation request failed:", err);
    alert("Recommendation failed — is the API server running?\n" + err.message);
    showView("rating");
  }
});

async function fetchRecommendations(algorithm, params, ratings) {
  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ algorithm, params, ratings }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  const data = await res.json();
  // Normalise response shape to what renderResults expects
  return data.recommendations.map((r) => ({
    movieId: r.movieId,
    title: r.title,
    genres: r.genres,
    score: r.score,
  }));
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------
function renderResults(recommendations, algoLabel) {
  resultsAlgo.textContent = algoLabel;
  resultsList.innerHTML = "";

  recommendations.forEach((rec, i) => {
    const card = document.createElement("div");
    card.className = "rec-card";
    card.innerHTML = `
      <div class="rec-rank">${i + 1}</div>
      <div class="rec-info">
        <div class="rec-title">${escapeHtml(rec.title)}</div>
        <div class="rec-genres">${rec.genres.join(" · ")}</div>
      </div>
      <div class="rec-score">score <strong>${rec.score.toFixed(2)}</strong></div>
    `;
    resultsList.appendChild(card);
  });
}

/** HTML-escape to prevent XSS from movie titles. */
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Back button
// ---------------------------------------------------------------------------
backBtn.addEventListener("click", () => showView("rating"));



// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async function init() {
  await Promise.all([loadMovies(), loadModels()]);
  buildGenreFilters();

  // Initial render
  state.filteredMovies = [...state.movies];
  renderNextPage();
  showView("rating");
})();
