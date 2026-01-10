document.addEventListener("DOMContentLoaded", async () => {
  const skeletonTpl = document.getElementById("card-skeleton").content;
  const addButton = document.getElementById("add-market-button");
  const parent = addButton.parentNode;
  const datePicker = document.getElementById("date-picker");
  const API_BASE = "https://llm.indonesiacore.com";
  if (datePicker) datePicker.value = new Date().toISOString().slice(0, 10);

  const alertContainer = document.getElementById("prediction-alert");
  function updatePredictionAlert() {
    if (!alertContainer) return;
    const val = datePicker && datePicker.value;
    const todayStr = new Date().toISOString().slice(0, 10);
    if (val && new Date(val) > new Date(todayStr)) {
      alertContainer.innerHTML = `<div class="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-lg flex gap-4 items-start" role="status">
                    <div class="p-2 bg-blue-100 dark:bg-blue-800 rounded-full text-primary dark:text-blue-300">
                        <span class="material-symbols-outlined text-[20px]">info</span>
                    </div>
                    <div>
                        <h4 class="font-bold text-slate-900 dark:text-white text-sm">Predicted data</h4>
                        <p class="text-sm text-slate-600 dark:text-slate-300 mt-1">Data untuk <strong>${val}</strong> adalah hasil prediksi/perkiraan dan dapat berbeda dari harga perdagangan nyata.</p>
                    </div>
                </div>`;
      alertContainer.classList.remove("hidden");
    } else {
      alertContainer.classList.add("hidden");
      alertContainer.innerHTML = "";
    }
  }

  const LOCAL_KEY = "user_markets";
  const LOCAL_FORECAST_KEY = "forecast_markets";

  // In-memory caches for Preferences (to avoid refactoring every call site to async)
  let cacheUserMarkets = [];
  let cacheForecastMarkets = [];
  let cacheLastForecast = null;

  // Initialize Capacitor Preferences (dynamic import with fallback)
  async function initPreferences() {
    try {
      // Determine Preferences implementation at runtime via globals only (avoid bundler resolving '@capacitor/preferences')
      let Pref = null;
      try {
        // Common runtime locations for Capacitor Preferences plugin
        if (window && window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.Preferences) {
          Pref = window.Capacitor.Plugins.Preferences;
        } else if (window && window.CapacitorPreferences) {
          // some environments expose it differently
          Pref = window.CapacitorPreferences;
        } else if (globalThis && globalThis.Capacitor && globalThis.Capacitor.Plugins && globalThis.Capacitor.Plugins.Preferences) {
          Pref = globalThis.Capacitor.Plugins.Preferences;
        } else {
          Pref = null;
        }
      } catch (e) {
        Pref = null;
      }
      window._Preferences = Pref;

      if (Pref && Pref.get) {
        const u = await Pref.get({ key: LOCAL_KEY });
        cacheUserMarkets = u && u.value ? JSON.parse(u.value) : [];
        const f = await Pref.get({ key: LOCAL_FORECAST_KEY });
        cacheForecastMarkets = f && f.value ? JSON.parse(f.value) : [];
        const l = await Pref.get({ key: 'last_forecast' });
        cacheLastForecast = l && l.value ? JSON.parse(l.value) : null;
      } else {
        // fallback to localStorage for non-capacitor environments
        try {
          cacheUserMarkets = JSON.parse(localStorage.getItem(LOCAL_KEY) || '[]');
        } catch (e) {
          cacheUserMarkets = [];
        }
        try {
          cacheForecastMarkets = JSON.parse(localStorage.getItem(LOCAL_FORECAST_KEY) || '[]');
        } catch (e) {
          cacheForecastMarkets = [];
        }
        try {
          const l = localStorage.getItem('last_forecast');
          cacheLastForecast = l ? JSON.parse(l) : null;
        } catch (e) {
          cacheLastForecast = null;
        }
      }
    } catch (e) {
      console.warn('initPreferences failed', e);
    }
  }

  function readUserMarkets() {
    return Array.isArray(cacheUserMarkets) ? cacheUserMarkets.slice() : [];
  }
  function writeUserMarkets(arr) {
    cacheUserMarkets = Array.isArray(arr) ? arr : [];
    if (window._Preferences && window._Preferences.set) {
      window._Preferences.set({ key: LOCAL_KEY, value: JSON.stringify(cacheUserMarkets) }).catch(e => console.warn('Failed to persist user markets', e));
    } else {
      try { localStorage.setItem(LOCAL_KEY, JSON.stringify(cacheUserMarkets)); } catch (e) {}
    }
  }

  // helpers
  function formatIDR(v) {
    if (v === null || v === undefined) return "-";
    const n = Number(v);
    if (isNaN(n)) return "-";
    if (Math.abs(n - Math.round(n)) < 0.005) {
      return "Rp " + Math.round(n).toLocaleString("id-ID");
    }
    return (
      "Rp " +
      n.toLocaleString("id-ID", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })
    );
  }

  function createSparkline(series) {
    if (!series || series.length < 2) return "";
    const values = series
      .slice(0, 20)
      .map((s) => (s && s.idr_value ? Number(s.idr_value) : null))
      .filter((v) => v !== null);
    if (!values.length) return "";
    const w = 120,
      h = 34,
      pad = 2;
    const min = Math.min(...values),
      max = Math.max(...values);
    const range = max - min || 1;
    const step = (w - pad * 2) / (values.length - 1);
    let d = "";
    values.forEach((v, i) => {
      const x = pad + i * step;
      const y = h - pad - ((v - min) / range) * (h - pad * 2);
      d += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });
    return `<svg class="w-full h-8" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${d}" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }

  function formatNativeAmount(v, currency) {
    if (v === null || v === undefined) return "-";
    const n = Number(v);
    if (isNaN(n)) return "-";
    if (currency === "USD") {
      return (
        "$" +
        n.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })
      );
    }
    // fallback: show number with 2 decimals when not integer
    if (Math.abs(n - Math.round(n)) < 0.005)
      return Math.round(n).toLocaleString();
    return n.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  function createCard(ex) {
    const priceIDR =
      ex && ex.idr_value !== undefined && ex.idr_value !== null
        ? ex.idr_value
        : (ex && ex.series && ex.series[0] && ex.series[0].idr_value) || null;
    const details = ex && ex.details ? ex.details : {};
    const currency = ex && ex.currency ? ex.currency : null;

    const card = document.createElement("div");
    card.className =
      "flex flex-col gap-2 p-5 rounded-xl bg-white dark:bg-[#1e293b] border border-slate-200 dark:border-slate-700/50 shadow-sm hover:shadow-lg transition-all";

    const defaults = ["USD", "GOLD", "SAR"];
    const symKeyUpper = String(ex.symbol).toUpperCase();

    if (
      details &&
      details.last_price !== undefined &&
      details.last_price !== null
    ) {
      // Stock-like (has last_price) -> show native price + long name
      const native = formatNativeAmount(details.last_price, currency || "USD");
      const longName =
        details.longName || details.longname || details.name || "";
      const exch = details.exchange ? ` · ${details.exchange}` : "";
      card.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex flex-col">
                        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">${
                          ex.symbol
                        }${exch} / IDR</span>
                        <span class="text-2xl font-bold text-slate-900 dark:text-white mt-1">${formatIDR(
                          priceIDR
                        )}</span>
                    </div>
                    <div class="text-sm text-slate-500">${native}</div>
                </div>
                ${
                  longName
                    ? `<div class="text-sm text-slate-500 mt-2">${longName}</div>`
                    : ""
                }
            `;
    } else {
      // Currency-like -> show unit and optional name
      const unit = details && details.unit ? details.unit : "";
      const name =
        details && (details.name || details.longName || details.longname)
          ? details.name || details.longName || details.longname
          : "";
      card.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex flex-col">
                        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">${
                          ex.symbol
                        } / IDR</span>
                        <span class="text-2xl font-bold text-slate-900 dark:text-white mt-1">${formatIDR(
                          priceIDR
                        )}</span>
                    </div>
                </div>
                <div class="text-sm text-slate-500 mt-2">${unit}${
        name ? " · " + name : ""
      }</div>
            `;
    }

    // Add delete button for non-default symbols (applies to stocks and currencies)
    if (!defaults.includes(symKeyUpper)) {
      const removeKey =
        details && details.original_yf_symbol
          ? details.original_yf_symbol
          : ex.symbol;
      const delHtml = `<button title="Remove" aria-label="Remove ${ex.symbol} from My Markets" class="absolute size-6 rounded-full bg-red-500 hover:bg-red-600 text-white p-1 delete-market-btn" style="top:-5px; right:-5px;" data-remove="${removeKey}"><span class="material-symbols-outlined text-[18px]">close</span></button>`;
      card.classList.add("relative");
      card.insertAdjacentHTML("afterbegin", delHtml);
      const delBtn = card.querySelector(".delete-market-btn");
      delBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        const key = delBtn.dataset.remove || "";
        let arr = readUserMarkets() || [];
        const keyUpper = String(key).trim().toUpperCase();
        // filter out matches by original yf symbol or cleaned symbol
        arr = arr.filter((x) => {
          const xu = String(x || "")
            .trim()
            .toUpperCase();
          return xu !== keyUpper && xu !== symKeyUpper;
        });
        writeUserMarkets(arr);
        updateModalSelectionCount();
        // refresh main grid
        loadAndRender();
      });
    }

    return card;
  }

            // load and render
            async function loadAndRender() {
                // clear existing cards except add button
                Array.from(parent.querySelectorAll('.flex.flex-col')).forEach(el => {
                    if (el.id !== 'add-market-button') el.remove();
                });
                // show skeletons
                for (let i = 0; i < 4; i++) parent.insertBefore(skeletonTpl.cloneNode(true), addButton);

                try {
                    // include user-selected symbols if any (normalize & dedupe)
                    let userSymbols = (readUserMarkets() || []).map(s => String(s || '').trim().toUpperCase()).filter(Boolean);
                    userSymbols = Array.from(new Set(userSymbols));
                    const params = [];
                    if (datePicker && datePicker.value) params.push('date=' + datePicker.value);
                    if (userSymbols && userSymbols.length) params.push('symbols=' + encodeURIComponent(userSymbols.join(',')));
                    const url = API_BASE + '/exchange/idr/exchange' + (params.length ? ('?' + params.join('&')) : '');
                    console.debug('Fetching exchange URL:', url);
                    const res = await fetch(url);
                    if (!res.ok) throw new Error('Network response was not ok');
                    const data = await res.json();
                    const exchanges = data.exchanges || data.results || [];

                    const map = {};
                    exchanges.forEach(e => { if (e && e.symbol) map[String(e.symbol).toUpperCase()] = e; });

                    const user = userSymbols; // normalized & deduped
                    const finalList = [];
                    const missing = [];
                    // add only user symbols that are present in the API; hide placeholders for missing ones
                    user.forEach(s => {
                        if (map[s]) finalList.push(map[s]);
                        else missing.push(s);
                    });
                    if (missing.length) console.debug('Hidden user symbols missing from API:', missing);
                    // then add top exchanges (avoid duplicates)
                    exchanges.forEach(e => {
                        const eSym = String(e && e.symbol || '').toUpperCase();
                        if (!user.includes(eSym)) finalList.push(e);
                    });

                    // remove skeletons
                    const nodes = Array.from(parent.querySelectorAll('.animate-pulse, .bg-slate-200'));
                    nodes.forEach(n => n.closest('.flex.flex-col')?.remove());

                    // render cards
                    finalList.forEach((ex) => {
                        const card = createCard(ex);
                        parent.insertBefore(card, addButton);
                    });

                } catch (err) {
                    const nodes = Array.from(parent.querySelectorAll('.animate-pulse, .bg-slate-200'));
                    nodes.forEach(n => n.closest('.flex.flex-col')?.remove());
                    const errEl = document.createElement('div');
                    errEl.className = 'p-5 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800';
                    errEl.innerHTML = '<p class="text-sm text-red-600 dark:text-red-300">Failed to load market data</p>';
                    parent.insertBefore(errEl, addButton);
                    console.error('Failed fetching market data', err);
                }
            }

  function openModal() {
    modal.classList.remove("hidden");
    modalError.classList.add("hidden");
  }
  function closeModal() {
    modal.classList.add("hidden");
  }

  addButton.addEventListener("click", (e) => {
    e.preventDefault();
    openModal();
    loadModalLists();
    updateModalSelectionCount();
  });

  const modal = document.getElementById("add-market-modal");
  const modalBackdrop = modal && modal.querySelector("[data-modal-backdrop]");
  const modalClose = document.getElementById("modal-close");
  const modalCancel = document.getElementById("modal-cancel");
  const modalAdd = document.getElementById("modal-add");
  const modalError = document.getElementById("modal-error");

  modalBackdrop?.addEventListener("click", closeModal);
  modalClose?.addEventListener("click", closeModal);
  modalCancel?.addEventListener("click", closeModal);

  // tab handling
  const tabCurrencies = document.getElementById("tab-currencies");
  const tabStocks = document.getElementById("tab-stocks");
  const currenciesList = document.getElementById("currencies-list");
  const stocksList = document.getElementById("stocks-list");
  const stocksSearchWrapper = document.getElementById("stocks-search-wrapper");
  const stocksSearchInput = document.getElementById("stocks-search");

  // store the trending list so we can fall back to it when search is cleared
  let modalTrendingStocks = [];
  let searchTimer = null;

  function renderStocks(items) {
    if (!items || !items.length) {
      stocksList.innerHTML =
        '<div class="text-sm text-slate-500 p-2">No results found.</div>';
      return;
    }
    stocksList.innerHTML = "";
    items.forEach((s) => {
      const code = (s.symbol || s.code || "").toUpperCase();
      // prefer long display name when available
      const longName =
        s.longname ||
        s.longName ||
        s.longNameDisplay ||
        s.long_name ||
        s.long ||
        s.shortName ||
        s.shortname ||
        s.name ||
        s.title ||
        "";
      const available = modalExchangeMap[code] ? true : false;
      const row = document.createElement("button");
      row.className =
        "w-full text-left p-2 rounded hover:bg-slate-100 dark:hover:bg-[#071829] flex items-center justify-between";
      row.dataset.symbol = code;
      const selected = readUserMarkets().includes(code);
      if (selected) row.classList.add("bg-primary/5");
      row.innerHTML = `<div><div class="font-medium">${code}</div><div class="text-xs text-slate-500">${longName}</div></div><div class="text-sm ${
        available ? "text-emerald-600" : "text-slate-400"
      }">${available ? "Available" : ""} Saham</div>`;
      row.addEventListener("click", (e) => {
        e.preventDefault();
        let arr = readUserMarkets();
        if (arr.includes(code)) {
          arr = arr.filter((x) => x !== code);
          writeUserMarkets(arr);
          row.classList.remove("bg-primary/5");
        } else {
          arr.push(code);
          writeUserMarkets(arr);
          row.classList.add("bg-primary/5");
        }
        // selection only updates localStorage; final reload happens when Add Market is clicked
        updateModalSelectionCount();
      });
      stocksList.appendChild(row);
    });
  }

  function renderTrendingStocks() {
    renderStocks(modalTrendingStocks);
  }

  if (stocksSearchInput) {
    stocksSearchInput.addEventListener("input", () => {
      const q = stocksSearchInput.value.trim();
      clearTimeout(searchTimer);
      searchTimer = setTimeout(async () => {
        if (!q) {
          renderTrendingStocks();
          return;
        }
        try {
          stocksList.innerHTML =
            '<div class="text-sm text-slate-500 p-2">Searching...</div>';
          const res = await fetch(
            API_BASE +
              "/exchange/idr/market/search?query=" +
              encodeURIComponent(q)
          );
          if (!res.ok) throw new Error("Search failed");
          const j = await res.json();
          const results = j.results || j.items || j.exchanges || [];
          if (!results.length) {
            stocksList.innerHTML =
              '<div class="text-sm text-slate-500 p-2">No results found.</div>';
          } else {
            renderStocks(results);
          }
        } catch (e) {
          stocksList.innerHTML =
            '<div class="text-sm text-red-500 p-2">Search failed</div>';
          console.error("Stock search failed", e);
        }
      }, 300);
    });
  }

  function activateTab(tab) {
    if (tab === "currencies") {
      tabCurrencies.classList.add("bg-white");
      tabCurrencies.classList.add("dark:bg-[#0b1220]");
      tabCurrencies.classList.remove("text-slate-600");

      tabStocks.classList.remove("bg-white");
      tabStocks.classList.remove("dark:bg-[#0b1220]");
      tabStocks.classList.add("text-slate-600");

      currenciesList.classList.remove("hidden");
      stocksList.classList.add("hidden");
      stocksSearchWrapper?.classList.add("hidden");
      if (stocksSearchInput) stocksSearchInput.value = "";
    } else {
      tabStocks.classList.add("bg-white");
      tabStocks.classList.add("dark:bg-[#0b1220]");
      tabStocks.classList.remove("text-slate-600");

      tabCurrencies.classList.remove("bg-white");
      tabCurrencies.classList.remove("dark:bg-[#0b1220]");
      tabCurrencies.classList.add("text-slate-600");

      currenciesList.classList.add("hidden");
      stocksList.classList.remove("hidden");
      stocksSearchWrapper?.classList.remove("hidden");
      stocksSearchInput?.focus();
    }
  }

  tabCurrencies.addEventListener("click", () => activateTab("currencies"));
  tabStocks.addEventListener("click", () => activateTab("stocks"));

  // Forecast modal tab wiring (mirror Add Market modal behavior)
  const forecastTabCurrencies = document.getElementById(
    "forecast-tab-currencies"
  );
  const forecastTabStocks = document.getElementById("forecast-tab-stocks");
  const forecastCurrenciesList = document.getElementById(
    "forecast-currencies-list"
  );
  const forecastStocksList = document.getElementById("forecast-stocks-list");
  const forecastStocksSearchWrapperElem = document.getElementById(
    "forecast-stocks-search-wrapper"
  );
  const forecastStocksSearchInput = document.getElementById(
    "forecast-stocks-search"
  );
  let forecastSearchTimer = null;

  function activateForecastTab(tab) {
    if (tab === "currencies") {
      forecastTabCurrencies.classList.add("bg-white");
      forecastTabCurrencies.classList.add("dark:bg-[#0b1220]");
      forecastTabCurrencies.classList.remove("text-slate-600");

      forecastTabStocks.classList.remove("bg-white");
      forecastTabStocks.classList.remove("dark:bg-[#0b1220]");
      forecastTabStocks.classList.add("text-slate-600");

      forecastCurrenciesList.classList.remove("hidden");
      forecastStocksList.classList.add("hidden");
      forecastStocksSearchWrapperElem?.classList.add("hidden");
      if (forecastStocksSearchInput) forecastStocksSearchInput.value = "";
    } else {
      forecastTabStocks.classList.add("bg-white");
      forecastTabStocks.classList.add("dark:bg-[#0b1220]");
      forecastTabStocks.classList.remove("text-slate-600");

      forecastTabCurrencies.classList.remove("bg-white");
      forecastTabCurrencies.classList.remove("dark:bg-[#0b1220]");
      forecastTabCurrencies.classList.add("text-slate-600");

      forecastCurrenciesList.classList.add("hidden");
      forecastStocksList.classList.remove("hidden");
      forecastStocksSearchWrapperElem?.classList.remove("hidden");
      forecastStocksSearchInput?.focus();
    }
  }

  forecastTabCurrencies?.addEventListener("click", () =>
    activateForecastTab("currencies")
  );
  forecastTabStocks?.addEventListener("click", () =>
    activateForecastTab("stocks")
  );

  // Debounced search for forecast modal stocks (mirror behavior in main modal)
  if (forecastStocksSearchInput) {
    forecastStocksSearchInput.addEventListener("input", () => {
      const q = forecastStocksSearchInput.value.trim();
      clearTimeout(forecastSearchTimer);
      forecastSearchTimer = setTimeout(async () => {
        if (!q) {
          // render trending fallback
          if (!modalTrendingStocks || !modalTrendingStocks.length) {
            forecastStocksList.innerHTML =
              '<div class="text-sm text-slate-500 p-2">No trending stocks found.</div>';
            return;
          }
          forecastStocksList.innerHTML = "";
          modalTrendingStocks.forEach((s) => {
            const code = (s.symbol || "").toUpperCase();
            const longName = s.shortName || s.longName || "";
            const available = modalExchangeMap[code] ? true : false;
            const row = document.createElement("button");
            row.className =
              "w-full text-left p-2 rounded hover:bg-slate-100 dark:hover:bg-[#071829] flex items-center justify-between";
            row.dataset.symbol = code;
            const selected = readForecastMarkets().includes(code);
            if (selected) row.classList.add("bg-primary/5");
            row.innerHTML = `<div><div class="font-medium">${code}</div><div class="text-xs text-slate-500">${longName}</div></div><div class="text-sm ${
              available ? "text-emerald-600" : "text-slate-400"
            }">${available ? "Available" : ""} Saham</div>`;
            row.addEventListener("click", (e) => {
              e.preventDefault();
              let arr = readForecastMarkets();
              if (arr.includes(code)) {
                arr = arr.filter((x) => x !== code);
                writeForecastMarkets(arr);
                row.classList.remove("bg-primary/5");
              } else {
                arr.push(code);
                writeForecastMarkets(arr);
                row.classList.add("bg-primary/5");
              }
              updateForecastModalSelectionCount();
            });
            forecastStocksList.appendChild(row);
          });
          return;
        }
        try {
          forecastStocksList.innerHTML =
            '<div class="text-sm text-slate-500 p-2">Searching...</div>';
          const res = await fetch(
            API_BASE +
              "/exchange/idr/market/search?query=" +
              encodeURIComponent(q)
          );
          if (!res.ok) throw new Error("Search failed");
          const j = await res.json();
          const results = j.results || j.items || j.exchanges || [];
          if (!results.length) {
            forecastStocksList.innerHTML =
              '<div class="text-sm text-slate-500 p-2">No results found.</div>';
          } else {
            forecastStocksList.innerHTML = "";
            results.forEach((r) => {
              const code = (r.symbol || r.code || "").toUpperCase();
              const longName = r.longname || r.longName || r.name || "";
              const available = modalExchangeMap[code] ? true : false;
              const row = document.createElement("button");
              row.className =
                "w-full text-left p-2 rounded hover:bg-slate-100 dark:hover:bg-[#071829] flex items-center justify-between";
              row.dataset.symbol = code;
              const selected = readForecastMarkets().includes(code);
              if (selected) row.classList.add("bg-primary/5");
              row.innerHTML = `<div><div class="font-medium">${code}</div><div class="text-xs text-slate-500">${longName}</div></div><div class="text-sm ${
                available ? "text-emerald-600" : "text-slate-400"
              }">${available ? "Available" : ""}</div>`;
              row.addEventListener("click", (e) => {
                e.preventDefault();
                let arr = readForecastMarkets();
                if (arr.includes(code)) {
                  arr = arr.filter((x) => x !== code);
                  writeForecastMarkets(arr);
                  row.classList.remove("bg-primary/5");
                } else {
                  arr.push(code);
                  writeForecastMarkets(arr);
                  row.classList.add("bg-primary/5");
                }
                updateForecastModalSelectionCount();
              });
              forecastStocksList.appendChild(row);
            });
          }
        } catch (e) {
          forecastStocksList.innerHTML =
            '<div class="text-sm text-red-500 p-2">Search failed</div>';
          console.error("Forecast stock search failed", e);
        }
      }, 300);
    });
  }

  // map of symbols fetched from the main exchange endpoint for validation/availability
  let modalExchangeMap = {};

  async function loadModalLists() {
    // show loading placeholders
    currenciesList.innerHTML =
      '<div class="text-sm text-slate-500 p-2">Loading currencies...</div>';
    stocksList.innerHTML =
      '<div class="text-sm text-slate-500 p-2">Loading trending stocks...</div>';
    activateTab("currencies");

    try {
      // Always fetch the main exchange list first (default behavior requested)
      modalExchangeMap = {};

      try {
        // include user symbols in the preload request so availability is accurate
        let userSymbols = (readUserMarkets() || [])
          .map((s) =>
            String(s || "")
              .trim()
              .toUpperCase()
          )
          .filter(Boolean);
        userSymbols = Array.from(new Set(userSymbols));
        const params = [];
        if (datePicker && datePicker.value)
          params.push("date=" + datePicker.value);
        if (userSymbols && userSymbols.length)
          params.push("symbols=" + encodeURIComponent(userSymbols.join(",")));
        const exUrl =
          API_BASE +
          "/exchange/idr/exchange" +
          (params.length ? "?" + params.join("&") : "");
        console.debug("Preloading modal exchange URL:", exUrl);
        const exRes = await fetch(exUrl);
        const exJson = await exRes.json();
        const exList = exJson.exchanges || exJson.results || [];
        exList.forEach((e) => {
          if (e && e.symbol)
            modalExchangeMap[String(e.symbol).toUpperCase()] = e;
        });
      } catch (exErr) {
        console.warn("Failed to preload exchange list for modal", exErr);
      }

      const [cRes, sRes] = await Promise.all([
        fetch(API_BASE + "/exchange/idr/currencies"),
        fetch(API_BASE + "/exchange/idr/get_trend?count=15"),
      ]);

      const cJson = await cRes.json();
      const sJson = await sRes.json();
      const cur = cJson.currencies || [];
      const st = sJson.results || [];

      // render currencies
      if (!cur.length) {
        currenciesList.innerHTML =
          '<div class="text-sm text-slate-500 p-2">No currencies found.</div>';
      } else {
        currenciesList.innerHTML = "";
        cur.forEach((c) => {
          const code = (c.code || c.currency || c.symbol || "").toUpperCase();
          const name = c.name || "";
          const available = modalExchangeMap[code] ? true : false;
          const row = document.createElement("button");
          row.className =
            "w-full text-left p-2 rounded hover:bg-slate-100 dark:hover:bg-[#071829] flex items-center justify-between";
          row.dataset.symbol = code;
          const selected = readUserMarkets().includes(code);
          if (selected) row.classList.add("bg-primary/5");
          row.innerHTML = `<div><div class="font-medium">${code}</div><div class="text-xs text-slate-500">${name}</div></div><div class="text-sm ${
            available ? "text-emerald-600" : "text-slate-400"
          }">${available ? "Available" : ""} Mata Uang</div>`;
          row.addEventListener("click", (e) => {
            e.preventDefault();
            const s = code;
            let arr = readUserMarkets();
            if (arr.includes(s)) {
              // deselect
              arr = arr.filter((x) => x !== s);
              writeUserMarkets(arr);
              row.classList.remove("bg-primary/5");
            } else {
              // select
              arr.push(s);
              writeUserMarkets(arr);
              row.classList.add("bg-primary/5");
            }
            // only update localStorage and UI; actual reload happens when Add Market is clicked
            updateModalSelectionCount();
          });
          currenciesList.appendChild(row);
        });
      }

      // render stocks (keep trending list so we can fall back when search is cleared)
      modalTrendingStocks = st || [];
      if (!modalTrendingStocks.length) {
        stocksList.innerHTML =
          '<div class="text-sm text-slate-500 p-2">No trending stocks found.</div>';
      } else {
        renderTrendingStocks();
      }
    } catch (e) {
      currenciesList.innerHTML =
        '<div class="text-sm text-red-500 p-2">Failed loading lists</div>';
      stocksList.innerHTML =
        '<div class="text-sm text-red-500 p-2">Failed loading lists</div>';
      console.error("Failed to load lists", e);
    }
  }

  async function addSymbol(sym) {
    const s = (sym || "").trim().toUpperCase();
    if (!s) return;
    const user = readUserMarkets();
    if (user.includes(s)) {
      modalError.textContent = "Symbol already added.";
      modalError.classList.remove("hidden");
      return;
    }

    // Prefer using preloaded exchange map for validation
    if (modalExchangeMap && modalExchangeMap[s]) {
      user.push(s);
      writeUserMarkets(user);
      closeModal();
      await loadAndRender();
      return;
    }

    // fallback: validate by fetching exchange list
    try {
      const res = await fetch(
        API_BASE +
          "/exchange/idr/exchange" +
          (datePicker && datePicker.value ? "?date=" + datePicker.value : "")
      );
      const data = await res.json();
      const items = data.exchanges || data.results || [];
      const found = items.find((i) => i.symbol === s);
      if (!found) {
        modalError.textContent = "Symbol tidak ditemukan pada data terbaru.";
        modalError.classList.remove("hidden");
        return;
      }
      user.push(s);
      writeUserMarkets(user);
      closeModal();
      await loadAndRender();
    } catch (e) {
      modalError.textContent = "Failed to validate symbol. Try again.";
      modalError.classList.remove("hidden");
      console.error(e);
    }
  }

  // helper to update add button label with selection count
  function updateModalSelectionCount() {
    const arr = readUserMarkets();
    const n = arr.length;
    modalAdd.textContent = n ? `Add Market (${n})` : "Add Market";
  }

  // When Add Market clicked: make a single request to /exchange/idr/exchange with symbols (comma-separated) and date if selected
  modalAdd.addEventListener("click", async () => {
    try {
      const user = readUserMarkets();
      if (!user || !user.length) {
        modalError.textContent = "No symbols selected.";
        modalError.classList.remove("hidden");
        return;
      }
      modalError.classList.add("hidden");
      // Close modal first to avoid conflicts, then reload exchanges which will read from localStorage
      closeModal();
      await loadAndRender();
    } catch (e) {
      modalError.textContent = "Failed to add markets. Try again.";
      modalError.classList.remove("hidden");
      console.error(e);
    }
  });

  // wire date change
  datePicker?.addEventListener("change", async () => {
    updatePredictionAlert();
    await loadAndRender();
  });

  // ensure alert is correct on initial render
  updatePredictionAlert();

  // AI Forecast form wiring
  const aiAmountInput = document.getElementById("ai-amount");
  const aiStartInput = document.getElementById("ai-start-date");
  const aiEndInput = document.getElementById("ai-end-date");
  const btn1w = document.getElementById("btn-period-1w");
  const btn1m = document.getElementById("btn-period-1m");
  const btn3m = document.getElementById("btn-period-3m");
  const runAiBtn = document.getElementById("run-ai-btn");
  const aiErrorEl = document.getElementById("ai-error");
  const aiResultsEl = document.getElementById("ai-results");

  // Forecast markets persistent storage (separate from overview markets)
  const forecastSelectedEl = document.getElementById("forecast-selected");

  function readForecastMarkets() {
    return Array.isArray(cacheForecastMarkets) ? cacheForecastMarkets.slice() : [];
  }
  function writeForecastMarkets(arr) {
    cacheForecastMarkets = Array.isArray(arr) ? arr : [];
    if (window._Preferences && window._Preferences.set) {
      window._Preferences.set({ key: LOCAL_FORECAST_KEY, value: JSON.stringify(cacheForecastMarkets) }).catch(e => console.warn('Failed to persist forecast markets', e));
    } else {
      try { localStorage.setItem(LOCAL_FORECAST_KEY, JSON.stringify(cacheForecastMarkets)); } catch (e) {}
    }
  }

  function renderForecastSelected() {
    const arr = readForecastMarkets();
    if (!forecastSelectedEl) return;
    forecastSelectedEl.innerHTML = "";
    arr.forEach((s) => {
      const chip = document.createElement("div");
      chip.className =
        "inline-flex items-center gap-2 bg-slate-100 dark:bg-[#07101a] px-2 py-1 rounded-full text-sm";
      chip.innerHTML = `<span class="font-medium">${s}</span><button class="ml-2 text-slate-400 hover:text-red-500 remove-forecast-btn" data-symbol="${s}"><span class="material-symbols-outlined text-[16px]">close</span></button>`;
      forecastSelectedEl.appendChild(chip);
      chip
        .querySelector(".remove-forecast-btn")
        ?.addEventListener("click", (e) => {
          e.preventDefault();
          const sym = e.currentTarget.dataset.symbol;
          let list = readForecastMarkets();
          list = list.filter((x) => x !== sym);
          writeForecastMarkets(list);
          renderForecastSelected();
        });
    });
  }

  // Forecast modal wiring
  const selectForecastBtn = document.getElementById("select-forecast-markets");
  const forecastModal = document.getElementById("forecast-market-modal");
  const forecastBackdrop =
    forecastModal && forecastModal.querySelector("[data-forecast-backdrop]");
  const forecastModalClose = document.getElementById("forecast-modal-close");
  const forecastModalCancel = document.getElementById("forecast-modal-cancel");
  const forecastModalAdd = document.getElementById("forecast-modal-add");
  const forecastModalError = document.getElementById("forecast-modal-error");

  function openForecastModal() {
    forecastModal.classList.remove("hidden");
    forecastModalError.classList.add("hidden");
    activateForecastTab("currencies");
    if (forecastStocksSearchInput) forecastStocksSearchInput.value = "";
  }
  function closeForecastModal() {
    forecastModal.classList.add("hidden");
  }

  selectForecastBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    openForecastModal();
    loadForecastModalLists();
    updateForecastModalSelectionCount();
  });
  forecastBackdrop?.addEventListener("click", closeForecastModal);
  forecastModalClose?.addEventListener("click", closeForecastModal);
  forecastModalCancel?.addEventListener("click", closeForecastModal);

  function updateForecastModalSelectionCount() {
    const arr = readForecastMarkets();
    const n = arr.length;
    if (forecastModalAdd)
      forecastModalAdd.textContent = n ? `Add Markets (${n})` : "Add Markets";
  }

  async function loadForecastModalLists() {
    const fCurrenciesList = document.getElementById("forecast-currencies-list");
    const fStocksList = document.getElementById("forecast-stocks-list");
    const fStocksSearchWrapper = document.getElementById(
      "forecast-stocks-search-wrapper"
    );
    const fStocksSearch = document.getElementById("forecast-stocks-search");

    if (fCurrenciesList)
      fCurrenciesList.innerHTML =
        '<div class="text-sm text-slate-500 p-2">Loading currencies...</div>';
    if (fStocksList)
      fStocksList.innerHTML =
        '<div class="text-sm text-slate-500 p-2">Loading trending stocks...</div>';

    try {
      // Preload exchange map for availability
      try {
        let userSymbols = (readUserMarkets() || [])
          .map((s) =>
            String(s || "")
              .trim()
              .toUpperCase()
          )
          .filter(Boolean);
        userSymbols = Array.from(new Set(userSymbols));
        const params = [];
        if (datePicker && datePicker.value)
          params.push("date=" + datePicker.value);
        if (userSymbols && userSymbols.length)
          params.push("symbols=" + encodeURIComponent(userSymbols.join(",")));
        const exUrl =
          API_BASE +
          "/exchange/idr/exchange" +
          (params.length ? "?" + params.join("&") : "");
        console.debug("Preloading forecast modal exchange URL:", exUrl);
        const exRes = await fetch(exUrl);
        const exJson = await exRes.json();
        const exList = exJson.exchanges || exJson.results || [];
        exList.forEach((e) => {
          if (e && e.symbol)
            modalExchangeMap[String(e.symbol).toUpperCase()] = e;
        });
      } catch (exErr) {
        console.warn(
          "Failed to preload exchange list for forecast modal",
          exErr
        );
      }

      const [cRes, sRes] = await Promise.all([
        fetch(API_BASE + "/exchange/idr/currencies"),
        fetch(API_BASE + "/exchange/idr/get_trend?count=15"),
      ]);
      const cJson = await cRes.json();
      const sJson = await sRes.json();
      const cur = cJson.currencies || [];
      const st = sJson.results || [];

      // render currencies
      if (!cur.length) {
        if (fCurrenciesList)
          fCurrenciesList.innerHTML =
            '<div class="text-sm text-slate-500 p-2">No currencies found.</div>';
      } else {
        if (fCurrenciesList) fCurrenciesList.innerHTML = "";
        cur.forEach((c) => {
          const code = (c.code || c.currency || c.symbol || "").toUpperCase();
          const name = c.name || "";
          const available = modalExchangeMap[code] ? true : false;
          const row = document.createElement("button");
          row.className =
            "w-full text-left p-2 rounded hover:bg-slate-100 dark:hover:bg-[#071829] flex items-center justify-between";
          row.dataset.symbol = code;
          const selected = readForecastMarkets().includes(code);
          if (selected) row.classList.add("bg-primary/5");
          row.innerHTML = `<div><div class="font-medium">${code}</div><div class="text-xs text-slate-500">${name}</div></div><div class="text-sm ${
            available ? "text-emerald-600" : "text-slate-400"
          }">${available ? "Available" : ""} Mata Uang</div>`;
          row.addEventListener("click", (e) => {
            e.preventDefault();
            const s = code;
            let arr = readForecastMarkets();
            if (arr.includes(s)) {
              arr = arr.filter((x) => x !== s);
              writeForecastMarkets(arr);
              row.classList.remove("bg-primary/5");
            } else {
              arr.push(s);
              writeForecastMarkets(arr);
              row.classList.add("bg-primary/5");
            }
            updateForecastModalSelectionCount();
          });
          fCurrenciesList.appendChild(row);
        });
      }

      // render stocks
      if (!st.length) {
        if (fStocksList)
          fStocksList.innerHTML =
            '<div class="text-sm text-slate-500 p-2">No trending stocks found.</div>';
      } else {
        if (fStocksList) fStocksList.innerHTML = "";
        st.forEach((s) => {
          const code = (s.symbol || "").toUpperCase();
          const longName = s.shortName || s.longName || "";
          const available = modalExchangeMap[code] ? true : false;
          const row = document.createElement("button");
          row.className =
            "w-full text-left p-2 rounded hover:bg-slate-100 dark:hover:bg-[#071829] flex items-center justify-between";
          row.dataset.symbol = code;
          const selected = readForecastMarkets().includes(code);
          if (selected) row.classList.add("bg-primary/5");
          row.innerHTML = `<div><div class="font-medium">${code}</div><div class="text-xs text-slate-500">${longName}</div></div><div class="text-sm ${
            available ? "text-emerald-600" : "text-slate-400"
          }">${available ? "Available" : ""} Saham</div>`;
          row.addEventListener("click", (e) => {
            e.preventDefault();
            let arr = readForecastMarkets();
            if (arr.includes(code)) {
              arr = arr.filter((x) => x !== code);
              writeForecastMarkets(arr);
              row.classList.remove("bg-primary/5");
            } else {
              arr.push(code);
              writeForecastMarkets(arr);
              row.classList.add("bg-primary/5");
            }
            updateForecastModalSelectionCount();
          });
          fStocksList.appendChild(row);
        });
      }
    } catch (e) {
      if (fCurrenciesList)
        fCurrenciesList.innerHTML =
          '<div class="text-sm text-red-500 p-2">Failed loading lists</div>';
      if (fStocksList)
        fStocksList.innerHTML =
          '<div class="text-sm text-red-500 p-2">Failed loading lists</div>';
      console.error("Failed to load forecast lists", e);
    }
  }

  forecastModalAdd?.addEventListener("click", async () => {
    try {
      const user = readForecastMarkets();
      if (!user || !user.length) {
        forecastModalError.textContent = "No symbols selected.";
        forecastModalError.classList.remove("hidden");
        return;
      }
      forecastModalError.classList.add("hidden");
      closeForecastModal();
      renderForecastSelected();
    } catch (e) {
      forecastModalError.textContent = "Failed to add markets. Try again.";
      forecastModalError.classList.remove("hidden");
      console.error(e);
    }
  });

  // initialize Preferences cache then render initial forecast selections
  await initPreferences();
  renderForecastSelected();

  // Restore last forecast if available from Preferences/local cache (symbols, amount, start, end, results)
  try {
    const saved = cacheLastForecast;
    if (saved) {
      const lf = saved;
      if (lf && lf.results) {
        // restore forecast markets (exclude auto-included defaults from chips)
        const rawSymbols =
          typeof lf.symbols === "string"
            ? lf.symbols
                .split(",")
                .map((x) => x.trim())
                .filter(Boolean)
            : Array.isArray(lf.symbols)
            ? lf.symbols
            : [];
        const autoIncluded = ["USD", "SAR", "GOLD"];
        const userSymbols = rawSymbols
          .map((s) => s.toUpperCase())
          .filter((s) => !autoIncluded.includes(s));
        writeForecastMarkets(userSymbols);
        renderForecastSelected();

        // restore form values (keep raw numeric value for number input)
        if (lf.amount !== undefined && lf.amount !== null)
          aiAmountInput.value = String(lf.amount);
        if (lf.start) aiStartInput.value = lf.start;
        if (lf.end) aiEndInput.value = lf.end;

        // render chart and insights from cached results
        renderForecastChart(lf.results, lf.start, lf.end, lf.amount);
        renderAIInsights(lf.results, lf.amount);
      }
    }
  } catch (e) {
    console.warn("Failed to load saved forecast", e);
  }

  function setPeriod(days, activeBtn) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const end = new Date(today);
    end.setDate(end.getDate() + days);
    if (aiStartInput) aiStartInput.value = today.toISOString().slice(0, 10);
    if (aiEndInput) aiEndInput.value = end.toISOString().slice(0, 10);
    [btn1w, btn1m, btn3m].forEach(
      (b) => b && b.classList.remove("bg-primary", "text-white")
    );
    if (activeBtn) activeBtn.classList.add("bg-primary", "text-white");
  }
  btn1w?.addEventListener("click", () => setPeriod(7, btn1w));
  btn1m?.addEventListener("click", () => setPeriod(30, btn1m));
  btn3m?.addEventListener("click", () => setPeriod(90, btn3m));
  // initialize default period
  setPeriod(7, btn1w);

  async function runAIPrediction() {
    aiErrorEl.classList.add("hidden");
    aiResultsEl.innerHTML = "";
    const amountRaw = aiAmountInput?.value || "";
    // Strip any non-digit characters (thousand separators, currency symbols, spaces) to avoid locale issues
    const amount = Number(String(amountRaw).replace(/[^0-9]/g, ""));
    if (!amount || isNaN(amount) || amount <= 0) {
      aiErrorEl.textContent = "Masukkan jumlah IDR yang valid.";
      aiErrorEl.classList.remove("hidden");
      return;
    }
    const start = aiStartInput?.value;
    const end = aiEndInput?.value;
    if (!start || !end) {
      aiErrorEl.textContent = "Pilih start dan end date.";
      aiErrorEl.classList.remove("hidden");
      return;
    }
    if (new Date(end) < new Date(start)) {
      aiErrorEl.textContent = "End date harus sama atau setelah Start date.";
      aiErrorEl.classList.remove("hidden");
      return;
    }
    const diffDays =
      Math.round((new Date(end) - new Date(start)) / (1000 * 60 * 60 * 24)) + 1;
    if (diffDays > 120) {
      aiErrorEl.textContent = "Range terlalu besar (maks 120 hari).";
      aiErrorEl.classList.remove("hidden");
      return;
    }
    // Use forecast-specific selections (separate from main localStorage markets); USD, SAR, and GOLD are included automatically
    const userSelected = (readForecastMarkets() || [])
      .map((s) => String(s).trim())
      .filter(Boolean);
    if (!userSelected.length) {
      aiErrorEl.textContent =
        "Pilih minimal 1 market untuk forecast. (USD, SAR dan GOLD akan otomatis disertakan)";
      aiErrorEl.classList.remove("hidden");
      return;
    }
    const autoIncluded = ["USD", "SAR", "GOLD"];
    const symbols = Array.from(
      new Set([...userSelected.map((s) => s.toUpperCase()), ...autoIncluded])
    ).join(",");

    // disable button and show loading state
    const btn = runAiBtn;
    const prevBtnHtml = btn ? btn.innerHTML : "";
    if (btn) {
      btn.disabled = true;
      btn.classList.add("opacity-60", "cursor-not-allowed");
      btn.innerHTML = `<svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path></svg>Memproses...`;
    }

    aiResultsEl.innerHTML =
      '<div class="p-3 bg-slate-50 dark:bg-[#07101a] rounded">Loading...</div>';
    try {
      const url = `${API_BASE}/exchange/idr/invest?max_days=120&symbols=${encodeURIComponent(
        symbols
      )}&amount=${encodeURIComponent(amount)}&start_date=${encodeURIComponent(
        start
      )}&end_date=${encodeURIComponent(end)}`;
      const r = await fetch(url);
      if (!r.ok) throw new Error("Network failed");
      const j = await r.json();
      let rows = [];
      if (j.results) rows = j.results;
      else rows = [j];
      // Persist selected forecast markets (ensure forecast_markets reflects what we just used)
      writeForecastMarkets(userSelected.map((s) => s.toUpperCase()));
      // Clear left-side results and render chart + insights on the right
      aiResultsEl.innerHTML = "";
      // Render chart and AI insights in the right panel
      renderForecastChart(rows, start, end, amount);
      renderAIInsights(rows, amount);
      // Save forecast results to Preferences/local cache so it can be restored on reload
      try {
        const lastForecast = {
          symbols: symbols,
          amount: amount,
          start: start,
          end: end,
          results: rows,
        };
        // update in-memory cache and persist asynchronously
        cacheLastForecast = lastForecast;
        if (window._Preferences && window._Preferences.set) {
          window._Preferences.set({ key: 'last_forecast', value: JSON.stringify(lastForecast) }).catch(e => console.warn('Failed to persist last forecast', e));
        } else {
          try { localStorage.setItem('last_forecast', JSON.stringify(lastForecast)); } catch (e) {}
        }
      } catch (e) {
        console.warn("Failed to persist last forecast", e);
      }
    } catch (e) {
      aiResultsEl.innerHTML =
        '<div class="text-sm text-red-500">Gagal mengambil prediksi</div>';
      console.error(e);
    } finally {
      // re-enable button and restore text
      if (btn) {
        btn.disabled = false;
        btn.classList.remove("opacity-60", "cursor-not-allowed");
        btn.innerHTML = prevBtnHtml;
      }
    }
  }
  // Render functions for chart and AI insights
  function renderForecastChart(results, startStr, endStr, amountPerSymbol) {
    try {
      const start = new Date(startStr);
      const end = new Date(endStr);
      const dates = [];
      for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
        dates.push(new Date(d).toISOString().slice(0, 10));
      }

      // colors helper
      function pickColor(i) {
        const hue = (i * 47) % 360;
        return `hsl(${hue} 70% 45%)`;
      }

      // prepare datasets: per-symbol portfolio value series (units_bought * idr_value)
      const datasets = [];
      results.forEach((r, idx) => {
        const units = Number(r.units_bought || 0);
        const data = dates.map((dt) => {
          const p = (r.series || []).find((s) => String(s.date) === String(dt));
          const idr =
            p && p.idr_value !== undefined && p.idr_value !== null
              ? Number(p.idr_value)
              : null;
          return idr !== null && units
            ? Number((idr * units).toFixed(6))
            : null;
        });
        datasets.push({
          label: r.symbol || "",
          data,
          borderColor: pickColor(idx),
          backgroundColor: pickColor(idx),
          spanGaps: true,
          tension: 0.2,
        });
      });

      // render with Chart.js
      const chartInner = document.getElementById("forecast-chart-inner");
      if (!chartInner) return;
      // create canvas if not present
      if (!document.getElementById("forecast-chart-canvas"))
        chartInner.innerHTML =
          '<canvas id="forecast-chart-canvas" class="w-full h-full"></canvas>';
      const canvas = document.getElementById("forecast-chart-canvas");
      const ctx = canvas.getContext("2d");

      // destroy previous chart if exists
      if (window.forecastChart) {
        try {
          window.forecastChart.destroy();
        } catch (e) {}
        window.forecastChart = null;
      }

      window.forecastChart = new Chart(ctx, {
        type: "line",
        data: { labels: dates, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          scales: {
            x: { display: true },
            y: {
              display: true,
              ticks: {
                callback: function (v) {
                  return v ? formatIDR(v) : "";
                },
              },
            },
          },
          plugins: {
            tooltip: {
              callbacks: {
                label: (context) => {
                  const val = context.parsed.y;
                  return (
                    context.dataset.label + ": " + (val ? formatIDR(val) : "-")
                  );
                },
              },
            },
            legend: { position: "bottom" },
          },
        },
      });
    } catch (e) {
      console.error("Failed to render forecast chart", e);
    }
  }

  function renderAIInsights(results, amountPerSymbol) {
    try {
      const container = document.getElementById("ai-insights");
      if (!container) return;
      container.innerHTML = "";
      results.forEach((r) => {
        const sym = r.symbol || "";
        const ai = r.ai_analysis || {};
        const text = ai.text || "";
        const conf = ai.confidence || "";
        const proj = r.projected_final ? formatIDR(r.projected_final) : "-";
        const card = document.createElement("div");
        card.className =
          "p-4 bg-white dark:bg-[#0b1220] border border-slate-200 dark:border-slate-700 rounded";
        card.innerHTML = `<div class="flex justify-between items-start gap-4"><div><div class="font-bold">${sym}</div><div class="text-sm text-slate-500 mt-1">${text}</div></div><div class="text-right"><div class="text-sm font-semibold">${proj}</div><div class="text-xs text-slate-400">Confidence: ${conf}%</div></div></div>`;
        container.appendChild(card);
      });
    } catch (e) {
      console.error("Failed to render AI insights", e);
    }
  }

  runAiBtn?.addEventListener("click", runAIPrediction);

  // ensure Preferences cache is ready, then set initial modal add label count and initial load
  await initPreferences();
  updateModalSelectionCount();
  await loadAndRender();
});
