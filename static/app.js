/* GameSales AI dashboard scripts. All chart data is fetched from Flask JSON APIs
   (CSV / SQLite warehouse / trained-model metadata) - nothing is hard-coded here. */
(function () {
    "use strict";

    var CYAN = "#22d3ee", PURPLE = "#a855f7", BLUE = "#3b82f6", AMBER = "#f5b942", MUTED = "#9eacd0", GRID = "rgba(120,140,200,.14)";
    var GS = window.GS = {};

    if (window.Chart) {
        Chart.defaults.color = MUTED;
        Chart.defaults.font.family = "Inter, Arial, sans-serif";
        Chart.defaults.font.size = 11;
        Chart.defaults.borderColor = GRID;
        Chart.defaults.plugins.legend.labels.boxWidth = 12;
        Chart.defaults.plugins.tooltip.backgroundColor = "rgba(8,14,36,.95)";
        Chart.defaults.plugins.tooltip.borderColor = "rgba(139,107,255,.5)";
        Chart.defaults.plugins.tooltip.borderWidth = 1;
    }

    function $(id) { return document.getElementById(id); }
    function esc(v) {
        return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
            return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
        });
    }
    function getJSON(url) {
        return fetch(url).then(function (r) {
            return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.statusText); return j; });
        });
    }
    function showMsg(canvasId, msg) {
        var c = $(canvasId); if (!c) return;
        var d = document.createElement("div"); d.className = "chart-msg"; d.textContent = msg;
        c.parentNode.appendChild(d);
    }
    function fmt(n, d) { return Number(n).toLocaleString(undefined, { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); }

    /* ---------- sidebar ---------- */
    (function sidebar() {
        var body = document.body, btn = $("sbToggle"), menu = $("menuBtn");
        try { if (localStorage.getItem("gs-sb") === "1") body.classList.add("sb-collapsed"); } catch (e) {}
        if (btn) btn.addEventListener("click", function () {
            body.classList.toggle("sb-collapsed");
            try { localStorage.setItem("gs-sb", body.classList.contains("sb-collapsed") ? "1" : "0"); } catch (e) {}
            setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 260);
        });
        if (menu) menu.addEventListener("click", function () { body.classList.toggle("sb-open"); });
        document.addEventListener("click", function (e) {
            if (!body.classList.contains("sb-open")) return;
            if (!e.target.closest("#sidebar") && !e.target.closest("#menuBtn")) body.classList.remove("sb-open");
        });
    })();

    /* ---------- clickable pipelines ---------- */
    function initPipelines() {
        document.querySelectorAll("[data-pipeline]").forEach(function (el) {
            var data = JSON.parse($(el.id + "-data").textContent);
            var detail = $(el.id + "-detail");
            function show(i) {
                var s = data[i];
                el.querySelectorAll(".stage").forEach(function (b, k) { b.classList.toggle("active", k === i); });
                var cls = s.impl.indexOf("ACTUAL") === 0 ? "b-actual" : "b-concept";
                detail.innerHTML =
                    '<span class="badge ' + cls + '">' + esc(s.impl) + '</span>' +
                    '<h3>' + esc(s.icon) + " " + esc(s.title) + '</h3>' +
                    '<dl><dt>What it is</dt><dd>' + esc(s.what) + '</dd>' +
                    '<dt>In this project</dt><dd>' + esc(s.project) + '</dd>' +
                    '<dt>Implemented by</dt><dd><code>' + esc(s.component) + '</code></dd></dl>';
            }
            el.querySelectorAll(".stage").forEach(function (b) {
                b.addEventListener("click", function () { show(parseInt(b.getAttribute("data-idx"), 10)); });
            });
            show(0);
        });
    }

    /* ---------- star schema connector lines ---------- */
    GS.starLines = function () {
        document.querySelectorAll(".star-diagram").forEach(function (box) {
            var svg = box.querySelector("svg");
            function draw() {
                var fact = box.querySelector('[data-node="fact"]');
                var b = box.getBoundingClientRect(), f = fact.getBoundingClientRect();
                var html = "";
                box.querySelectorAll("[data-node]").forEach(function (n) {
                    if (n === fact) return;
                    var r = n.getBoundingClientRect();
                    var x1 = r.left + r.width / 2 - b.left, y1 = r.top + r.height / 2 - b.top;
                    var x2 = f.left + f.width / 2 - b.left, y2 = f.top + f.height / 2 - b.top;
                    html += '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '"/>';
                });
                svg.innerHTML = html;
            }
            draw();
            window.addEventListener("resize", draw);
            setTimeout(draw, 400);
        });
    };

    /* ---------- paginated fact table ---------- */
    var FACT_COLS = [
        ["Sales_ID", "Sales_ID", "fk-col", true], ["Game_ID", "Game_ID", "fk-col", true],
        ["Name", "Game", "", false],
        ["Platform_ID", "Platform_ID", "fk-col", true], ["Platform", "Platform", "", false],
        ["Genre_ID", "Genre_ID", "fk-col", true], ["Genre", "Genre", "", false],
        ["Publisher_ID", "Publisher_ID", "fk-col", true], ["Publisher", "Publisher", "", false],
        ["Year_ID", "Year_ID", "fk-col", true], ["Year", "Year", "", false],
        ["NA_Sales", "NA", "num", false], ["EU_Sales", "EU", "num", false], ["JP_Sales", "JP", "num", false],
        ["Other_Sales", "Other", "num", false], ["Global_Sales", "Global", "num", false]
    ];
    GS.factTable = function () {
        var app = $("factApp"); if (!app) return;
        var state = { page: 1 }, timer = null;
        var els = {
            q: $("fq"), platform: $("fplatform"), genre: $("fgenre"), publisher: $("fpublisher"), year: $("fyear"),
            per: $("fper"), prev: $("fprev"), next: $("fnext"), info: $("factInfo"), pg: $("fpage"),
            table: $("factTable"), fk: $("showFk"), reset: $("freset")
        };
        function head() {
            var showFk = els.fk.checked;
            els.table.querySelector("thead").innerHTML = "<tr>" + FACT_COLS.filter(function (c) { return showFk || !c[3]; })
                .map(function (c) { return '<th class="' + (c[2] === "num" ? "num" : "") + '">' + esc(c[1]) + "</th>"; }).join("") + "</tr>";
        }
        function load() {
            var p = new URLSearchParams({
                page: state.page, per_page: els.per.value, q: els.q.value.trim(),
                platform: els.platform.value, genre: els.genre.value, publisher: els.publisher.value, year: els.year.value
            });
            els.table.classList.add("loading");
            getJSON(app.getAttribute("data-api") + "?" + p.toString()).then(function (d) {
                state.page = d.page;
                head();
                var showFk = els.fk.checked, cols = FACT_COLS.filter(function (c) { return showFk || !c[3]; });
                els.table.querySelector("tbody").innerHTML = d.rows.length ? d.rows.map(function (r) {
                    return "<tr>" + cols.map(function (c) {
                        var v = r[c[0]];
                        if (c[0] === "Year" && v == null) v = "Unknown";
                        if (c[2] === "num") v = Number(v).toFixed(2);
                        return '<td class="' + c[2] + '">' + esc(v) + "</td>";
                    }).join("") + "</tr>";
                }).join("") : '<tr><td colspan="' + cols.length + '">No matching records.</td></tr>';
                els.info.textContent = d.total ? "Showing " + fmt(d.start) + "-" + fmt(d.end) + " of " + fmt(d.total) + " records" : "0 records";
                els.pg.textContent = "Page " + fmt(d.page) + " of " + fmt(d.pages);
                els.prev.disabled = d.page <= 1;
                els.next.disabled = d.page >= d.pages;
            }).catch(function (e) {
                els.table.querySelector("tbody").innerHTML = '<tr><td>Could not load data: ' + esc(e.message) + "</td></tr>";
            }).then(function () { els.table.classList.remove("loading"); });
        }
        function reset1() { state.page = 1; load(); }
        els.q.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(reset1, 250); });
        [els.platform, els.genre, els.publisher, els.year, els.per].forEach(function (e) { e.addEventListener("change", reset1); });
        els.fk.addEventListener("change", load);
        els.prev.addEventListener("click", function () { state.page--; load(); });
        els.next.addEventListener("click", function () { state.page++; load(); });
        els.reset.addEventListener("click", function () {
            els.q.value = ""; els.platform.value = ""; els.genre.value = ""; els.publisher.value = ""; els.year.value = "";
            reset1();
        });
        load();
    };

    GS.dimPreviews = function () {
        document.querySelectorAll(".dim-preview").forEach(function (box) {
            var name = box.getAttribute("data-table");
            getJSON("/api/warehouse/dimension/" + name + "?limit=10").then(function (d) {
                box.querySelector("[data-total]").textContent = fmt(d.total) + " rows";
                box.querySelector("thead").innerHTML = "<tr>" + d.columns.map(function (c) { return "<th>" + esc(c) + "</th>"; }).join("") + "</tr>";
                box.querySelector("tbody").innerHTML = d.rows.map(function (r) {
                    return "<tr>" + d.columns.map(function (c) { return "<td>" + esc(r[c] == null ? "NULL (unknown)" : r[c]) + "</td>"; }).join("") + "</tr>";
                }).join("");
            }).catch(function (e) { box.querySelector("tbody").innerHTML = "<tr><td>" + esc(e.message) + "</td></tr>"; });
        });
    };

    /* ---------- chart helpers ---------- */
    function barChart(id, labels, values, color, opts) {
        opts = opts || {};
        var c = $(id); if (!c) return null;
        var colors = Array.isArray(color) ? color : labels.map(function () { return color; });
        return new Chart(c, {
            type: "bar",
            data: { labels: labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6, maxBarThickness: 46 }] },
            options: {
                indexAxis: opts.horizontal ? "y" : "x", responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (ctx) { return (opts.prefix || "") + fmt(ctx.parsed[opts.horizontal ? "x" : "y"], opts.dec || 0) + (opts.suffix || ""); } } } },
                scales: { x: { grid: { display: !!opts.horizontal } }, y: { beginAtZero: true, grid: { display: !opts.horizontal } } }
            }
        });
    }

    GS.datasetCharts = function () {
        getJSON("/api/dataset-stats").then(function (d) {
            barChart("chGenre", d.genre_sales.labels, d.genre_sales.values, PURPLE, { dec: 1, suffix: " M" });
            barChart("chPlatform", d.platform_counts.labels, d.platform_counts.values, CYAN, { suffix: " titles" });
            new Chart($("chYear"), {
                type: "line",
                data: { labels: d.year_sales.labels, datasets: [{ label: "Global sales (M)", data: d.year_sales.values, borderColor: CYAN, backgroundColor: "rgba(34,211,238,.14)", fill: true, tension: .3, pointRadius: 2 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }).catch(function (e) { ["chGenre", "chPlatform", "chYear"].forEach(function (i) { showMsg(i, e.message); }); });
    };

    function scatterChart(id, series, maxAxis, zoomBtnId) {
        var c = $(id); if (!c) return null;
        var zoomMax = Math.max(1, quantile(series[0].actual, 0.99));
        var ideal = { type: "line", label: "Perfect prediction", data: [{ x: 0, y: 0 }, { x: maxAxis, y: maxAxis }], borderColor: "rgba(255,255,255,.55)", borderDash: [6, 5], pointRadius: 0, borderWidth: 1.2 };
        var datasets = series.map(function (s) {
            return {
                type: "scatter", label: s.label, backgroundColor: s.color + "99", borderColor: s.color, pointRadius: 2.2,
                data: s.actual.map(function (a, i) { return { x: a, y: s.pred[i] }; })
            };
        });
        var chart = new Chart(c, {
            data: { datasets: datasets.concat([ideal]) },
            options: {
                animation: false, responsive: true, maintainAspectRatio: false, parsing: false,
                plugins: { legend: { display: series.length > 1 || true }, tooltip: { callbacks: { label: function (ctx) { return "actual " + ctx.parsed.x.toFixed(2) + " → predicted " + ctx.parsed.y.toFixed(2); } } } },
                scales: {
                    x: { type: "linear", min: 0, max: zoomMax, title: { display: true, text: "Actual Global Sales (M)" } },
                    y: { type: "linear", min: undefined, suggestedMax: zoomMax, title: { display: true, text: "Predicted Global Sales (M)" } }
                }
            }
        });
        var zoomed = true;
        if (zoomBtnId && $(zoomBtnId)) {
            $(zoomBtnId).textContent = "Show full range";
            $(zoomBtnId).addEventListener("click", function () {
                zoomed = !zoomed;
                chart.options.scales.x.max = zoomed ? zoomMax : maxAxis;
                chart.options.scales.y.suggestedMax = zoomed ? zoomMax : maxAxis;
                chart.update();
                this.textContent = zoomed ? "Show full range" : "Zoom to 99th percentile";
            });
        }
        return chart;
    }
    function quantile(arr, q) {
        var s = arr.slice().sort(function (a, b) { return a - b; });
        return s[Math.min(s.length - 1, Math.floor(q * s.length))];
    }

    GS.modelCharts = function (kind) {
        var isLin = kind === "linear";
        var ids = isLin ? { sc: "linScatter", rs: "linResid", note: "linResidNote", z: "linZoom" } : { sc: "rfScatter", rs: "rfResid", note: "rfResidNote", z: "rfZoom" };
        var color = isLin ? CYAN : PURPLE;
        getJSON("/api/charts/" + kind).then(function (d) {
            scatterChart(ids.sc, [{ label: isLin ? "Linear Regression" : "Random Forest", color: color, actual: d.actual, pred: d.predicted }], d.max_axis, ids.z);
            var h = d.residual_hist;
            barChart(ids.rs, h.labels, h.counts, color, { suffix: " games" });
            if ($(ids.note)) $(ids.note).textContent = "Histogram of residuals for the central 98% of test games (" + fmt(h.shown) + " of " + fmt(h.total) + "), bin centres in million units. A good model centres on 0 with a narrow spread.";
            if (!isLin && d.feature_importance && d.feature_importance.length) {
                var fi = d.feature_importance;
                barChart("rfImportance", fi.map(function (f) { return f.feature; }), fi.map(function (f) { return f.importance; }), PURPLE, { horizontal: true, dec: 4 });
                var ch = Chart.getChart("rfImportance"); if (ch) ch.options.scales.y.ticks = { autoSkip: false }, ch.options.scales.y.reverse = false, ch.update();
            }
        }).catch(function (e) { [ids.sc, ids.rs, "rfImportance"].forEach(function (i) { showMsg(i, e.message); }); });
    };

    GS.comparisonCharts = function () {
        getJSON("/api/charts/comparison").then(function (d) {
            var rows = {}; d.comparison.rows.forEach(function (r) { rows[r.key] = r; });
            var names = ["Linear Regression", "Random Forest"];
            [["MAE", "cMAE", 4], ["RMSE", "cRMSE", 4], ["R2", "cR2", 4], ["MAPE", "cMAPE", 1]].forEach(function (m) {
                var r = rows[m[0]]; if (!r) return;
                var cols = [CYAN, PURPLE].map(function (c, i) { return (i === 0 && r.winner === "linear") || (i === 1 && r.winner === "random_forest") ? c : c + "66"; });
                barChart(m[1], names, [r.linear, r.random_forest], cols, { dec: m[2], suffix: m[0] === "MAPE" ? " %" : "" });
            });
            var lin = d.linear, rf = d.random_forest;
            scatterChart("cScatter", [
                { label: "Linear Regression", color: CYAN, actual: lin.actual, pred: lin.predicted },
                { label: "Random Forest", color: PURPLE, actual: rf.actual, pred: rf.predicted }
            ], Math.max(lin.max_axis, rf.max_axis), "cmpZoom");
            var f = d.first_n, idx = f.actual.map(function (_, i) { return i + 1; });
            new Chart($("cLine"), {
                type: "line",
                data: { labels: idx, datasets: [
                    { label: "Actual", data: f.actual, borderColor: "#e6edff", backgroundColor: "#e6edff", tension: .15, pointRadius: 2.5, borderWidth: 2 },
                    { label: "Linear Regression", data: f.linear, borderColor: CYAN, backgroundColor: CYAN, tension: .15, pointRadius: 2, borderWidth: 1.5 },
                    { label: "Random Forest", data: f.random_forest, borderColor: PURPLE, backgroundColor: PURPLE, tension: .15, pointRadius: 2, borderWidth: 1.5 }
                ] },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: "Test game #" } }, y: { title: { display: true, text: "Global Sales (M)" } } } }
            });
        }).catch(function (e) { ["cMAE", "cRMSE", "cR2", "cMAPE", "cScatter", "cLine"].forEach(function (i) { showMsg(i, e.message); }); });
    };

    GS.resultChart = function (lin, rf) {
        var c = $("resultChart"); if (!c) return;
        barChart("resultChart", ["Linear Regression", "Random Forest"], [lin, rf], [CYAN, PURPLE], { dec: 3, suffix: " M units" });
    };

    initPipelines();
})();
