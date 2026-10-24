/* Data Science Coach – Interaktionsschicht.
 *
 * Eine Datei, reines Vanilla-JS, keine Abhängigkeiten. Sie wird von jeder
 * Seite geladen (build.py setzt <script defer src=".../app.js">). Jede Funktion
 * prüft selbst, ob ihre Elemente existieren, damit dasselbe Skript auf
 * Start-, Lektions-, Übungs- und Code-Seiten läuft.
 *
 * Persistente Einstellungen (localStorage):
 *   dsc-theme  Farbschema (light|dark|system)   dsc-lang  Codesprache (py|r|beide)
 *   dsc-level  Tiefe (grund|alles)    dsc-done   erledigte Module (JSON-Array)
 */
(function () {
  "use strict";
  var root = document.documentElement, doc = document;
  var NAV_BREAK = 860;
  var store = {
    get: function (k, d) { try { var v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
    getJSON: function (k, d) { try { return JSON.parse(localStorage.getItem(k)) || d; } catch (e) { return d; } },
    setJSON: function (k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} }
  };
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function ready(fn) { if (doc.readyState !== "loading") fn(); else doc.addEventListener("DOMContentLoaded", fn); }

  /* Nur EIN Topbar-Overlay gleichzeitig. Auf schmalen Viewports deckt die
     Navigations-Schublade die ganze Breite ab; ohne diese Regel blieb sie offen,
     während sich das Einstellungs-Panel dahinter öffnete – der Klick auf
     „Ansicht“ schien das falsche Menü zu öffnen. Beim Öffnen eines Overlays
     werden die anderen geschlossen. */
  function closeTopbarOverlays(keep) {
    if (keep !== "settings") {
      var sb = doc.getElementById("settingsBox"), sbtn = doc.getElementById("settingsBtn");
      if (sb) sb.classList.remove("open");
      if (sbtn) sbtn.setAttribute("aria-expanded", "false");
    }
    if (keep !== "nav") {
      var bar = doc.querySelector(".topbar"), nbtn = doc.getElementById("navBurger");
      if (bar) bar.classList.remove("nav-open");
      if (nbtn) { nbtn.setAttribute("aria-expanded", "false"); nbtn.setAttribute("aria-label", "Menü öffnen"); }
    }
    if (keep !== "modules") {
      var nm = doc.getElementById("navModules"), mbtn = doc.getElementById("navModulesBtn");
      if (nm) nm.classList.remove("nm-open");
      if (mbtn) mbtn.setAttribute("aria-expanded", "false");
    }
  }

  /* ---------- Mobile-Navigation (Burger) -------------------------------- *
   * Auf schmalen Viewports klappt die Linkliste unter der Leiste auf. Der
   * Burger schaltet .nav-open auf der Topbar; Klick auf einen Link, Escape,
   * ein Klick außerhalb oder das Verbreitern des Fensters schließen wieder.  */
  function initNav() {
    var bar = doc.querySelector(".topbar"), btn = doc.getElementById("navBurger"),
        nav = doc.getElementById("topnav");
    if (!bar || !btn || !nav) return;
    function set(open) {
      if (open) closeTopbarOverlays("nav");
      bar.classList.toggle("nav-open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      btn.setAttribute("aria-label", open ? "Menü schließen" : "Menü öffnen");
    }
    btn.addEventListener("click", function (e) { e.stopPropagation(); set(!bar.classList.contains("nav-open")); });
    nav.addEventListener("click", function (e) { if (e.target.closest("a")) set(false); });
    doc.addEventListener("click", function (e) {
      if (bar.classList.contains("nav-open") && !bar.contains(e.target)) set(false);
    });
    addEventListener("keydown", function (e) { if (e.key === "Escape") set(false); });
    addEventListener("resize", function () { if (innerWidth > NAV_BREAK) set(false); });
  }

  /* ---------- „Module ▾“-Dropdown (Desktop) ----------------------------- *
   * Kursweites Sprungmenü. Auf dem Desktop klappt es per Klick auf; Klick auf
   * einen Link, außerhalb, Escape oder Fensterbreite schließen. Auf schmalen
   * Viewports ist es dauerhaft in der Burger-Schublade sichtbar (CSS), daher
   * bleibt der Toggle dort wirkungslos.                                      */
  function initModuleMenu() {
    var box = doc.getElementById("navModules"), btn = doc.getElementById("navModulesBtn");
    if (!box || !btn) return;
    function set(open) {
      if (open) closeTopbarOverlays("modules");
      box.classList.toggle("nm-open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    }
    btn.addEventListener("click", function (e) { e.stopPropagation(); set(!box.classList.contains("nm-open")); });
    doc.addEventListener("click", function (e) {
      if (box.classList.contains("nm-open") && !box.contains(e.target)) set(false);
    });
    addEventListener("keydown", function (e) { if (e.key === "Escape") set(false); });
    addEventListener("resize", function () { set(false); });
  }

  /* ---------- Modulgruppen im mobilen Menü ------------------------------ */
  function initModuleGroups() {
    var groups = Array.prototype.slice.call(doc.querySelectorAll(".nm-group"));
    if (!groups.length) return;
    function sync() {
      if (innerWidth > NAV_BREAK) {
        groups.forEach(function (g) { g.open = true; });
        root.removeAttribute("data-nm-mobile-ready");
        return;
      }
      if (root.getAttribute("data-nm-mobile-ready") === "1") return;
      var hasCurrent = groups.some(function (g) { return g.getAttribute("data-current") === "true"; });
      groups.forEach(function (g, i) {
        g.open = g.getAttribute("data-current") === "true" || (!hasCurrent && i === 0);
      });
      root.setAttribute("data-nm-mobile-ready", "1");
    }
    sync();
    addEventListener("resize", sync);
  }

  /* ---------- Kursweite Einstellungen ---------------------------------- */
  function initSettings() {
    var box = doc.getElementById("settingsBox"), btn = doc.getElementById("settingsBtn");
    if (!box || !btn) return;
    function set(open) {
      if (open) closeTopbarOverlays("settings");
      box.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    }
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      set(!box.classList.contains("open"));
    });
    box.addEventListener("click", function (e) { e.stopPropagation(); });
    doc.addEventListener("click", function () { set(false); });
    addEventListener("keydown", function (e) { if (e.key === "Escape") set(false); });
  }

  /* ---------- Voraussetzungs-Hinweis: einmal ausblenden ----------------- */
  function initPrereq() {
    var box = doc.querySelector(".prereq[data-prereq]");
    if (!box) return;
    if (store.get("dsc-prereq", "") === "hidden") { box.hidden = true; return; }
    var x = box.querySelector(".pq-close");
    if (x) x.addEventListener("click", function () { box.hidden = true; store.set("dsc-prereq", "hidden"); });
  }

  /* ---------- Segmentierte Umschalter (Sprache, Niveau, Farbschema) ------ *
   * Ein generischer Handler für alle .seg-Leisten. data-target setzt das
   * Attribut auf <html> (z. B. data-lang), data-store persistiert es. Das
   * Farbschema ist eine dieser Leisten und braucht keinen eigenen Knopf.    */
  function initSegments() {
    doc.querySelectorAll(".seg").forEach(function (seg) {
      var target = seg.getAttribute("data-target");   // "lang" | "level"
      var key = seg.getAttribute("data-store");
      var def = seg.getAttribute("data-default") || "";
      var cur = store.get(key, def);
      apply(cur);
      seg.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () { apply(b.getAttribute("data-val")); store.set(key, b.getAttribute("data-val")); });
      });
      function apply(val) {
        // "system" heißt: gar keine Vorgabe. Nur ohne data-theme greift die
        // prefers-color-scheme-Regel im Stylesheet – ein data-theme="system"
        // würde auf keinen der beiden Selektoren passen und die Seite bei
        // dunkler Systemeinstellung stumm hell lassen.
        if (val === "system") root.removeAttribute("data-" + target);
        else root.setAttribute("data-" + target, val);
        seg.querySelectorAll("button").forEach(function (b) {
          b.classList.toggle("on", b.getAttribute("data-val") === val);
          b.setAttribute("aria-pressed", b.getAttribute("data-val") === val);
        });
        if (target === "level") syncTracks(val);
      }
    });
    // Anfangszustand auch dann setzen, wenn keine Leiste sichtbar ist.
    if (!root.hasAttribute("data-lang")) root.setAttribute("data-lang", store.get("dsc-lang", "py"));
    if (!root.hasAttribute("data-level")) root.setAttribute("data-level", store.get("dsc-level", "grund"));
    syncTracks(root.getAttribute("data-level"));
  }
  // Zwei symmetrische Vertiefungsspuren (beide als <details>):
  //   .adv   „Für Fortgeschrittene“ → im Grundlagen-Modus zu, sonst offen.
  //   .basis „Grundlage“ (Einsteiger) → im Alles-Modus zu, sonst offen.
  function syncTracks(level) {
    doc.querySelectorAll("details.callout.adv").forEach(function (d) {
      if (!d.dataset.touched) d.open = level !== "grund";
    });
    doc.querySelectorAll("details.callout.basis").forEach(function (d) {
      if (!d.dataset.touched) d.open = level !== "alles";
    });
    // Ganze Abschnittsgruppen (z. B. Modul 14, §7–§11). Der Tiefen-Umschalter
    // konnte bisher nur einzelne Callouts einklappen, nie eine ganze
    // Vertiefung – 90 Minuten Stoff blieben im Grundlagen-Modus sichtbar.
    doc.querySelectorAll("details.vertiefung").forEach(function (d) {
      if (!d.dataset.touched) d.open = level !== "grund";
    });
  }

  /* ---------- Selbstcheck: aktive Erinnerung ---------------------------- */
  /* Selbstcheck-Karten: der Button trug weder aria-expanded noch aria-controls,
   * die Antwort war für Screenreader auch im eingeklappten Zustand sichtbar. */
  var quizSeq = 0;
  function initQuiz() {
    doc.querySelectorAll(".quiz-card").forEach(function (card) {
      var q = card.querySelector(".quiz-q");
      var a = card.querySelector(".quiz-a");
      if (!q) return;

      if (a && !a.id) a.id = "quiz-a-" + (++quizSeq);
      q.setAttribute("aria-expanded", "false");
      if (a) {
        q.setAttribute("aria-controls", a.id);
        a.setAttribute("aria-hidden", "true");
      }

      q.addEventListener("click", function () {
        var open = card.classList.toggle("show");
        q.setAttribute("aria-expanded", open ? "true" : "false");
        if (a) a.setAttribute("aria-hidden", open ? "false" : "true");
        var t = card.querySelector(".quiz-toggle");
        if (t) t.textContent = open ? "Antwort verbergen" : "Antwort zeigen";
      });
    });
  }

  /* ---------- Fortschritt: Modul als erledigt --------------------------- */
  function totalModules() {
    return doc.querySelectorAll("#navModulesPanel a[data-slug]").length ||
      doc.querySelectorAll(".karte[data-slug]").length || 34;
  }
  function doneSet() { return store.getJSON("dsc-done", []); }
  function initProgress() {
    var done = doneSet();
    // „Erledigt“-Umschalter in der Kontrollzeile jeder Lektion
    doc.querySelectorAll(".done-toggle").forEach(function (btn) {
      var slug = btn.getAttribute("data-slug");
      setToggle(btn, done.indexOf(slug) !== -1);
      btn.addEventListener("click", function () {
        var d = doneSet(), i = d.indexOf(slug), now = i === -1;
        if (now) d.push(slug); else d.splice(i, 1);
        store.setJSON("dsc-done", d); setToggle(btn, now);
        paintPill(d); paintCards(d); paintRing(d); paintContinue(d);
      });
    });
    paintPill(done); paintCards(done); paintRing(done); paintContinue(done);
  }
  function setToggle(btn, on) {
    btn.classList.toggle("on", on);
    btn.setAttribute("aria-pressed", on);
    var l = btn.querySelector(".dt-label");
    if (l) l.textContent = on ? "Erledigt" : "Als erledigt markieren";
  }
  function getPathCounts(done, key) {
    var seq = PFADE[key];
    if (!seq) {
      return { done: done.length, total: totalModules() };
    }
    var dCount = 0;
    done.forEach(function (slug) {
      var num = slug.split("-")[0];
      if (seq.indexOf(num) !== -1) {
        dCount++;
      }
    });
    return { done: dCount, total: seq.length };
  }
  function paintPill(done) {
    var key = store.get("dsc-path", "all");
    var counts = getPathCounts(done, key);
    var badge = doc.getElementById("nmBtnProgress");
    if (badge) {
      badge.textContent = counts.done + " / " + counts.total;
    }

    doc.querySelectorAll('#navModulesPanel a[data-slug]').forEach(function (link) {
      var slug = link.getAttribute('data-slug');
      var status = link.querySelector('.nm-status');
      if (status) {
        var isDone = done.indexOf(slug) !== -1;
        status.classList.toggle('done', isDone);
      }
    });
  }
  function paintCards(done) {
    doc.querySelectorAll(".karte[data-slug]").forEach(function (k) {
      k.classList.toggle("done", done.indexOf(k.getAttribute("data-slug")) !== -1);
    });
  }
  function paintRing(done) {
    var ring = doc.getElementById("ringFill"), lab = doc.getElementById("ringLabel");
    if (!ring) return;
    var key = store.get("dsc-path", "all");
    var counts = getPathCounts(done, key);
    var frac = Math.min(counts.done / (counts.total || 1), 1), C = 2 * Math.PI * 26;
    ring.style.strokeDasharray = C;
    ring.style.strokeDashoffset = reduce ? C * (1 - frac) : ring.style.strokeDashoffset || C;
    if (!reduce) requestAnimationFrame(function () { ring.style.strokeDashoffset = C * (1 - frac); });
    else ring.style.strokeDashoffset = C * (1 - frac);
    if (lab) lab.textContent = counts.done + "/" + counts.total;
  }

  /* ---------- Lernpfad-Auswahl (Startseite) ----------------------------- *
   * PFADE_ALIAS migriert alte localStorage-Werte (vor einer Umbenennung) auf
   * die aktuellen Schlüssel – jede Pfad-Modulliste existiert nur einmal.     */
  var PFADE = {
    all: null,
    basis: ["02", "03", "04", "06", "08", "09", "10"],
    daten: ["02", "03", "04", "05", "06", "07"],
    paper: ["02", "03", "04", "06", "08", "09", "10", "12", "22"],
    forschung: ["06", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"],
    statistik: ["09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    ki: ["23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34"]
  };
  // Migriert alte localStorage-Schlüssel. `forschung` ist seit der Ergänzung
  // des vollständigen Publikationspfads ein eigener Pfad und darf NICHT mehr
  // auf `paper` (den kurzen Shortcut) zeigen – das hat ihn stillschweigend
  // gegen die 7-Schritt-Abkürzung getauscht.
  var PFADE_ALIAS = { schnell: "basis", extraktion: "daten", ml: "ki" };
  function initPaths() {
    var cur = store.get("dsc-path", "all");
    if (PFADE_ALIAS[cur]) {
      cur = PFADE_ALIAS[cur];
      store.set("dsc-path", cur);
    }
    if (!Object.prototype.hasOwnProperty.call(PFADE, cur)) cur = "all";
    applyPathGlobal(cur);
    if (cur !== "all") {
      var details = doc.getElementById("pfadDetails");
      if (details) details.setAttribute("open", "");
    }
    var picker = doc.getElementById("pfadPicker");
    if (picker) {
      picker.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () {
          var key = b.getAttribute("data-path");
          store.set("dsc-path", key);
          applyPathGlobal(key);
          var d = doneSet();
          paintPill(d);
          paintRing(d);
          paintContinue(d);
        });
      });
    }
  }

  function applyPathGlobal(key) {
    var seq = PFADE[key];
    doc.querySelectorAll(".karte[data-num]").forEach(function (k) {
      var num = k.getAttribute("data-num"), badge = k.querySelector(".k-step");
      if (!seq) { k.classList.remove("dim", "on-path"); if (badge) badge.textContent = ""; return; }
      var idx = seq.indexOf(num);
      k.classList.toggle("on-path", idx !== -1);
      k.classList.toggle("dim", idx === -1);
      if (badge) badge.textContent = idx !== -1 ? "Schritt " + (idx + 1) : "";
    });
    var picker = doc.getElementById("pfadPicker");
    if (picker) {
      picker.querySelectorAll("button").forEach(function (b) {
        var on = b.getAttribute("data-path") === key;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
    }
    doc.querySelectorAll(".nm-panel a[data-num]").forEach(function (a) {
      var num = a.getAttribute("data-num");
      if (!seq) {
        a.classList.remove("dim", "on-path");
        var st = a.querySelector(".nm-step");
        if (st) st.remove();
        return;
      }
      var idx = seq.indexOf(num);
      a.classList.toggle("on-path", idx !== -1);
      a.classList.toggle("dim", idx === -1);
      var st = a.querySelector(".nm-step");
      if (idx !== -1) {
        if (!st) {
          st = doc.createElement("span");
          st.className = "nm-step";
          a.appendChild(st);
        }
        st.textContent = "S" + (idx + 1);
      } else {
        if (st) st.remove();
      }
    });
  }

  /* ---------- Fortsetzen-Link auf der Startseite ------------------------ */
  function initContinue() {
    var current = doc.querySelector(".done-toggle[data-slug]");
    if (current) store.set("dsc-last", current.getAttribute("data-slug"));
    paintContinue(doneSet());
  }
  /* Der Fortsetzen-Link folgt dem GEWÄHLTEN Lernpfad. Vorher lief er stur die
   * DOM-Reihenfolge aller Module ab: wer den KI-Pfad wählte, sah die Zählung
   * „0 / 12“ und wurde trotzdem nach Modul 01 geschickt. */
  function pathCards(key) {
    var cards = Array.prototype.slice.call(doc.querySelectorAll(".karte[data-slug]"));
    var seq = PFADE[key];
    if (!seq) return cards;
    // In Pfad-Reihenfolge sortieren, nicht in DOM-Reihenfolge.
    var byNum = {};
    cards.forEach(function (c) { byNum[c.getAttribute("data-num")] = c; });
    var ordered = [];
    seq.forEach(function (num) { if (byNum[num]) ordered.push(byNum[num]); });
    return ordered.length ? ordered : cards;
  }

  function paintContinue(done) {
    var link = doc.getElementById("continueLink");
    if (!link) return;
    var key = store.get("dsc-path", "all");
    var cards = pathCards(key);
    if (!cards.length) return;

    var last = store.get("dsc-last", "");
    var onPath = function (c) { return cards.indexOf(c) !== -1; };
    var lastCard = cards.find(function (c) {
      return c.getAttribute("data-slug") === last && done.indexOf(last) === -1;
    });
    var target = (lastCard && onPath(lastCard) ? lastCard : null) || cards.find(function (c) {
      return done.indexOf(c.getAttribute("data-slug")) === -1;
    }) || cards[0];

    var counts = getPathCounts(done, key);
    var slug = target.getAttribute("data-slug"), title = target.querySelector(".k-titel");
    link.href = target.getAttribute("href");
    link.textContent = counts.done >= counts.total ? "Pfad geschafft · Übersicht öffnen" :
      "Weiter mit " + slug.slice(0, 2) + " · " + (title ? title.textContent : "Modul");
  }

  /* ---------- Lange Codeblöcke einklappen ------------------------------- *
   * Blöcke über LIMIT Zeilen werden gekürzt gezeigt (Fade + „Ganzen Code
   * zeigen"), damit Boilerplate (Pfad-Setup, Verbindungen …) nicht erschlägt.
   * Nimmt optional ein root-Element (Standard: ganzes Dokument), damit
   * dieselbe Logik auch auf nachgeladene Ressourcen-Fragmente angewendet
   * werden kann (initResourcesOverlay). */
  function initCodeClamp(root) {
    var LIMIT = 16;
    (root || doc).querySelectorAll(".code-wrap").forEach(function (w) {
      if (w.dataset.clampBound) return;
      w.dataset.clampBound = "1";
      var pre = w.querySelector("pre");
      if (!pre) return;
      var lines = (pre.textContent.match(/\n/g) || []).length + 1;
      if (lines <= LIMIT) return;
      w.classList.add("clamped");
      var btn = doc.createElement("button");
      btn.className = "code-more"; btn.type = "button";
      var shut = "Ganzen Code zeigen · " + lines + " Zeilen";
      btn.textContent = shut;
      btn.addEventListener("click", function () {
        var open = w.classList.toggle("open");
        btn.textContent = open ? "Weniger zeigen" : shut;
      });
      w.appendChild(btn);
    });
  }

  /* ---------- „Wo eingeben?“ – Hinweis je Codeblock --------------------- */
  function initWhere(root) {
    (root || doc).querySelectorAll(".wo-btn").forEach(function (btn) {
      if (btn.dataset.bound) return;
      btn.dataset.bound = "1";
      var wrap = btn.closest(".code-wrap"), hint = wrap ? wrap.querySelector(".wo-hint") : null;
      if (!hint) return;
      btn.addEventListener("click", function () {
        var open = hint.hidden;
        hint.hidden = !open;
        btn.setAttribute("aria-expanded", open ? "true" : "false");
      });
    });
  }

  /* ---------- Code kopieren --------------------------------------------- */
  function initCopy(root) {
    (root || doc).querySelectorAll(".code-wrap").forEach(function (w) {
      var b = w.querySelector(".kopieren"), pre = w.querySelector("pre");
      if (b && pre && !b.dataset.bound) {
        b.dataset.bound = "1";
        b.addEventListener("click", function () {
          navigator.clipboard.writeText(pre.innerText).then(function () {
            b.textContent = "Kopiert"; setTimeout(function () { b.textContent = "Kopieren"; }, 1200);
          }).catch(function () {
            b.textContent = "Fehler – manuell kopieren";
            setTimeout(function () { b.textContent = "Kopieren"; }, 1800);
          });
        });
      }
    });
  }

  /* ---------- Daten-URL kopieren (Download-Panel) ------------------------ */
  function initDataCopy(root) {
    (root || doc).querySelectorAll(".dl-copy").forEach(function (b) {
      if (b.dataset.bound) return;
      b.dataset.bound = "1";
      var url = b.getAttribute("data-url"), shut = b.textContent;
      b.addEventListener("click", function () {
        navigator.clipboard.writeText(url).then(function () {
          b.textContent = "Kopiert"; setTimeout(function () { b.textContent = shut; }, 1200);
        }).catch(function () {
          b.textContent = "Fehler – manuell kopieren";
          setTimeout(function () { b.textContent = shut; }, 1800);
        });
      });
    });
  }

  /* ---------- Ankerlinks an Überschriften ------------------------------- */
  function initAnchors() {
    var body = doc.getElementById("body");
    (body ? body.querySelectorAll("h2,h3") : []).forEach(function (h) {
      if (!h.id) return;
      var a = doc.createElement("a"); a.className = "anchor-link"; a.href = "#" + h.id;
      a.textContent = "#"; a.setAttribute("aria-hidden", "true"); h.insertBefore(a, h.firstChild);
    });
  }

  /* ---------- TOC-Scrollspy --------------------------------------------- */
  function initSpy() {
    var links = Array.prototype.slice.call(doc.querySelectorAll("#tocList a"));
    if (!links.length || !("IntersectionObserver" in window)) return;
    var heads = links.map(function (a) { return doc.getElementById(a.getAttribute("href").slice(1)); }).filter(Boolean);
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) links.forEach(function (a) {
          a.classList.toggle("active", a.getAttribute("href").slice(1) === e.target.id);
        });
      });
    }, { rootMargin: "-72px 0px -70% 0px" });
    heads.forEach(function (h) { io.observe(h); });
  }

  /* ---------- Fortschrittsbalken, Topbar, FAB --------------------------- */
  function initScroll() {
    var bar = doc.getElementById("progressBar"), top = doc.querySelector(".topbar"),
        fab = doc.getElementById("topBtn"), title = doc.querySelector(".titleblock");
    function onScroll() {
      var st = window.scrollY, h = doc.documentElement.scrollHeight - window.innerHeight;
      if (bar) bar.style.width = (h > 0 ? (st / h) * 100 : 0) + "%";
      if (top) top.classList.toggle("scrolled", st > 4);
      if (fab) fab.classList.toggle("show", st > 600);
    }
    addEventListener("scroll", function () { requestAnimationFrame(onScroll); }, { passive: true });
    onScroll();
    if (fab) fab.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" }); });
  }

  /* ---------- Lightbox für Abbildungen ---------------------------------- */
  /* Vorher war die Lightbox reine Maus-Interaktion: ein click-Handler auf einem
   * nackten <img> ohne tabindex, ohne Rolle, ohne Fokusverwaltung. Mit Tastatur
   * oder Screenreader liess sich keine Abbildung vergrößern (WCAG 2.1.1), und
   * beim Öffnen blieb der Fokus hinter dem Overlay (WCAG 2.4.3). */
  function initLightbox() {
    var figs = doc.querySelectorAll("figure img");
    if (!figs.length) return;

    var lb = doc.createElement("div");
    lb.className = "lightbox";
    lb.setAttribute("role", "dialog");
    lb.setAttribute("aria-modal", "true");
    lb.setAttribute("aria-label", "Abbildung vergrößert");
    lb.hidden = true;

    var stage = doc.createElement("div"); stage.className = "lb-stage";
    var img = doc.createElement("img");
    img.alt = "";   // Standard, bis open() das alt der geklickten Abbildung setzt
    var cap = doc.createElement("div"); cap.className = "lb-caption";
    var codeLink = doc.createElement("a"); codeLink.className = "lb-code";
    codeLink.innerHTML = '<span aria-hidden="true">&lt;/&gt;</span> Code ansehen';
    codeLink.addEventListener("click", function (e) { e.stopPropagation(); });

    var closeBtn = doc.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "lb-close";
    closeBtn.setAttribute("aria-label", "Abbildung schließen");
    closeBtn.innerHTML = '<span aria-hidden="true">×</span>';

    stage.appendChild(closeBtn); stage.appendChild(img);
    stage.appendChild(cap); stage.appendChild(codeLink);
    lb.appendChild(stage); doc.body.appendChild(lb);

    var lastFocus = null;

    function lbFocusable() {
      return Array.prototype.slice.call(
        lb.querySelectorAll('button, a[href]')
      ).filter(function (el) { return el.offsetParent !== null; });
    }
    function open(im) {
      var fig = im.closest("figure"), fc = fig ? fig.querySelector("figcaption") : null;
      var codeUrl = fig ? fig.getAttribute("data-code") : null;
      var capText = fc ? fc.textContent : "";
      if (codeUrl) capText = capText.replace(/\s*·\s*Code ansehen\s*$/, "");
      img.src = im.currentSrc || im.src;
      img.alt = im.alt || "";
      cap.textContent = capText;
      cap.style.display = capText ? "" : "none";
      if (codeUrl) { codeLink.href = codeUrl; codeLink.style.display = ""; }
      else { codeLink.style.display = "none"; }

      lastFocus = doc.activeElement;
      lb.hidden = false;
      doc.body.style.overflow = "hidden";
      lb.classList.add("show");
      closeBtn.focus();
    }
    function close() {
      lb.classList.remove("show");
      lb.hidden = true;
      doc.body.style.overflow = "";
      if (lastFocus && doc.contains(lastFocus) && typeof lastFocus.focus === "function") {
        lastFocus.focus();
      }
    }

    figs.forEach(function (im) {
      // Das Bild selbst wird zum Button: fokussierbar, mit Rolle und Namen.
      im.setAttribute("tabindex", "0");
      im.setAttribute("role", "button");
      im.setAttribute("aria-haspopup", "dialog");
      var name = im.alt ? im.alt.slice(0, 60) : "Abbildung";
      im.setAttribute("aria-label", name + " – vergrößern");
      im.addEventListener("click", function () { open(im); });
      im.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
          e.preventDefault();
          open(im);
        }
      });
    });

    closeBtn.addEventListener("click", function (e) { e.stopPropagation(); close(); });
    lb.addEventListener("click", close);
    stage.addEventListener("click", function (e) { e.stopPropagation(); });

    lb.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { close(); return; }
      if (e.key !== "Tab") return;
      var items = lbFocusable();
      if (!items.length) return;
      var first = items[0], last = items[items.length - 1];
      if (e.shiftKey && doc.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && doc.activeElement === last) { e.preventDefault(); first.focus(); }
    });
  }

  /* ---------- Interaktive Checklisten ----------------------------------- *
   * Nimmt optional ein root-Element, damit dieselbe Logik auch auf ein per
   * fetch() nachgeladenes Ressourcen-Fragment angewendet werden kann. Der
   * Index ist innerhalb des jeweiligen Dokuments 0-basiert und damit über
   * Standalone-Seite und nachgeladenes Fragment hinweg identisch – der
   * localStorage-Zustand bleibt so zwischen beiden Ansichten synchron. */
  function initChecklist(root) {
    var scope = root || doc;
    var checkboxes = scope.querySelectorAll('input.task-list-item-checkbox');
    if (checkboxes.length === 0) return;

    checkboxes.forEach(function (cb, index) {
      if (cb.dataset.bound) return;
      cb.dataset.bound = "1";
      var key = "dsc-check-global-" + index;
      cb.checked = store.get(key, false) === true;
      cb.addEventListener("change", function () {
        store.set(key, cb.checked);
        // Sync checkboxes with the same index within the same scope (e.g. a
        // standalone reference page that also renders the drawer excerpt).
        scope.querySelectorAll('input.task-list-item-checkbox').forEach(function (other) {
          if (other !== cb && other.getAttribute("data-index") === cb.getAttribute("data-index")) {
            other.checked = cb.checked;
          }
        });
      });
      cb.setAttribute("data-index", index);
    });
  }

  /* ---------- Ressourcen Overlay (Sheet Drawer) -------------------------- *
   * Die sechs Referenzdokumente werden NICHT mehr pro Seite dupliziert. Beim
   * Öffnen einer Kachel wird die zugehörige eigenständige Seite (RESOURCE_PAGES
   * in build.py) per fetch() nachgeladen, ihr #body-Inhalt extrahiert, relative
   * Links/Bilder gegen die geladene URL aufgelöst und einmal pro Seitenaufruf
   * zwischengespeichert. Schlägt fetch() fehl (z. B. file://-Aufruf ohne CORS),
   * navigiert der Klick stattdessen ganz normal zur eigenständigen Seite –
   * die Kacheln sind deshalb echte <a href>-Links, kein reines JS-Konstrukt. */
  function initResourcesOverlay() {
    var btn = doc.getElementById("resourcesBtn");
    var overlay = doc.getElementById("resourcesOverlay");
    var sheet = overlay ? overlay.querySelector(".ro-sheet") : null;
    var closeBtn = doc.getElementById("resourcesClose");
    var backdrop = doc.getElementById("resourcesBackdrop");
    var backBtn = doc.getElementById("resourcesBack");
    var title = doc.getElementById("roTitle");
    var menu = doc.getElementById("roMenu");
    var content = doc.getElementById("roContent");
    var detail = doc.getElementById("roDetail");

    if (!btn || !overlay || !sheet) return;

    var titles = {
      daten: "Datenübersicht",
      projektkompass: "Projektkompass",
      spickzettel: "Spickzettel",
      checkliste: "Publikations-Checkliste",
      glossar: "Klinisches Glossar",
      rechtliches: "Rechtliches und Datenschutz"
    };
    var cache = {};      // key -> roher (nicht angebundener) Fragment-HTML
    var lastFocus = null;
    var lastClickedKey = null; // Behebt Focus-Trap-Verlust: speichert die geöffnete Kachel

    function focusable(container) {
      return Array.prototype.slice.call(container.querySelectorAll(
        'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )).filter(function (el) { return el.offsetParent !== null; });
    }
    function trapKeydown(e) {
      if (e.key !== "Tab") return;
      var items = focusable(sheet);
      if (!items.length) return;
      var first = items[0], last = items[items.length - 1];
      if (e.shiftKey && doc.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && doc.activeElement === last) { e.preventDefault(); first.focus(); }
    }

    function open() {
      var settings = doc.getElementById("settingsBox"), settingsBtn = doc.getElementById("settingsBtn");
      if (settings) settings.classList.remove("open");
      if (settingsBtn) settingsBtn.setAttribute("aria-expanded", "false");
      // Der Auslöser sitzt jetzt im Modulmenü; sonst bliebe es hinter der
      // Overlay-Fläche offen stehen.
      var nm = doc.getElementById("navModules"), nmBtn = doc.getElementById("navModulesBtn");
      if (nm) nm.classList.remove("nm-open");
      if (nmBtn) nmBtn.setAttribute("aria-expanded", "false");
      lastFocus = doc.activeElement;
      overlay.classList.add("open");
      overlay.setAttribute("aria-hidden", "false");
      doc.body.style.overflow = "hidden";
      sheet.addEventListener("keydown", trapKeydown);
      var first = focusable(sheet)[0];
      (first || sheet).focus();
    }

    function close() {
      overlay.classList.remove("open");
      overlay.setAttribute("aria-hidden", "true");
      doc.body.style.overflow = "";
      sheet.removeEventListener("keydown", trapKeydown);
      goBack();
      if (lastFocus && doc.contains(lastFocus) && typeof lastFocus.focus === "function") lastFocus.focus();
      lastFocus = null;
    }

    function resolveUrl(v, base) {
      try { return new URL(v).href; } catch (e) {}
      try { return new URL(v, base).href; } catch (e2) { return v; }
    }
    function absolutize(container, baseUrl) {
      container.querySelectorAll("[href]").forEach(function (el) {
        var v = el.getAttribute("href");
        if (v) el.setAttribute("href", resolveUrl(v, baseUrl));
      });
      container.querySelectorAll("[src]").forEach(function (el) {
        var v = el.getAttribute("src");
        if (v) el.setAttribute("src", resolveUrl(v, baseUrl));
      });
    }
    function enhance(container) {
      initCopy(container); initWhere(container); initCodeClamp(container);
      initChecklist(container); initDataCopy(container);
    }

    function showDetail(key, url) {
      menu.style.display = "none";
      content.style.display = "block";
      backBtn.style.display = "inline-flex";
      title.textContent = titles[key] || "Referenz";
      content.scrollTop = 0;

      // Behebt Focus-Trap-Verlust: Setze Fokus auf den Zurück-Button, sobald Detailbereich aufklappt
      backBtn.focus();

      if (cache[key]) {
        detail.innerHTML = cache[key];
        enhance(detail);
        return;
      }
      detail.innerHTML = '<p class="ro-loading">Wird geladen …</p>';
      fetch(url).then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.text();
      }).then(function (html) {
        var parsed = new DOMParser().parseFromString(html, "text/html");
        var body = parsed.getElementById("body");
        if (!body) throw new Error("Kein Inhalt gefunden");
        absolutize(body, url);
        cache[key] = body.innerHTML;
        detail.innerHTML = cache[key];
        enhance(detail);
      }).catch(function () {
        // fetch nicht möglich (z. B. lokaler file://-Aufruf ohne CORS) –
        // auf die eigenständige Seite ausweichen statt leere Schublade zeigen.
        location.href = url;
      });
    }

    function goBack() {
      menu.style.display = "flex";
      content.style.display = "none";
      backBtn.style.display = "none";
      title.textContent = "Kursmaterial und Referenz";

      // Behebt Focus-Trap-Verlust: Fokus zurück auf den Kachel-Link setzen
      if (lastClickedKey) {
        var btnToFocus = menu.querySelector('[data-target="' + lastClickedKey + '"]');
        if (btnToFocus) btnToFocus.focus();
        else sheet.focus();
      } else {
        sheet.focus();
      }
    }

    btn.addEventListener("click", open);
    if (closeBtn) closeBtn.addEventListener("click", close);
    if (backdrop) backdrop.addEventListener("click", close);
    if (backBtn) backBtn.addEventListener("click", goBack);

    overlay.querySelectorAll(".ro-link-btn").forEach(function (b) {
      b.addEventListener("click", function (e) {
        e.preventDefault();
        lastClickedKey = b.getAttribute("data-target");
        // b.href (DOM-Property, nicht getAttribute) liefert die vom Browser
        // bereits zu absolut aufgelöste URL – nötig als fetch()-Ziel und als
        // Basis, um relative Links/Bilder im geladenen Fragment aufzulösen.
        showDetail(lastClickedKey, b.href);
      });
    });

    addEventListener("keydown", function (e) {
      if (e.key === "Escape" && overlay.classList.contains("open")) {
        close();
      }
    });
  }

  /* ---------- Tastatur: ← / → zwischen Modulen -------------------------- */
  function initKeys() {
    addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var target = e.target;
      var t = target.tagName;
      if (t === "INPUT" || t === "TEXTAREA" || target.isContentEditable) return;
      // Behebt Tastatur-Navigationskonflikt: Keine Modulnavigation, wenn Fokus in scrollbaren/interaktiven Elementen liegt
      if (target.closest("pre") || target.closest("code") || target.closest("table") || target.closest("select") || target.closest("details")) return;
      if (e.key === "ArrowLeft") { var p = doc.querySelector(".weiter-nav a:not(.weiter-rechts)"); if (p && p.href) location.href = p.href; }
      if (e.key === "ArrowRight") { var n = doc.querySelector(".weiter-nav a.weiter-rechts"); if (n && n.href) location.href = n.href; }
    });
  }

  /* ---------- Merken, wenn ein Block MANUELL umgeschaltet wird
     Nicht am "toggle"-Ereignis hängen: das feuert der Browser laut Spezifikation
     verzögert und auch dann, wenn syncTracks() oder revealHashTarget() ein
     <details> selbst öffnen. Damit hätte sich jede Box, die der Tiefen-Schalter
     einmal aufgeklappt hat, sofort als "vom Menschen angefasst" markiert und
     wäre dem Schalter für den Rest der Sitzung entzogen. Nur ein echter Klick
     (oder Enter/Leertaste, beides löst click aus) auf die Zusammenfassung
     zählt als Handlung der Lesenden. */
  function initDetailsMemory() {
    doc.querySelectorAll("details.callout.adv, details.callout.basis, details.vertiefung")
      .forEach(function (d) {
        var summary = d.querySelector(":scope > summary");
        if (summary) summary.addEventListener("click", function () { d.dataset.touched = "1"; });
      });
  }

  /* ---------- Sprungziele in eingeklappten Bereichen erreichbar machen
     Das Inhaltsverzeichnis verlinkt auch Überschriften, die in einer
     zugeklappten <details>-Spur liegen (Modul 14, §7–§11). Chrome klappt beim
     Sprung von selbst auf, Firefox nicht: dort landet man wortlos am falschen
     Ort. Also selbst aufklappen – für jeden Vorfahren, nicht nur den ersten. */
  function revealHashTarget() {
    var id = location.hash.slice(1);
    if (!id) return;
    var ziel = doc.getElementById(id);
    if (!ziel) return;
    var offen = false;
    for (var el = ziel.parentElement; el; el = el.parentElement) {
      if (el.tagName === "DETAILS" && !el.open) {
        el.open = true;
        // Wer gezielt hierher springt, will den Abschnitt sehen. Ohne diese
        // Markierung würde ihn der nächste Griff zum Tiefen-Schalter wieder
        // zuklappen – und zwar den Abschnitt, den man gerade aufgerufen hat.
        el.dataset.touched = "1";
        offen = true;
      }
    }
    if (offen) ziel.scrollIntoView();
  }

  function initHashReveal() {
    window.addEventListener("hashchange", revealHashTarget);
    revealHashTarget();   // auch beim Laden mit vorhandenem #anchor
  }

  ready(function () {
    initNav(); initModuleMenu(); initModuleGroups(); initSettings(); initPrereq();
    initSegments(); initQuiz(); initProgress(); initPaths();
    initContinue();
    initCodeClamp(); initWhere(); initCopy(); initDataCopy(); initAnchors(); initSpy(); initScroll(); initLightbox();
    initKeys(); initDetailsMemory(); initHashReveal(); initChecklist(); initResourcesOverlay();
  });
})();
