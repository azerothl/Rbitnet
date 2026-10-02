const SITE_DOCS = new Set([
  "get-started",
  "usage",
  "status-and-roadmap",
  "native-first",
  "integrations",
  "env-reference",
  "limitations",
  "bitnet-native",
  "akasha-infer",
  "deployment",
  "brand",
  "benchmarks",
  "changelog",
  "demarrage-5min",
]);

const SLUG_ALIASES = {
  "get_started": "get-started",
  "get-started.md": "get-started",
  "usage.md": "usage",
  "status_and_roadmap": "status-and-roadmap",
  "status-and-roadmap.md": "status-and-roadmap",
  "native_first": "native-first",
  "native-first.md": "native-first",
  "integrations.md": "integrations",
  "env_reference": "env-reference",
  "env-reference.md": "env-reference",
  "limitations.md": "limitations",
  "bitnet_native": "bitnet-native",
  "bitnet-native.md": "bitnet-native",
  "akasha_infer": "akasha-infer",
  "akasha-infer.md": "akasha-infer",
  "deployment.md": "deployment",
  "brand.md": "brand",
  "benchmarks.md": "benchmarks",
  "changelog.md": "changelog",
  "demarrage_5min": "demarrage-5min",
  "demarrage-5min.md": "demarrage-5min",
  "GET_STARTED.md": "get-started",
  "USAGE.md": "usage",
  "STATUS_AND_ROADMAP.md": "status-and-roadmap",
  "NATIVE_FIRST.md": "native-first",
  "INTEGRATIONS.md": "integrations",
  "ENV_REFERENCE.md": "env-reference",
  "LIMITATIONS.md": "limitations",
  "BITNET_NATIVE.md": "bitnet-native",
  "AKASHA_INFER.md": "akasha-infer",
  "DEPLOYMENT.md": "deployment",
  "BRAND.md": "brand",
  "BENCHMARKS.md": "benchmarks",
  "CHANGELOG.md": "changelog",
  "DEMARRAGE_5MIN.md": "demarrage-5min",
};

function normalizeSlug(raw) {
  if (!raw) return "get-started";
  const key = decodeURIComponent(raw).trim();
  if (SLUG_ALIASES[key]) return SLUG_ALIASES[key];
  const bare = key.replace(/\.md$/i, "").toLowerCase().replace(/_/g, "-");
  return SLUG_ALIASES[bare] || bare;
}

function rewriteDocHref(href) {
  if (!href || href.startsWith("http") || href.startsWith("#") || href.startsWith("mailto:")) {
    return href;
  }
  const clean = href.split("#")[0].replace(/^\.\//, "").replace(/^\.\.\//, "");
  const base = clean.split("/").pop();
  if (!base) return href;
  const slug = normalizeSlug(base);
  if (SITE_DOCS.has(slug)) {
    const hash = href.includes("#") ? `#${href.split("#")[1]}` : "";
    return `page.html?doc=${encodeURIComponent(slug)}${hash}`;
  }
  return `https://github.com/azerothl/Rbitnet/blob/main/docs/${base}`;
}

async function loadManifestNav(active) {
  const aside = document.getElementById("docs-aside");
  if (!aside) return;
  try {
    const res = await fetch("./manifest.json");
    const manifest = await res.json();
    const nav = document.createElement("nav");
    for (const group of manifest.groups) {
      const label = document.createElement("p");
      label.className = "aside-label";
      label.textContent = group.title;
      nav.appendChild(label);
      for (const item of group.items) {
        const a = document.createElement("a");
        a.href = `page.html?doc=${encodeURIComponent(item.slug)}`;
        a.textContent = item.title;
        if (item.slug === active) a.classList.add("is-active");
        nav.appendChild(a);
      }
    }
    aside.appendChild(nav);
  } catch {
    aside.hidden = true;
  }
}

(async () => {
  const params = new URLSearchParams(window.location.search);
  const slug = normalizeSlug(params.get("doc"));
  const article = document.getElementById("docs-article");
  const slugEl = document.getElementById("docs-slug");
  const sourceEl = document.getElementById("docs-source");

  if (slugEl) slugEl.textContent = slug;
  await loadManifestNav(slug);

  if (!SITE_DOCS.has(slug)) {
    article.innerHTML = `<p class="docs-error">Unknown doc <code>${slug}</code>. <a href="./">Back to docs</a>.</p>`;
    return;
  }

  try {
    const res = await fetch(`./content/${slug}.md`);
    if (!res.ok) throw new Error(`${res.status}`);
    const md = await res.text();
    if (typeof marked === "undefined") throw new Error("marked unavailable");

    marked.use({
      renderer: {
        link({ href, title, text }) {
          const next = rewriteDocHref(href);
          const titleAttr = title ? ` title="${title}"` : "";
          const ext = next.startsWith("http") ? ' rel="noopener"' : "";
          return `<a href="${next}"${titleAttr}${ext}>${text}</a>`;
        },
      },
    });

    article.innerHTML = marked.parse(md);
    const h1 = article.querySelector("h1");
    document.title = `${h1 ? h1.textContent : slug} — Rbitnet docs`;
    if (sourceEl) {
      sourceEl.innerHTML = `Synced from repo docs · <a href="https://github.com/azerothl/Rbitnet/blob/main/docs/" rel="noopener">browse full tree</a>`;
    }

    if (location.hash) {
      const target = document.querySelector(location.hash);
      if (target) target.scrollIntoView();
    }
  } catch (err) {
    article.innerHTML = `<p class="docs-error">Failed to load <code>${slug}.md</code>. Run <code>website/scripts/sync-docs.sh</code> locally. (${String(err.message || err)})</p>`;
  }
})();
