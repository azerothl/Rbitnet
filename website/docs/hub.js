(async () => {
  const root = document.getElementById("docs-hub");
  if (!root) return;

  try {
    const res = await fetch("./manifest.json");
    if (!res.ok) throw new Error(`manifest ${res.status}`);
    const manifest = await res.json();
    root.innerHTML = "";

    for (const group of manifest.groups) {
      const section = document.createElement("section");
      section.className = "docs-group";
      const h2 = document.createElement("h2");
      h2.textContent = group.title;
      section.appendChild(h2);

      const ul = document.createElement("ul");
      ul.className = "docs-grid";
      for (const item of group.items) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = `page.html?doc=${encodeURIComponent(item.slug)}`;
        a.innerHTML = `<strong>${item.title}</strong><span>${item.blurb}</span>`;
        li.appendChild(a);
        ul.appendChild(li);
      }
      section.appendChild(ul);
      root.appendChild(section);
    }
  } catch (err) {
    root.innerHTML = `<p class="docs-error">Could not load docs index. (${String(err.message || err)})</p>`;
  }
})();
