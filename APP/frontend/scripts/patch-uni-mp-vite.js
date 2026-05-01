/**
 * Patch @dcloudio/uni-mp-vite to guard against non-JSON content in pagesJson transform.
 *
 * Root cause: Rollup 4.x calls the transform hook twice for the same pages-json-js
 * virtual module. The 2nd call receives the JS output of the 1st transform instead of
 * fresh JSON, causing parseMiniProgramPagesJson to crash with "pages parse failed".
 *
 * Fix: return null early if code doesn't start with '{'.
 */
const fs = require('fs');
const path = require('path');

const file = path.join(
  __dirname,
  '../node_modules/@dcloudio/uni-mp-vite/dist/plugins/pagesJson.js'
);

if (!fs.existsSync(file)) {
  console.warn('[patch] pagesJson.js not found, skipping patch.');
  process.exit(0);
}

let src = fs.readFileSync(file, 'utf8');

const GUARD = `if (!code.trimStart().startsWith('{')) {\n                    return null;\n                }\n                `;
const ANCHOR = `this.addWatchFile(path_1.default.resolve(inputDir, 'pages.json'));`;

if (src.includes(GUARD)) {
  console.log('[patch] pagesJson.js already patched.');
  process.exit(0);
}

if (!src.includes(ANCHOR)) {
  console.warn('[patch] Anchor not found in pagesJson.js, patch skipped.');
  process.exit(0);
}

src = src.replace(ANCHOR, GUARD + ANCHOR);
fs.writeFileSync(file, src);
console.log('[patch] pagesJson.js patched successfully.');
