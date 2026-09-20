Third-party assets, bundled so the UI works offline (no CDN at demo time).

| File | What | License |
|---|---|---|
| `marked.min.js` | marked 12.0.2, markdown renderer | MIT |
| `purify.min.js` | DOMPurify 3.1.6, HTML sanitiser for rendered markdown | Apache-2.0 / MPL-2.0 |
| `fonts/ibm-plex-*.woff2` | IBM Plex Sans and Mono (latin, 400/600), via @fontsource | SIL OFL 1.1 |

Home screen background (built from `frontend-src/shapewaves/`, run `npm install && npm run build` there):

| File | What | License |
|---|---|---|
| `shapewaves.js`, `shapewaves.css` | "Shape Waves" background from [React Bits](https://github.com/DavidHDev/react-bits), copied unmodified and bundled with React, React DOM and vgpu 0.3.1 (the version React Bits pins) | React Bits: MIT + Commons Clause (see `frontend-src/shapewaves/LICENSE-react-bits.md`); used here as part of this application, not redistributed on its own |
